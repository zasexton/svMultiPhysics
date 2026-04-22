/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Backends/Interfaces/DofPermutation.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/Types.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::backends {
namespace {

using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::DenseVectorView;
using svmp::FE::assembly::GlobalSystemView;
using svmp::FE::sparsity::DistributedSparsityPattern;
using svmp::FE::sparsity::IndexRange;

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

std::uint64_t allreduceSum(std::uint64_t local, MPI_Comm comm)
{
    std::uint64_t global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_SUM, comm);
    return global;
}

struct MaxDiff {
    Real max_abs{0.0};
    std::size_t idx{0};
    Real a{0.0};
    Real b{0.0};
};

MaxDiff maxAbsDiffWithIndex(std::span<const Real> a, std::span<const Real> b)
{
    MaxDiff d{};
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const Real da = a[i];
        const Real db = b[i];
        const Real m = static_cast<Real>(std::abs(da - db));
        if (m > d.max_abs) {
            d.max_abs = m;
            d.idx = i;
            d.a = da;
            d.b = db;
        }
    }
    return d;
}

std::vector<Real> gatherLocalMatrixEntries(const GenericMatrix& A, GlobalIndex n_global)
{
    std::vector<Real> local(static_cast<std::size_t>(n_global) * static_cast<std::size_t>(n_global), Real(0.0));
    for (GlobalIndex i = 0; i < n_global; ++i) {
        for (GlobalIndex j = 0; j < n_global; ++j) {
            local[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_global) + static_cast<std::size_t>(j)] =
                A.getEntry(i, j);
        }
    }
    return local;
}

std::vector<Real> gatherOwnedLocalMatrixEntries(const GenericMatrix& A, GlobalIndex n_global)
{
    const auto* fsils = dynamic_cast<const FsilsMatrix*>(&A);
    EXPECT_NE(fsils, nullptr);

    std::vector<Real> local(static_cast<std::size_t>(n_global) * static_cast<std::size_t>(n_global), Real(0.0));
    if (fsils == nullptr) {
        return local;
    }

    for (GlobalIndex i = 0; i < n_global; ++i) {
        if (!fsils->ownsFeDofRow(i)) {
            continue;
        }
        for (GlobalIndex j = 0; j < n_global; ++j) {
            local[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_global) + static_cast<std::size_t>(j)] =
                A.getEntry(i, j);
        }
    }
    return local;
}

std::vector<Real> gatherOwnedLocalMatrixEntries(const DenseMatrixView& A,
                                                const FsilsMatrix& layout,
                                                GlobalIndex n_global)
{
    std::vector<Real> local(static_cast<std::size_t>(n_global) * static_cast<std::size_t>(n_global), Real(0.0));
    const auto data = A.data();
    for (GlobalIndex i = 0; i < n_global; ++i) {
        if (!layout.ownsFeDofRow(i)) {
            continue;
        }
        for (GlobalIndex j = 0; j < n_global; ++j) {
            const auto idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(n_global) +
                             static_cast<std::size_t>(j);
            local[idx] = data[idx];
        }
    }
    return local;
}

std::vector<Real> gatherOwnedLocalVectorEntries(GenericVector& v, GlobalIndex n_global)
{
    auto view = v.createAssemblyView();
    const auto* fsils = dynamic_cast<const FsilsVector*>(&v);
    EXPECT_NE(fsils, nullptr);

    std::vector<Real> local(static_cast<std::size_t>(n_global), Real(0.0));
    if (fsils == nullptr) {
        return local;
    }

    for (GlobalIndex i = 0; i < n_global; ++i) {
        if (!fsils->ownsFeDof(i)) {
            continue;
        }
        local[static_cast<std::size_t>(i)] = view->getVectorEntry(i);
    }
    return local;
}

std::vector<Real> gatherOwnedLocalVectorEntries(const DenseVectorView& v,
                                                const FsilsVector& layout,
                                                GlobalIndex n_global)
{
    std::vector<Real> local(static_cast<std::size_t>(n_global), Real(0.0));
    const auto data = v.data();
    for (GlobalIndex i = 0; i < n_global; ++i) {
        if (!layout.ownsFeDof(i)) {
            continue;
        }
        local[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(i)];
    }
    return local;
}

std::vector<Real> makeScaledComponentTwoNodeMatrix(int dof_per_node, std::span<const Real> scales)
{
    const int dof = dof_per_node;
    FE_THROW_IF(dof <= 0, FEException, "makeScaledComponentTwoNodeMatrix: dof must be > 0");
    FE_THROW_IF(static_cast<int>(scales.size()) != dof, FEException,
                "makeScaledComponentTwoNodeMatrix: scales size must equal dof");

    const int edof = 2 * dof;
    std::vector<Real> Ke(static_cast<std::size_t>(edof) * static_cast<std::size_t>(edof), Real(0.0));

    // Per-component SPD 1D stiffness: alpha * [[ 2, -1], [-1, 2]].
    for (int c = 0; c < dof; ++c) {
        const Real a = scales[static_cast<std::size_t>(c)];
        const int i0 = c;       // node0 component c
        const int i1 = dof + c; // node1 component c
        Ke[static_cast<std::size_t>(i0 * edof + i0)] = 2.0 * a;
        Ke[static_cast<std::size_t>(i0 * edof + i1)] = -1.0 * a;
        Ke[static_cast<std::size_t>(i1 * edof + i0)] = -1.0 * a;
        Ke[static_cast<std::size_t>(i1 * edof + i1)] = 2.0 * a;
    }
    return Ke;
}

std::vector<Real> matVec(std::span<const Real> A, int n, std::span<const Real> x)
{
    std::vector<Real> y(static_cast<std::size_t>(n), Real(0.0));
    for (int i = 0; i < n; ++i) {
        Real s = 0.0;
        for (int j = 0; j < n; ++j) {
            s += A[static_cast<std::size_t>(i * n + j)] * x[static_cast<std::size_t>(j)];
        }
        y[static_cast<std::size_t>(i)] = s;
    }
    return y;
}

void assembleElement(GlobalSystemView& mat_view,
                     GlobalSystemView& vec_view,
                     std::span<const GlobalIndex> edofs,
                     std::span<const Real> Ke,
                     std::span<const Real> be)
{
    mat_view.addMatrixEntries(edofs, Ke, assembly::AddMode::Add);
    vec_view.addVectorEntries(edofs, be, assembly::AddMode::Add);
}

void assembleElementOwnedRows(GlobalSystemView& mat_view,
                              GlobalSystemView& vec_view,
                              std::span<const GlobalIndex> edofs,
                              std::span<const Real> Ke,
                              std::span<const Real> be,
                              const std::function<int(GlobalIndex)>& row_owner,
                              int rank)
{
    const auto n = static_cast<GlobalIndex>(edofs.size());
    FE_THROW_IF(Ke.size() != static_cast<std::size_t>(n * n) ||
                    be.size() != static_cast<std::size_t>(n),
                FEException,
                "assembleElementOwnedRows: element matrix/vector size mismatch");

    std::vector<GlobalIndex> owned_rows;
    std::vector<Real> owned_matrix;
    std::vector<Real> owned_vector;
    owned_rows.reserve(edofs.size());
    owned_matrix.reserve(Ke.size());
    owned_vector.reserve(be.size());

    for (GlobalIndex i = 0; i < n; ++i) {
        const auto row = edofs[static_cast<std::size_t>(i)];
        if (row_owner(row) != rank) {
            continue;
        }
        owned_rows.push_back(row);
        owned_vector.push_back(be[static_cast<std::size_t>(i)]);
        const auto row_begin = Ke.begin() + static_cast<std::ptrdiff_t>(i * n);
        owned_matrix.insert(owned_matrix.end(),
                            row_begin,
                            row_begin + static_cast<std::ptrdiff_t>(n));
    }

    if (!owned_rows.empty()) {
        mat_view.addMatrixEntries(std::span<const GlobalIndex>(owned_rows),
                                  edofs,
                                  std::span<const Real>(owned_matrix),
                                  assembly::AddMode::Add);
        vec_view.addVectorEntries(std::span<const GlobalIndex>(owned_rows),
                                  std::span<const Real>(owned_vector),
                                  assembly::AddMode::Add);
    }
}

} // namespace

TEST(FsilsAssemblyParityMPI, OwnedRowDirectAssemblyMatchesDenseWithoutPermutation)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    // Node-block FE ordering (identity permutation): [u0 v0 p0 | u1 v1 p1 | u2 v2 p2]
    const IndexRange owned = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs = {0, 1, 2, 3, 4, 5};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    } else {
        const std::array<GlobalIndex, 6> edofs = {3, 4, 5, 6, 7, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    // Ghost rows for shared node 1 on rank 0 provide the input-vector halo.
    // Direct backend views intentionally keep matrix/RHS rows owner-only.
    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 4, 5};
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        for (int r = 0; r < dof; ++r) {
            for (GlobalIndex c = 0; c < 2 * dof; ++c) {
                ghost_cols.push_back(c);
            }
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    const std::array<Real, dof> scales = {2.0, 3.0, 5.0};
    const auto Ke = makeScaledComponentTwoNodeMatrix(dof, scales);
    const std::vector<Real> ones(6, 1.0);
    const auto be = matVec(Ke, /*n=*/6, ones);

    FsilsFactory factory(dof);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);
    auto x = factory.createVector(n_global);

    A->zero();
    b->zero();
    x->zero();

    auto viewA = A->createAssemblyView();
    auto viewb = b->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewb->beginAssemblyPhase();
    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs = {0, 1, 2, 3, 4, 5};
        assembleElement(*viewA, *viewb, edofs, Ke, be);
    } else {
        const std::array<GlobalIndex, 6> edofs = {3, 4, 5, 6, 7, 8};
        assembleElement(*viewA, *viewb, edofs, Ke, be);
    }
    viewA->finalizeAssembly();
    viewb->finalizeAssembly();
    A->finalizeAssembly();

    DenseMatrixView Ad(n_global);
    DenseVectorView bd(n_global);
    Ad.zero();
    bd.zero();
    Ad.beginAssemblyPhase();
    bd.beginAssemblyPhase();
    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs = {0, 1, 2, 3, 4, 5};
        assembleElement(Ad, bd, edofs, Ke, be);
    } else {
        const std::array<GlobalIndex, 6> edofs = {3, 4, 5, 6, 7, 8};
        assembleElement(Ad, bd, edofs, Ke, be);
    }
    Ad.finalizeAssembly();
    bd.finalizeAssembly();

    const auto* A_fs = dynamic_cast<const FsilsMatrix*>(A.get());
    ASSERT_NE(A_fs, nullptr);
    const auto* b_fs = dynamic_cast<FsilsVector*>(b.get());
    ASSERT_NE(b_fs, nullptr);

    const auto dense_local_A = gatherOwnedLocalMatrixEntries(Ad, *A_fs, n_global);
    const auto dense_local_b = gatherOwnedLocalVectorEntries(bd, *b_fs, n_global);
    const auto dense_global_A = allreduceSum(dense_local_A, comm);
    const auto dense_global_b = allreduceSum(dense_local_b, comm);

    const auto fsils_local_A = gatherOwnedLocalMatrixEntries(*A, n_global);
    const auto fsils_local_b = gatherOwnedLocalVectorEntries(*b, n_global);
    const auto fsils_global_A = allreduceSum(fsils_local_A, comm);
    const auto fsils_global_b = allreduceSum(fsils_local_b, comm);

    if (rank == 0) {
        constexpr Real tol = 1e-14;
        const auto dA = maxAbsDiffWithIndex(dense_global_A, fsils_global_A);
        const auto db = maxAbsDiffWithIndex(dense_global_b, fsils_global_b);
        EXPECT_LT(dA.max_abs, tol) << "Max |A_dense-A_fsils|=" << dA.max_abs
                                  << " at idx=" << dA.idx
                                  << " (dense=" << dA.a << ", fsils=" << dA.b << ")";
        EXPECT_LT(db.max_abs, tol) << "Max |b_dense-b_fsils|=" << db.max_abs
                                  << " at idx=" << db.idx
                                  << " (dense=" << db.a << ", fsils=" << db.b << ")";
    }

    // Solve and verify x ~= 1 in FE index space.
    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    x->updateGhosts();
    auto* fx = dynamic_cast<FsilsVector*>(x.get());
    ASSERT_TRUE(fx);
    const auto* shared = fx->shared();
    ASSERT_TRUE(shared);

    const auto perm = shared->dof_permutation;
    auto viewx = x->createAssemblyView();
    std::vector<int> local_nodes;
    local_nodes.reserve(static_cast<std::size_t>(shared->owned_node_count) + shared->ghost_nodes.size());
    for (int i = 0; i < shared->owned_node_count; ++i) {
        local_nodes.push_back(shared->owned_node_start + i);
    }
    local_nodes.insert(local_nodes.end(), shared->ghost_nodes.begin(), shared->ghost_nodes.end());

    for (const int node : local_nodes) {
        for (int c = 0; c < dof; ++c) {
            const GlobalIndex backend_dof = static_cast<GlobalIndex>(node) * dof + c;
            const GlobalIndex fe_dof = (perm && !perm->empty())
                                           ? perm->inverse[static_cast<std::size_t>(backend_dof)]
                                           : backend_dof;
            EXPECT_NEAR(viewx->getVectorEntry(fe_dof), 1.0, 1e-10);
        }
    }
}

TEST(FsilsAssemblyParityMPI, OwnedRowDirectAssemblyMatchesDenseWithPermutation)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    // FE ordering (component-blocked): [u0 u1 u2 | v0 v1 v2 | p0 p1 p2]
    // Backend ordering (node-block):   [u0 v0 p0 | u1 v1 p1 | u2 v2 p2]
    auto perm = std::make_shared<DofPermutation>();
    perm->forward.resize(static_cast<std::size_t>(n_global));
    perm->inverse.resize(static_cast<std::size_t>(n_global));
    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        const GlobalIndex u = node;
        const GlobalIndex v = n_nodes + node;
        const GlobalIndex p = 2 * n_nodes + node;
        const GlobalIndex backend_base = node * dof;
        perm->forward[static_cast<std::size_t>(u)] = backend_base + 0;
        perm->forward[static_cast<std::size_t>(v)] = backend_base + 1;
        perm->forward[static_cast<std::size_t>(p)] = backend_base + 2;
    }
    for (GlobalIndex fe = 0; fe < n_global; ++fe) {
        const GlobalIndex be = perm->forward[static_cast<std::size_t>(fe)];
        perm->inverse[static_cast<std::size_t>(be)] = fe;
    }
    for (GlobalIndex i = 0; i < n_global; ++i) {
        ASSERT_EQ(perm->inverse[static_cast<std::size_t>(perm->forward[static_cast<std::size_t>(i)])], i);
    }

    // FSILS distributed sparsity must be specified in backend (node-block) ordering.
    const IndexRange owned_backend = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned_backend, owned_backend, n_global, n_global);
    pattern.setDofIndexing(DistributedSparsityPattern::DofIndexing::NodalInterleaved);

    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs_backend = {0, 1, 2, 3, 4, 5};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs_backend.data(), edofs_backend.size()));
    } else {
        const std::array<GlobalIndex, 6> edofs_backend = {3, 4, 5, 6, 7, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs_backend.data(), edofs_backend.size()));
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 4, 5};
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        for (int r = 0; r < dof; ++r) {
            for (GlobalIndex c = 0; c < 2 * dof; ++c) {
                ghost_cols.push_back(c);
            }
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    const std::array<Real, dof> scales = {2.0, 3.0, 5.0};
    const auto Ke = makeScaledComponentTwoNodeMatrix(dof, scales);
    const std::vector<Real> ones(6, 1.0);
    const auto be = matVec(Ke, /*n=*/6, ones);

    FsilsFactory factory(dof, perm);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);
    auto x = factory.createVector(n_global);

    A->zero();
    b->zero();
    x->zero();

    // Element DOFs in FE index space, but ordered per-node within the element.
    const auto u0 = GlobalIndex(0);
    const auto u1 = GlobalIndex(1);
    const auto u2 = GlobalIndex(2);
    const auto v0 = GlobalIndex(3);
    const auto v1 = GlobalIndex(4);
    const auto v2 = GlobalIndex(5);
    const auto p0 = GlobalIndex(6);
    const auto p1 = GlobalIndex(7);
    const auto p2 = GlobalIndex(8);

    auto viewA = A->createAssemblyView();
    auto viewb = b->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewb->beginAssemblyPhase();
    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs_fe = {u0, v0, p0, u1, v1, p1};
        assembleElement(*viewA, *viewb, edofs_fe, Ke, be);
    } else {
        const std::array<GlobalIndex, 6> edofs_fe = {u1, v1, p1, u2, v2, p2};
        assembleElement(*viewA, *viewb, edofs_fe, Ke, be);
    }
    viewA->finalizeAssembly();
    viewb->finalizeAssembly();
    A->finalizeAssembly();

    DenseMatrixView Ad(n_global);
    DenseVectorView bd(n_global);
    Ad.zero();
    bd.zero();
    Ad.beginAssemblyPhase();
    bd.beginAssemblyPhase();
    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs_fe = {u0, v0, p0, u1, v1, p1};
        assembleElement(Ad, bd, edofs_fe, Ke, be);
    } else {
        const std::array<GlobalIndex, 6> edofs_fe = {u1, v1, p1, u2, v2, p2};
        assembleElement(Ad, bd, edofs_fe, Ke, be);
    }
    Ad.finalizeAssembly();
    bd.finalizeAssembly();

    const auto* A_fs = dynamic_cast<const FsilsMatrix*>(A.get());
    ASSERT_NE(A_fs, nullptr);
    const auto* b_fs = dynamic_cast<FsilsVector*>(b.get());
    ASSERT_NE(b_fs, nullptr);

    const auto dense_local_A = gatherOwnedLocalMatrixEntries(Ad, *A_fs, n_global);
    const auto dense_local_b = gatherOwnedLocalVectorEntries(bd, *b_fs, n_global);
    const auto dense_global_A = allreduceSum(dense_local_A, comm);
    const auto dense_global_b = allreduceSum(dense_local_b, comm);

    const auto fsils_local_A = gatherOwnedLocalMatrixEntries(*A, n_global);
    const auto fsils_local_b = gatherOwnedLocalVectorEntries(*b, n_global);
    const auto fsils_global_A = allreduceSum(fsils_local_A, comm);
    const auto fsils_global_b = allreduceSum(fsils_local_b, comm);

    if (rank == 0) {
        constexpr Real tol = 1e-14;
        const auto dA = maxAbsDiffWithIndex(dense_global_A, fsils_global_A);
        const auto db = maxAbsDiffWithIndex(dense_global_b, fsils_global_b);
        EXPECT_LT(dA.max_abs, tol) << "Max |A_dense-A_fsils|=" << dA.max_abs
                                  << " at idx=" << dA.idx
                                  << " (dense=" << dA.a << ", fsils=" << dA.b << ")";
        EXPECT_LT(db.max_abs, tol) << "Max |b_dense-b_fsils|=" << db.max_abs
                                  << " at idx=" << db.idx
                                  << " (dense=" << db.a << ", fsils=" << db.b << ")";
    }

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    x->updateGhosts();
    auto* fx = dynamic_cast<FsilsVector*>(x.get());
    ASSERT_TRUE(fx);
    const auto* shared = fx->shared();
    ASSERT_TRUE(shared);

    const auto perm_x = shared->dof_permutation;
    auto viewx = x->createAssemblyView();
    std::vector<int> local_nodes;
    local_nodes.reserve(static_cast<std::size_t>(shared->owned_node_count) + shared->ghost_nodes.size());
    for (int i = 0; i < shared->owned_node_count; ++i) {
        local_nodes.push_back(shared->owned_node_start + i);
    }
    local_nodes.insert(local_nodes.end(), shared->ghost_nodes.begin(), shared->ghost_nodes.end());

    for (const int node : local_nodes) {
        for (int c = 0; c < dof; ++c) {
            const GlobalIndex backend_dof = static_cast<GlobalIndex>(node) * dof + c;
            const GlobalIndex fe_dof = (perm_x && !perm_x->empty())
                                           ? perm_x->inverse[static_cast<std::size_t>(backend_dof)]
                                           : backend_dof;
            EXPECT_NEAR(viewx->getVectorEntry(fe_dof), 1.0, 1e-10);
        }
    }
}

TEST(FsilsAssemblyParityMPI, OwnedRowDirectAssemblyMatchesDenseWithPermutationNaturalIndexing)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    // FE ordering (rank-contiguous, but not node-interleaved on rank 1):
    //   [u0 v0 p0 | u1 u2 v1 v2 p1 p2]
    // Backend ordering (node-block):
    //   [u0 v0 p0 | u1 v1 p1 | u2 v2 p2]
    //
    // This exercises the FSILS distributed layout builder when the distributed sparsity is in FE
    // index space, requiring the DOF permutation to construct the node adjacency graph.
    auto perm = std::make_shared<DofPermutation>();
    perm->forward.resize(static_cast<std::size_t>(n_global));
    perm->inverse.resize(static_cast<std::size_t>(n_global));

    // FE indices:
    // 0:u0 1:v0 2:p0 3:u1 4:u2 5:v1 6:v2 7:p1 8:p2
    // Backend indices:
    // 0:u0 1:v0 2:p0 3:u1 4:v1 5:p1 6:u2 7:v2 8:p2
    perm->forward[0] = 0;
    perm->forward[1] = 1;
    perm->forward[2] = 2;
    perm->forward[3] = 3;
    perm->forward[4] = 6;
    perm->forward[5] = 4;
    perm->forward[6] = 7;
    perm->forward[7] = 5;
    perm->forward[8] = 8;

    for (GlobalIndex fe = 0; fe < n_global; ++fe) {
        const GlobalIndex be = perm->forward[static_cast<std::size_t>(fe)];
        perm->inverse[static_cast<std::size_t>(be)] = fe;
    }
    for (GlobalIndex i = 0; i < n_global; ++i) {
        ASSERT_EQ(perm->inverse[static_cast<std::size_t>(perm->forward[static_cast<std::size_t>(i)])], i);
    }

    // Distributed sparsity is specified in FE index space with rank-contiguous ownership.
    const IndexRange owned_fe = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned_fe, owned_fe, n_global, n_global);
    pattern.setDofIndexing(DistributedSparsityPattern::DofIndexing::Natural);

    if (rank == 0) {
        // Element between nodes 0-1 in FE index space: [u0 v0 p0 | u1 v1 p1] -> {0,1,2,3,5,7}
        const std::array<GlobalIndex, 6> edofs_fe = {0, 1, 2, 3, 5, 7};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs_fe.data(), edofs_fe.size()));
    } else {
        // Element between nodes 1-2 in FE index space: [u1 v1 p1 | u2 v2 p2] -> {3,5,7,4,6,8}
        const std::array<GlobalIndex, 6> edofs_fe = {3, 5, 7, 4, 6, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs_fe.data(), edofs_fe.size()));
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    // Ghost rows for shared node 1 on rank 0, in FE index space.
    // Direct backend views intentionally keep matrix/RHS rows owner-only.
    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 5, 7};
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        const std::array<GlobalIndex, 6> cols = {0, 1, 2, 3, 5, 7};
        for (int r = 0; r < dof; ++r) {
            ghost_cols.insert(ghost_cols.end(), cols.begin(), cols.end());
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    const std::array<Real, dof> scales = {2.0, 3.0, 5.0};
    const auto Ke = makeScaledComponentTwoNodeMatrix(dof, scales);
    const std::vector<Real> ones(6, 1.0);
    const auto be = matVec(Ke, /*n=*/6, ones);

    FsilsFactory factory(dof, perm);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);
    auto x = factory.createVector(n_global);

    A->zero();
    b->zero();
    x->zero();

    auto viewA = A->createAssemblyView();
    auto viewb = b->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewb->beginAssemblyPhase();
    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs_fe = {0, 1, 2, 3, 5, 7};
        assembleElement(*viewA, *viewb, edofs_fe, Ke, be);
    } else {
        const std::array<GlobalIndex, 6> edofs_fe = {3, 5, 7, 4, 6, 8};
        assembleElement(*viewA, *viewb, edofs_fe, Ke, be);
    }
    viewA->finalizeAssembly();
    viewb->finalizeAssembly();
    A->finalizeAssembly();

    DenseMatrixView Ad(n_global);
    DenseVectorView bd(n_global);
    Ad.zero();
    bd.zero();
    Ad.beginAssemblyPhase();
    bd.beginAssemblyPhase();
    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs_fe = {0, 1, 2, 3, 5, 7};
        assembleElement(Ad, bd, edofs_fe, Ke, be);
    } else {
        const std::array<GlobalIndex, 6> edofs_fe = {3, 5, 7, 4, 6, 8};
        assembleElement(Ad, bd, edofs_fe, Ke, be);
    }
    Ad.finalizeAssembly();
    bd.finalizeAssembly();

    const auto* A_fs = dynamic_cast<const FsilsMatrix*>(A.get());
    ASSERT_NE(A_fs, nullptr);
    const auto* b_fs = dynamic_cast<FsilsVector*>(b.get());
    ASSERT_NE(b_fs, nullptr);

    const auto dense_local_A = gatherOwnedLocalMatrixEntries(Ad, *A_fs, n_global);
    const auto dense_local_b = gatherOwnedLocalVectorEntries(bd, *b_fs, n_global);
    const auto dense_global_A = allreduceSum(dense_local_A, comm);
    const auto dense_global_b = allreduceSum(dense_local_b, comm);

    const auto fsils_local_A = gatherOwnedLocalMatrixEntries(*A, n_global);
    const auto fsils_local_b = gatherOwnedLocalVectorEntries(*b, n_global);
    const auto fsils_global_A = allreduceSum(fsils_local_A, comm);
    const auto fsils_global_b = allreduceSum(fsils_local_b, comm);

    if (rank == 0) {
        constexpr Real tol = 1e-14;
        const auto dA = maxAbsDiffWithIndex(dense_global_A, fsils_global_A);
        const auto db = maxAbsDiffWithIndex(dense_global_b, fsils_global_b);
        EXPECT_LT(dA.max_abs, tol) << "Max |A_dense-A_fsils|=" << dA.max_abs
                                  << " at idx=" << dA.idx
                                  << " (dense=" << dA.a << ", fsils=" << dA.b << ")";
        EXPECT_LT(db.max_abs, tol) << "Max |b_dense-b_fsils|=" << db.max_abs
                                  << " at idx=" << db.idx
                                  << " (dense=" << db.a << ", fsils=" << db.b << ")";
    }

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    x->updateGhosts();
    auto* fx = dynamic_cast<FsilsVector*>(x.get());
    ASSERT_TRUE(fx);
    const auto* shared = fx->shared();
    ASSERT_TRUE(shared);

    const auto perm_x = shared->dof_permutation;
    auto viewx = x->createAssemblyView();
    std::vector<int> local_nodes;
    local_nodes.reserve(static_cast<std::size_t>(shared->owned_node_count) + shared->ghost_nodes.size());
    for (int i = 0; i < shared->owned_node_count; ++i) {
        local_nodes.push_back(shared->owned_node_start + i);
    }
    local_nodes.insert(local_nodes.end(), shared->ghost_nodes.begin(), shared->ghost_nodes.end());

    for (const int node : local_nodes) {
        for (int c = 0; c < dof; ++c) {
            const GlobalIndex backend_dof = static_cast<GlobalIndex>(node) * dof + c;
            const GlobalIndex fe_dof = (perm_x && !perm_x->empty())
                                           ? perm_x->inverse[static_cast<std::size_t>(backend_dof)]
                                           : backend_dof;
            EXPECT_NEAR(viewx->getVectorEntry(fe_dof), 1.0, 1e-10);
        }
    }
}

TEST(FsilsAssemblyParityMPI, OwnedRowDirectAssemblyMatchesDenseWithPermutationNaturalIndexing_NormalizesInverse)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    // FE ordering (rank-contiguous, but not node-interleaved on rank 1):
    //   [u0 v0 p0 | u1 u2 v1 v2 p1 p2]
    // Backend ordering (node-block):
    //   [u0 v0 p0 | u1 v1 p1 | u2 v2 p2]
    auto perm = std::make_shared<DofPermutation>();
    perm->forward.resize(static_cast<std::size_t>(n_global));
    perm->inverse.resize(static_cast<std::size_t>(n_global));

    perm->forward[0] = 0;
    perm->forward[1] = 1;
    perm->forward[2] = 2;
    perm->forward[3] = 3;
    perm->forward[4] = 6;
    perm->forward[5] = 4;
    perm->forward[6] = 7;
    perm->forward[7] = 5;
    perm->forward[8] = 8;

    // Intentionally corrupt inverse (still a valid permutation) to ensure the FSILS layout builder
    // recomputes a consistent inverse from the forward mapping.
    for (GlobalIndex be = 0; be < n_global; ++be) {
        perm->inverse[static_cast<std::size_t>(be)] = (be + 1) % n_global;
    }

    const IndexRange owned_fe = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned_fe, owned_fe, n_global, n_global);
    pattern.setDofIndexing(DistributedSparsityPattern::DofIndexing::Natural);

    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs_fe = {0, 1, 2, 3, 5, 7};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs_fe.data(), edofs_fe.size()));
    } else {
        const std::array<GlobalIndex, 6> edofs_fe = {3, 5, 7, 4, 6, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs_fe.data(), edofs_fe.size()));
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 5, 7};
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        const std::array<GlobalIndex, 6> cols = {0, 1, 2, 3, 5, 7};
        for (int r = 0; r < dof; ++r) {
            ghost_cols.insert(ghost_cols.end(), cols.begin(), cols.end());
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    const std::array<Real, dof> scales = {2.0, 3.0, 5.0};
    const auto Ke = makeScaledComponentTwoNodeMatrix(dof, scales);
    const std::vector<Real> ones(6, 1.0);
    const auto be = matVec(Ke, /*n=*/6, ones);

    FsilsFactory factory(dof, perm);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);
    auto x = factory.createVector(n_global);

    A->zero();
    b->zero();
    x->zero();

    auto viewA = A->createAssemblyView();
    auto viewb = b->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewb->beginAssemblyPhase();
    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs_fe = {0, 1, 2, 3, 5, 7};
        assembleElement(*viewA, *viewb, edofs_fe, Ke, be);
    } else {
        const std::array<GlobalIndex, 6> edofs_fe = {3, 5, 7, 4, 6, 8};
        assembleElement(*viewA, *viewb, edofs_fe, Ke, be);
    }
    viewA->finalizeAssembly();
    viewb->finalizeAssembly();
    A->finalizeAssembly();

    DenseMatrixView Ad(n_global);
    DenseVectorView bd(n_global);
    Ad.zero();
    bd.zero();
    Ad.beginAssemblyPhase();
    bd.beginAssemblyPhase();
    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs_fe = {0, 1, 2, 3, 5, 7};
        assembleElement(Ad, bd, edofs_fe, Ke, be);
    } else {
        const std::array<GlobalIndex, 6> edofs_fe = {3, 5, 7, 4, 6, 8};
        assembleElement(Ad, bd, edofs_fe, Ke, be);
    }
    Ad.finalizeAssembly();
    bd.finalizeAssembly();

    const auto* A_fs = dynamic_cast<const FsilsMatrix*>(A.get());
    ASSERT_NE(A_fs, nullptr);
    const auto* b_fs = dynamic_cast<FsilsVector*>(b.get());
    ASSERT_NE(b_fs, nullptr);

    const auto dense_local_A = gatherOwnedLocalMatrixEntries(Ad, *A_fs, n_global);
    const auto dense_local_b = gatherOwnedLocalVectorEntries(bd, *b_fs, n_global);
    const auto dense_global_A = allreduceSum(dense_local_A, comm);
    const auto dense_global_b = allreduceSum(dense_local_b, comm);

    const auto fsils_local_A = gatherOwnedLocalMatrixEntries(*A, n_global);
    const auto fsils_local_b = gatherOwnedLocalVectorEntries(*b, n_global);
    const auto fsils_global_A = allreduceSum(fsils_local_A, comm);
    const auto fsils_global_b = allreduceSum(fsils_local_b, comm);

    if (rank == 0) {
        constexpr Real tol = 1e-14;
        const auto dA = maxAbsDiffWithIndex(dense_global_A, fsils_global_A);
        const auto db = maxAbsDiffWithIndex(dense_global_b, fsils_global_b);
        EXPECT_LT(dA.max_abs, tol) << "Max |A_dense-A_fsils|=" << dA.max_abs
                                  << " at idx=" << dA.idx
                                  << " (dense=" << dA.a << ", fsils=" << dA.b << ")";
        EXPECT_LT(db.max_abs, tol) << "Max |b_dense-b_fsils|=" << db.max_abs
                                  << " at idx=" << db.idx
                                  << " (dense=" << db.a << ", fsils=" << db.b << ")";
    }

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    x->updateGhosts();
    auto* fx = dynamic_cast<FsilsVector*>(x.get());
    ASSERT_TRUE(fx);
    const auto* shared = fx->shared();
    ASSERT_TRUE(shared);

    const auto perm_x = shared->dof_permutation;
    auto viewx = x->createAssemblyView();
    std::vector<int> local_nodes;
    local_nodes.reserve(static_cast<std::size_t>(shared->owned_node_count) + shared->ghost_nodes.size());
    for (int i = 0; i < shared->owned_node_count; ++i) {
        local_nodes.push_back(shared->owned_node_start + i);
    }
    local_nodes.insert(local_nodes.end(), shared->ghost_nodes.begin(), shared->ghost_nodes.end());

    for (const int node : local_nodes) {
        for (int c = 0; c < dof; ++c) {
            const GlobalIndex backend_dof = static_cast<GlobalIndex>(node) * dof + c;
            const GlobalIndex fe_dof = (perm_x && !perm_x->empty())
                                           ? perm_x->inverse[static_cast<std::size_t>(backend_dof)]
                                           : backend_dof;
            EXPECT_NEAR(viewx->getVectorEntry(fe_dof), 1.0, 1e-10);
        }
    }
}

TEST(FsilsAssemblyParityMPI, MixedAuxOwnerMapDrivesSparsityAndNumericAssembly)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 4; // three field components plus one native auxiliary component
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex field_unknowns = 3 * n_nodes;
    constexpr GlobalIndex n_global = dof * n_nodes;

    auto perm = std::make_shared<DofPermutation>();
    perm->forward.assign(static_cast<std::size_t>(n_global), INVALID_GLOBAL_INDEX);
    perm->inverse.assign(static_cast<std::size_t>(n_global), INVALID_GLOBAL_INDEX);
    perm->owner_rank.assign(static_cast<std::size_t>(n_global), -1);

    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        for (int c = 0; c < dof; ++c) {
            const GlobalIndex fe = static_cast<GlobalIndex>(c) * n_nodes + node;
            const GlobalIndex be = node * dof + c;
            perm->forward[static_cast<std::size_t>(fe)] = be;
            perm->inverse[static_cast<std::size_t>(be)] = fe;
            perm->owner_rank[static_cast<std::size_t>(be)] = (node == 0) ? 0 : 1;
        }
    }

    MixedBlockLayout layout;
    layout.field_unknowns = field_unknowns;
    layout.auxiliary_unknowns = n_nodes;
    layout.total_unknowns = n_global;

    MixedBlockDescriptor field;
    field.name = "field";
    field.offset = 0;
    field.size = field_unknowns;
    field.kind = MixedBlockKind::Field;
    field.node_component_start = 0;
    field.node_component_count = 3;
    layout.blocks.push_back(field);

    MixedBlockDescriptor aux;
    aux.name = "node_aux";
    aux.offset = field_unknowns;
    aux.size = n_nodes;
    aux.kind = MixedBlockKind::Auxiliary;
    aux.role = BlockRole::AuxiliaryField;
    aux.assembly_mode = MixedBlockAssemblyMode::NativeOwnedRows;
    aux.row_ownership = MixedRowOwnershipPolicy::BackendDofOwner;
    aux.row_owner_ranks = {0, 1, 1};
    aux.node_component_start = 3;
    aux.node_component_count = 1;
    layout.blocks.push_back(aux);

    const std::function<int(GlobalIndex)> row_owner = [&](GlobalIndex fe_row) -> int {
        const int mixed_owner = layout.ownerRankForGlobalRow(fe_row);
        if (mixed_owner >= 0) {
            return mixed_owner;
        }
        if (fe_row < 0 || fe_row >= n_global ||
            static_cast<std::size_t>(fe_row) >= perm->forward.size()) {
            return -1;
        }
        const GlobalIndex backend_row = perm->forward[static_cast<std::size_t>(fe_row)];
        if (backend_row < 0 ||
            static_cast<std::size_t>(backend_row) >= perm->owner_rank.size()) {
            return -1;
        }
        return perm->owner_rank[static_cast<std::size_t>(backend_row)];
    };

    const IndexRange owned_backend = (rank == 0) ? IndexRange{0, 4} : IndexRange{4, 12};
    DistributedSparsityPattern pattern(owned_backend, owned_backend, n_global, n_global);
    pattern.setDofIndexing(DistributedSparsityPattern::DofIndexing::NodalInterleaved);

    const std::array<GlobalIndex, 8> elem01_fe = {0, 3, 6, 9, 1, 4, 7, 10};
    const std::array<GlobalIndex, 8> elem12_fe = {1, 4, 7, 10, 2, 5, 8, 11};

    auto add_owned_pattern = [&](const std::array<GlobalIndex, 8>& edofs_fe) {
        for (const auto row_fe : edofs_fe) {
            if (row_owner(row_fe) != rank) {
                continue;
            }
            const auto row_be = perm->forward[static_cast<std::size_t>(row_fe)];
            for (const auto col_fe : edofs_fe) {
                const auto col_be = perm->forward[static_cast<std::size_t>(col_fe)];
                pattern.addEntry(row_be, col_be);
            }
        }
    };
    add_owned_pattern(elem01_fe);
    add_owned_pattern(elem12_fe);
    pattern.ensureDiagonal();
    pattern.finalize();

    auto set_ghost_node = [&](GlobalIndex node, GlobalIndex first_col, GlobalIndex last_col_exclusive) {
        std::vector<GlobalIndex> ghost_rows;
        std::vector<GlobalIndex> ghost_row_ptr;
        std::vector<GlobalIndex> ghost_cols;
        ghost_rows.reserve(static_cast<std::size_t>(dof));
        ghost_row_ptr.reserve(static_cast<std::size_t>(dof) + 1u);
        ghost_row_ptr.push_back(0);
        for (int c = 0; c < dof; ++c) {
            ghost_rows.push_back(node * dof + c);
            for (GlobalIndex col = first_col; col < last_col_exclusive; ++col) {
                ghost_cols.push_back(col);
            }
            ghost_row_ptr.push_back(static_cast<GlobalIndex>(ghost_cols.size()));
        }
        pattern.setGhostRows(std::move(ghost_rows),
                             std::move(ghost_row_ptr),
                             std::move(ghost_cols));
    };
    if (rank == 0) {
        set_ghost_node(/*node=*/1, /*first_col=*/0, /*last_col_exclusive=*/8);
    } else {
        set_ghost_node(/*node=*/0, /*first_col=*/0, /*last_col_exclusive=*/8);
    }

    const std::array<Real, dof> scales = {2.0, 3.0, 5.0, 7.0};
    const auto Ke = makeScaledComponentTwoNodeMatrix(dof, scales);
    const std::vector<Real> ones(8, 1.0);
    const auto be = matVec(Ke, /*n=*/8, ones);

    FsilsFactory factory(dof, perm);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);
    A->zero();
    b->zero();
    FsilsMatrix::resetDroppedEntryCount();
    FsilsMatrix::resetOffOwnerWriteCount();

    auto* A_fs = dynamic_cast<FsilsMatrix*>(A.get());
    ASSERT_NE(A_fs, nullptr);
    auto* b_fs = dynamic_cast<FsilsVector*>(b.get());
    ASSERT_NE(b_fs, nullptr);
    EXPECT_TRUE(A_fs->usesOwnedRowOperator());
    EXPECT_TRUE(b_fs->usesOwnedRowLayout());
    for (GlobalIndex fe = 0; fe < n_global; ++fe) {
        const bool owned_here = row_owner(fe) == rank;
        EXPECT_EQ(A_fs->ownsFeDofRow(fe), owned_here) << "fe row " << fe;
        EXPECT_EQ(b_fs->ownsFeDof(fe), owned_here) << "fe dof " << fe;
    }

    auto viewA = A->createAssemblyView();
    auto viewb = b->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewb->beginAssemblyPhase();
    assembleElementOwnedRows(*viewA, *viewb, elem01_fe, Ke, be, row_owner, rank);
    assembleElementOwnedRows(*viewA, *viewb, elem12_fe, Ke, be, row_owner, rank);
    viewA->finalizeAssembly();
    viewb->finalizeAssembly();
    A->finalizeAssembly();

    DenseMatrixView Ad(n_global);
    DenseVectorView bd(n_global);
    Ad.zero();
    bd.zero();
    Ad.beginAssemblyPhase();
    bd.beginAssemblyPhase();
    assembleElementOwnedRows(Ad, bd, elem01_fe, Ke, be, row_owner, rank);
    assembleElementOwnedRows(Ad, bd, elem12_fe, Ke, be, row_owner, rank);
    Ad.finalizeAssembly();
    bd.finalizeAssembly();

    const auto off_owner_global = allreduceSum(FsilsMatrix::offOwnerWriteCount(), comm);
    const auto dropped_global = allreduceSum(FsilsMatrix::droppedEntryCount(), comm);
    if (rank == 0) {
        EXPECT_EQ(off_owner_global, 0u);
        EXPECT_EQ(dropped_global, 0u);
    }

    const auto dense_local_A = gatherOwnedLocalMatrixEntries(Ad, *A_fs, n_global);
    const auto dense_local_b = gatherOwnedLocalVectorEntries(bd, *b_fs, n_global);
    const auto dense_global_A = allreduceSum(dense_local_A, comm);
    const auto dense_global_b = allreduceSum(dense_local_b, comm);

    const auto fsils_local_A = gatherOwnedLocalMatrixEntries(*A, n_global);
    const auto fsils_local_b = gatherOwnedLocalVectorEntries(*b, n_global);
    const auto fsils_global_A = allreduceSum(fsils_local_A, comm);
    const auto fsils_global_b = allreduceSum(fsils_local_b, comm);

    if (rank == 0) {
        constexpr Real tol = 1e-14;
        const auto dA = maxAbsDiffWithIndex(dense_global_A, fsils_global_A);
        const auto db = maxAbsDiffWithIndex(dense_global_b, fsils_global_b);
        EXPECT_LT(dA.max_abs, tol) << "Max |A_dense-A_fsils|=" << dA.max_abs
                                  << " at idx=" << dA.idx
                                  << " (dense=" << dA.a << ", fsils=" << dA.b << ")";
        EXPECT_LT(db.max_abs, tol) << "Max |b_dense-b_fsils|=" << db.max_abs
                                  << " at idx=" << db.idx
                                  << " (dense=" << db.a << ", fsils=" << db.b << ")";
    }
}

TEST(FsilsAssemblyParityMPI, VectorUpdateGhostsIsOwnerAuthoritativeAndRawAccumulationIsExplicit)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr GlobalIndex n_global = 3;
    const IndexRange owned = (rank == 0) ? IndexRange{0, 1} : IndexRange{1, 3};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    pattern.setDofIndexing(DistributedSparsityPattern::DofIndexing::NodalInterleaved);
    if (rank == 0) {
        pattern.addEntry(0, 0);
        pattern.addEntry(0, 1);
    } else {
        pattern.addEntry(1, 0);
        pattern.addEntry(1, 1);
        pattern.addEntry(1, 2);
        pattern.addEntry(2, 1);
        pattern.addEntry(2, 2);
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        pattern.setGhostRows(std::vector<GlobalIndex>{1},
                             std::vector<GlobalIndex>{0, 2},
                             std::vector<GlobalIndex>{0, 1});
    } else {
        pattern.setGhostRows(std::vector<GlobalIndex>{0},
                             std::vector<GlobalIndex>{0, 2},
                             std::vector<GlobalIndex>{0, 1});
    }

    FsilsFactory factory(/*dof_per_node=*/1);
    auto matrix_layout = factory.createMatrix(pattern);
    (void)matrix_layout;
    auto vec = factory.createVector(n_global);
    auto* fs = dynamic_cast<FsilsVector*>(vec.get());
    ASSERT_NE(fs, nullptr);
    ASSERT_TRUE(fs->usesOwnedRowLayout());

    auto view = vec->createAssemblyView();

    vec->set(-100.0);
    if (rank == 0) {
        view->setVectorEntries(std::array<GlobalIndex, 1>{0},
                               std::array<Real, 1>{10.0});
        EXPECT_FALSE(fs->ownsFeDof(1));
        EXPECT_NEAR(view->getVectorEntry(1), -100.0, 1e-14);
    } else {
        view->setVectorEntries(std::array<GlobalIndex, 2>{1, 2},
                               std::array<Real, 2>{20.0, 30.0});
        EXPECT_TRUE(fs->ownsFeDof(1));
    }

    vec->updateGhosts();
    if (rank == 0) {
        EXPECT_NEAR(view->getVectorEntry(1), 20.0, 1e-14)
            << "ordinary vectors must copy owner rows to ghosts";
    } else {
        EXPECT_NEAR(view->getVectorEntry(1), 20.0, 1e-14);
    }

    vec->zero();
    if (rank == 0) {
        view->addVectorEntries(std::array<GlobalIndex, 2>{0, 1},
                               std::array<Real, 2>{10.0, 5.0},
                               assembly::AddMode::Add);
    } else {
        view->addVectorEntries(std::array<GlobalIndex, 2>{1, 2},
                               std::array<Real, 2>{7.0, 11.0},
                               assembly::AddMode::Add);
    }

    vec->updateGhosts();
    if (rank == 0) {
        EXPECT_NEAR(view->getVectorEntry(1), 7.0, 1e-14)
            << "updateGhosts must not reverse-scatter raw ghost contributions";
    } else {
        EXPECT_NEAR(view->getVectorEntry(1), 7.0, 1e-14);
    }

    vec->zero();
    if (rank == 0) {
        view->addVectorEntries(std::array<GlobalIndex, 2>{0, 1},
                               std::array<Real, 2>{10.0, 5.0},
                               assembly::AddMode::Add);
    } else {
        view->addVectorEntries(std::array<GlobalIndex, 2>{1, 2},
                               std::array<Real, 2>{7.0, 11.0},
                               assembly::AddMode::Add);
    }

    fs->accumulateRawContributionsAndUpdateGhosts();
    if (rank == 0) {
        EXPECT_NEAR(view->getVectorEntry(1), 12.0, 1e-14)
            << "raw contribution buffers must explicitly reverse-scatter before ghost sync";
    } else {
        EXPECT_NEAR(view->getVectorEntry(1), 12.0, 1e-14);
    }
}

} // namespace svmp::FE::backends

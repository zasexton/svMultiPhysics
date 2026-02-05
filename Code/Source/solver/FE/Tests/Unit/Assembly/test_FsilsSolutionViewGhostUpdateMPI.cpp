/**
 * @file test_FsilsSolutionViewGhostUpdateMPI.cpp
 * @brief Regression: assembly must read correct ghost values from backend vectors (FSILS overlap + DOF permutation).
 *
 * Motivation: MPI-only nonlinear convergence issues can arise if the current solution
 * (or history) vectors have stale ghost entries when Forms kernels evaluate element
 * residuals/Jacobians. Systems wires backend vectors via SystemStateView::u_vector,
 * which StandardAssembler consumes through GlobalSystemView::getVectorEntry.
 *
 * This test verifies that:
 * - Assembly using a FSILS overlap vector view matches a dense span reference once
 *   FsilsVector::updateGhosts() has been called.
 * - Without updateGhosts(), stale ghost values change the assembled residual/Jacobian.
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Backends/Interfaces/DofPermutation.h"
#include "Dofs/DofMap.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Sparsity/DistributedSparsityPattern.h"
#include "Spaces/SpaceFactory.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::assembly::testing {
namespace {

using svmp::FE::backends::DofPermutation;
using svmp::FE::backends::FsilsFactory;
using svmp::FE::backends::FsilsVector;
using svmp::FE::dofs::DofMap;
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

Real maxAbsDiff(std::span<const Real> a, std::span<const Real> b)
{
    const std::size_t n = std::min(a.size(), b.size());
    Real m = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        m = std::max(m, static_cast<Real>(std::abs(a[i] - b[i])));
    }
    return m;
}

// 2D strip of Quad4 cells, with interleaved node IDs:
// x-index i has nodes {2*i (bottom), 2*i+1 (top)}.
class StripQuadMeshAccess final : public IMeshAccess {
public:
    StripQuadMeshAccess(int n_cells, int my_rank)
        : n_cells_(n_cells),
          my_rank_(my_rank)
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
    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        // We use a simple model: cell c is owned by rank c (requires n_cells == world_size in the test).
        return 1;
    }
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

DofMap buildComponentBlockedDofMap(int n_cells, int dof_per_node)
{
    const GlobalIndex n_nodes = static_cast<GlobalIndex>(2 * (n_cells + 1));
    const GlobalIndex n_dofs = n_nodes * static_cast<GlobalIndex>(dof_per_node);
    const LocalIndex dofs_per_cell = static_cast<LocalIndex>(4 * dof_per_node);

    DofMap map(static_cast<GlobalIndex>(n_cells), n_dofs, dofs_per_cell);
    map.setNumDofs(n_dofs);

    for (int c = 0; c < n_cells; ++c) {
        const std::array<GlobalIndex, 4> nodes = {
            static_cast<GlobalIndex>(2 * c + 0),
            static_cast<GlobalIndex>(2 * (c + 1) + 0),
            static_cast<GlobalIndex>(2 * (c + 1) + 1),
            static_cast<GlobalIndex>(2 * c + 1),
        };

        std::vector<GlobalIndex> dofs;
        dofs.reserve(static_cast<std::size_t>(dofs_per_cell));
        for (int comp = 0; comp < dof_per_node; ++comp) {
            const GlobalIndex comp_base = static_cast<GlobalIndex>(comp) * n_nodes;
            for (const auto node : nodes) {
                dofs.push_back(comp_base + node);
            }
        }
        map.setCellDofs(static_cast<GlobalIndex>(c), dofs);
    }

    map.finalize();
    return map;
}

std::shared_ptr<const DofPermutation> makeComponentToNodeBlockPermutation(GlobalIndex n_nodes, int dof_per_node)
{
    const GlobalIndex n_dofs = n_nodes * static_cast<GlobalIndex>(dof_per_node);
    auto perm = std::make_shared<DofPermutation>();
    perm->forward.resize(static_cast<std::size_t>(n_dofs));
    perm->inverse.resize(static_cast<std::size_t>(n_dofs));

    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        for (int comp = 0; comp < dof_per_node; ++comp) {
            const GlobalIndex fe = static_cast<GlobalIndex>(comp) * n_nodes + node;
            const GlobalIndex backend = node * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(comp);
            perm->forward[static_cast<std::size_t>(fe)] = backend;
            perm->inverse[static_cast<std::size_t>(backend)] = fe;
        }
    }
    return perm;
}

DistributedSparsityPattern buildFsilsPatternForStrip(MPI_Comm comm,
                                                     int rank,
                                                     int size,
                                                     int dof_per_node)
{
    FE_THROW_IF(size < 2, InvalidArgumentException, "buildFsilsPatternForStrip: size must be >= 2");
    const int n_cells = size;
    const GlobalIndex n_nodes = static_cast<GlobalIndex>(2 * (n_cells + 1));
    const GlobalIndex n_dofs = n_nodes * static_cast<GlobalIndex>(dof_per_node);

    // Node ownership ranges (contiguous):
    // - rank r < size-1 owns x=r nodes: [2r, 2r+2)
    // - last rank owns x=size-1 and x=size: [2(size-1), 2(size+1))
    const GlobalIndex owned_node_start = static_cast<GlobalIndex>(2 * rank);
    const GlobalIndex owned_node_count = (rank == size - 1) ? 4 : 2;
    const IndexRange owned_rows{
        owned_node_start * dof_per_node,
        (owned_node_start + owned_node_count) * dof_per_node,
    };

    DistributedSparsityPattern pattern(owned_rows, owned_rows, n_dofs, n_dofs);

    // Element (cell) owned by this rank: c = rank, with nodes at x=rank and x=rank+1.
    const int c = rank;
    const std::array<GlobalIndex, 4> cell_nodes = {
        static_cast<GlobalIndex>(2 * c + 0),
        static_cast<GlobalIndex>(2 * (c + 1) + 0),
        static_cast<GlobalIndex>(2 * (c + 1) + 1),
        static_cast<GlobalIndex>(2 * c + 1),
    };

    std::vector<GlobalIndex> edofs_backend;
    edofs_backend.reserve(static_cast<std::size_t>(4 * dof_per_node));
    for (const auto node : cell_nodes) {
        for (int comp = 0; comp < dof_per_node; ++comp) {
            edofs_backend.push_back(node * dof_per_node + comp);
        }
    }

    pattern.addElementCouplings(edofs_backend);
    pattern.ensureDiagonal();
    pattern.finalize();

    // Ghost rows for the "right" x=rank+1 node pair on all ranks except the last.
    if (rank != size - 1) {
        const std::array<GlobalIndex, 2> ghost_nodes = {
            static_cast<GlobalIndex>(2 * (rank + 1) + 0),
            static_cast<GlobalIndex>(2 * (rank + 1) + 1),
        };

        std::vector<GlobalIndex> ghost_rows;
        ghost_rows.reserve(static_cast<std::size_t>(ghost_nodes.size() * static_cast<std::size_t>(dof_per_node)));
        for (const auto gn : ghost_nodes) {
            for (int comp = 0; comp < dof_per_node; ++comp) {
                ghost_rows.push_back(gn * dof_per_node + comp);
            }
        }
        std::sort(ghost_rows.begin(), ghost_rows.end());

        // Column closure: all DOFs of the current cell (4 nodes * dof).
        std::vector<GlobalIndex> cell_cols = edofs_backend;
        std::sort(cell_cols.begin(), cell_cols.end());
        cell_cols.erase(std::unique(cell_cols.begin(), cell_cols.end()), cell_cols.end());

        const std::size_t cols_per_row = cell_cols.size();
        std::vector<GlobalIndex> ghost_row_ptr;
        ghost_row_ptr.resize(ghost_rows.size() + 1u);
        ghost_row_ptr[0] = 0;
        for (std::size_t r = 0; r < ghost_rows.size(); ++r) {
            ghost_row_ptr[r + 1] = static_cast<GlobalIndex>((r + 1u) * cols_per_row);
        }

        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(ghost_rows.size() * cols_per_row);
        for (std::size_t r = 0; r < ghost_rows.size(); ++r) {
            ghost_cols.insert(ghost_cols.end(), cell_cols.begin(), cell_cols.end());
        }

        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    // Avoid unused warning in non-MPI builds.
    (void)comm;
    return pattern;
}

struct GlobalAssembly {
    std::vector<Real> J;
    std::vector<Real> R;
};

GlobalAssembly assembleNonlinear(StripQuadMeshAccess& mesh,
                                 const spaces::FunctionSpace& space,
                                 const DofMap& dof_map,
                                 forms::NonlinearFormKernel& kernel,
                                 std::span<const Real> U,
                                 const GlobalSystemView* U_view,
                                 MPI_Comm comm)
{
    const GlobalIndex n_dofs = dof_map.getNumDofs();
    DenseMatrixView J_local(n_dofs);
    DenseVectorView R_local(n_dofs);
    J_local.zero();
    R_local.zero();

    StandardAssembler assembler;
    AssemblyOptions opts;
    opts.ghost_policy = GhostPolicy::ReverseScatter;
    opts.deterministic = true;
    opts.overlap_communication = false;
    assembler.setOptions(opts);
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);
    assembler.setCurrentSolutionView(U_view);

    (void)assembler.assembleBoth(mesh, space, space, kernel, J_local, R_local);
    assembler.finalize(&J_local, &R_local);

    return {
        allreduceSum(J_local.data(), comm),
        allreduceSum(R_local.data(), comm),
    };
}

GlobalAssembly assembleTransient(StripQuadMeshAccess& mesh,
                                 const spaces::FunctionSpace& space,
                                 const DofMap& dof_map,
                                 forms::NonlinearFormKernel& kernel,
                                 const TimeIntegrationContext& time,
                                 Real dt,
                                 std::span<const Real> U,
                                 const GlobalSystemView* U_view,
                                 std::span<const Real> U_prev,
                                 const GlobalSystemView* U_prev_view,
                                 std::span<const Real> U_prev2,
                                 const GlobalSystemView* U_prev2_view,
                                 MPI_Comm comm)
{
    const GlobalIndex n_dofs = dof_map.getNumDofs();
    DenseMatrixView J_local(n_dofs);
    DenseVectorView R_local(n_dofs);
    J_local.zero();
    R_local.zero();

    StandardAssembler assembler;
    AssemblyOptions opts;
    opts.ghost_policy = GhostPolicy::ReverseScatter;
    opts.deterministic = true;
    opts.overlap_communication = false;
    assembler.setOptions(opts);
    assembler.setDofMap(dof_map);
    assembler.setTimeIntegrationContext(&time);
    assembler.setTimeStep(dt);

    assembler.setCurrentSolution(U);
    assembler.setCurrentSolutionView(U_view);

    // Provide spans (possibly empty) and views (possibly null). If views are non-null,
    // StandardAssembler uses them to fetch global-indexed entries.
    assembler.setPreviousSolution(U_prev);
    assembler.setPreviousSolution2(U_prev2);
    assembler.setPreviousSolutionView(U_prev_view);
    assembler.setPreviousSolution2View(U_prev2_view);

    (void)assembler.assembleBoth(mesh, space, space, kernel, J_local, R_local);
    assembler.finalize(&J_local, &R_local);

    return {
        allreduceSum(J_local.data(), comm),
        allreduceSum(R_local.data(), comm),
        };
}

} // namespace

TEST(FsilsSolutionViewGhostUpdateMPI, StaleGhostValuesChangeAssemblyUntilUpdateGhosts)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    // This test uses a strip mesh with n_cells == world_size, with cell c owned by rank c.
    const int n_cells = size;
    StripQuadMeshAccess mesh(n_cells, rank);

    constexpr int dof = 3; // matches the (u,v,p) dof-per-node typical for 2D incompressible flow
    const GlobalIndex n_nodes = static_cast<GlobalIndex>(2 * (n_cells + 1));
    const GlobalIndex n_dofs = n_nodes * dof;

    const auto space = spaces::VectorSpace(spaces::SpaceType::H1, ElementType::Quad4, /*order=*/1, /*components=*/dof);
    ASSERT_TRUE(space);

    // Nonlinear but smooth: f(u) = (1 + ||u||^2) * u.
    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(*space, "u");
    const auto v = forms::TestFunction(*space, "v");
    const auto one = forms::FormExpr::constant(Real(1.0));
    const auto residual = ((one + forms::inner(u, u)) * forms::inner(u, v)).dx();

    auto ir = compiler.compileResidual(residual);
    forms::NonlinearFormKernel kernel(std::move(ir), forms::ADMode::Forward, forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    const auto dof_map = buildComponentBlockedDofMap(n_cells, dof);
    ASSERT_EQ(dof_map.getNumDofs(), n_dofs);

    // Global reference solution vector in FE ordering (component-blocked).
    std::vector<Real> U_ref(static_cast<std::size_t>(n_dofs), Real(0.0));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U_ref[static_cast<std::size_t>(i)] = Real(0.01) * static_cast<Real>(i + 1);
    }

    // Assemble reference using the dense span (no ghosting needed).
    const auto ref = assembleNonlinear(mesh, *space, dof_map, kernel, U_ref, /*U_view=*/nullptr, comm);

    // Build FSILS overlap vector layout (backend ordering) + FE<->backend permutation.
    const auto perm = makeComponentToNodeBlockPermutation(n_nodes, dof);
    auto pattern = buildFsilsPatternForStrip(comm, rank, size, dof);

    FsilsFactory factory(dof, perm);
    auto A_layout = factory.createMatrix(pattern);
    (void)A_layout;
    auto U = factory.createVector(n_dofs);

    auto* fU = dynamic_cast<FsilsVector*>(U.get());
    ASSERT_TRUE(fU);
    const auto* shared = fU->shared();
    ASSERT_TRUE(shared);
    ASSERT_TRUE(shared->dof_permutation);

    // Initialize local storage: set all local entries (owned + ghosts) to a wrong value,
    // then overwrite owned nodes with the correct values.
    std::fill(U->localSpan().begin(), U->localSpan().end(), Real(2.0));
    {
        auto view_set = U->createAssemblyView();
        view_set->beginAssemblyPhase();
        for (int i = 0; i < shared->owned_node_count; ++i) {
            const GlobalIndex node = static_cast<GlobalIndex>(shared->owned_node_start + i);
            for (int comp = 0; comp < dof; ++comp) {
                const GlobalIndex backend_dof = node * dof + comp;
                const GlobalIndex fe_dof = perm->inverse[static_cast<std::size_t>(backend_dof)];
                view_set->addVectorEntry(fe_dof, U_ref[static_cast<std::size_t>(fe_dof)], AddMode::Insert);
            }
        }
        view_set->finalizeAssembly();
    }

    // Assemble with stale ghosts (no updateGhosts).
    auto view_bad = U->createAssemblyView();
    const auto bad = assembleNonlinear(mesh, *space, dof_map, kernel, /*U=*/{}, view_bad.get(), comm);

    // Now update ghosts and assemble again.
    U->updateGhosts();
    auto view_good = U->createAssemblyView();
    const auto good = assembleNonlinear(mesh, *space, dof_map, kernel, /*U=*/{}, view_good.get(), comm);

    if (rank == 0) {
        const Real diff_bad_J = maxAbsDiff(ref.J, bad.J);
        const Real diff_bad_R = maxAbsDiff(ref.R, bad.R);
        const Real diff_good_J = maxAbsDiff(ref.J, good.J);
        const Real diff_good_R = maxAbsDiff(ref.R, good.R);

        // Bad assembly should differ noticeably (stale ghosts used in element evaluation).
        EXPECT_GT(diff_bad_J, 1e-6);
        EXPECT_GT(diff_bad_R, 1e-6);

        // After updateGhosts, assembly matches the dense span reference.
        EXPECT_LT(diff_good_J, 1e-10);
        EXPECT_LT(diff_good_R, 1e-10);
    }
}

TEST(FsilsSolutionViewGhostUpdateMPI, HistoryViewsRequireUpdateGhostsForDtTerms)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const int n_cells = size;
    StripQuadMeshAccess mesh(n_cells, rank);

    constexpr int dof = 3;
    const GlobalIndex n_nodes = static_cast<GlobalIndex>(2 * (n_cells + 1));
    const GlobalIndex n_dofs = n_nodes * dof;

    const auto space = spaces::VectorSpace(spaces::SpaceType::H1, ElementType::Quad4, /*order=*/1, /*components=*/dof);
    ASSERT_TRUE(space);

    // BDF2 stencil for u.dt(1): (3 u^n - 4 u^{n-1} + u^{n-2}) / (2 dt).
    constexpr Real dt = 0.1;
    TimeIntegrationContext time;
    time.integrator_name = "BDF2";
    time.dt1 = TimeDerivativeStencil{};
    time.dt1->order = 1;
    time.dt1->a = {Real(3.0) / (Real(2.0) * dt),
                   Real(-4.0) / (Real(2.0) * dt),
                   Real(1.0) / (Real(2.0) * dt)};

    forms::FormCompiler compiler;
    const auto u = forms::TrialFunction(*space, "u");
    const auto v = forms::TestFunction(*space, "v");
    const auto one = forms::FormExpr::constant(Real(1.0));
    const auto up = forms::FormExpr::previousSolution(1);

    // Make the Jacobian depend on u^{n-1} via a smooth weight.
    const auto weight = one + forms::inner(up, up);
    const auto residual = (weight * forms::inner(u.dt(1), v)).dx();

    auto ir = compiler.compileResidual(residual);
    forms::NonlinearFormKernel kernel(std::move(ir), forms::ADMode::Forward, forms::NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    const auto dof_map = buildComponentBlockedDofMap(n_cells, dof);
    ASSERT_EQ(dof_map.getNumDofs(), n_dofs);

    std::vector<Real> U0(static_cast<std::size_t>(n_dofs), Real(0.0));
    std::vector<Real> U1(static_cast<std::size_t>(n_dofs), Real(0.0));
    std::vector<Real> U2(static_cast<std::size_t>(n_dofs), Real(0.0));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U0[static_cast<std::size_t>(i)] = Real(0.01) * static_cast<Real>(i + 1);
        U1[static_cast<std::size_t>(i)] = Real(-0.005) * static_cast<Real>(i + 1);
        U2[static_cast<std::size_t>(i)] = Real(0.002) * static_cast<Real>(i + 1);
    }

    const auto ref = assembleTransient(mesh, *space, dof_map, kernel, time, dt,
                                       U0, /*U_view=*/nullptr,
                                       U1, /*U_prev_view=*/nullptr,
                                       U2, /*U_prev2_view=*/nullptr,
                                       comm);

    const auto perm = makeComponentToNodeBlockPermutation(n_nodes, dof);
    auto pattern = buildFsilsPatternForStrip(comm, rank, size, dof);

    FsilsFactory factory(dof, perm);
    auto A_layout = factory.createMatrix(pattern);
    (void)A_layout;

    auto U0v = factory.createVector(n_dofs);
    auto U1v = factory.createVector(n_dofs);
    auto U2v = factory.createVector(n_dofs);

    auto* fU0 = dynamic_cast<FsilsVector*>(U0v.get());
    ASSERT_TRUE(fU0);
    const auto* shared = fU0->shared();
    ASSERT_TRUE(shared);

    auto initOwnedOnly = [&](backends::GenericVector& vec, std::span<const Real> values) {
        std::fill(vec.localSpan().begin(), vec.localSpan().end(), Real(2.0));
        auto view = vec.createAssemblyView();
        view->beginAssemblyPhase();
        for (int i = 0; i < shared->owned_node_count; ++i) {
            const GlobalIndex node = static_cast<GlobalIndex>(shared->owned_node_start + i);
            for (int comp = 0; comp < dof; ++comp) {
                const GlobalIndex backend_dof = node * dof + comp;
                const GlobalIndex fe_dof = perm->inverse[static_cast<std::size_t>(backend_dof)];
                view->addVectorEntry(fe_dof, values[static_cast<std::size_t>(fe_dof)], AddMode::Insert);
            }
        }
        view->finalizeAssembly();
    };

    initOwnedOnly(*U0v, U0);
    initOwnedOnly(*U1v, U1);
    initOwnedOnly(*U2v, U2);

    // Ensure current state has correct ghosts; leave history ghosts stale to isolate the check.
    U0v->updateGhosts();

    auto view_u = U0v->createAssemblyView();
    auto view_p1 = U1v->createAssemblyView();
    auto view_p2 = U2v->createAssemblyView();

    const auto bad = assembleTransient(mesh, *space, dof_map, kernel, time, dt,
                                       /*U=*/{}, view_u.get(),
                                       /*U_prev=*/{}, view_p1.get(),
                                       /*U_prev2=*/{}, view_p2.get(),
                                       comm);

    U1v->updateGhosts();
    U2v->updateGhosts();
    auto view_u_ok = U0v->createAssemblyView();
    auto view_p1_ok = U1v->createAssemblyView();
    auto view_p2_ok = U2v->createAssemblyView();

    const auto good = assembleTransient(mesh, *space, dof_map, kernel, time, dt,
                                        /*U=*/{}, view_u_ok.get(),
                                        /*U_prev=*/{}, view_p1_ok.get(),
                                        /*U_prev2=*/{}, view_p2_ok.get(),
                                        comm);

    if (rank == 0) {
        const Real diff_bad_J = maxAbsDiff(ref.J, bad.J);
        const Real diff_bad_R = maxAbsDiff(ref.R, bad.R);
        const Real diff_good_J = maxAbsDiff(ref.J, good.J);
        const Real diff_good_R = maxAbsDiff(ref.R, good.R);

        EXPECT_GT(diff_bad_J, 1e-6);
        EXPECT_GT(diff_bad_R, 1e-6);
        EXPECT_LT(diff_good_J, 1e-10);
        EXPECT_LT(diff_good_R, 1e-10);
    }
}

} // namespace svmp::FE::assembly::testing

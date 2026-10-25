/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GlobalMultiplierMPI.cpp
 * @brief G1, G2 and G3 on 1, 2 and 4 ranks (G4); G1 under the bordered lowering on FSILS (G5)
 *
 * G1 (mean-constrained Poisson) runs on a strip of Q1 cells; G2 (pure Neumann
 * with a mean-zero multiplier) on P1 triangles and G3 (Taylor-Hood Stokes with
 * a mean-zero pressure multiplier) on P2/P1 triangles of the unit square, with
 * the same checks.  For G1:
 *
 * The mesh is a strip of Q1 cells along x on (0,1) x (0,b), seen by every
 * rank; cell c belongs to rank floor(c * size / N).  Every rank also builds
 * the same system on MPI_COMM_SELF with every cell its own: the one-rank
 * reference.  The checks:
 *
 *   - the residual entry of the global row, assembled through the ordinary
 *     parallel path, equals the one-rank entry bit for bit (exact sum);
 *   - assembled into exact accumulators, the matrix and vector equal the
 *     one-rank ones bit for bit, so lambda equals the one-rank lambda bit for
 *     bit: each cell's contribution reaches the global row once, at any rank
 *     count (the run at 2 and at 4 ranks each compare with one rank);
 *   - through the ordinary parallel path (double sums of the shared entries),
 *     lambda agrees with the one-rank lambda to round-off, and with the closed
 *     form 12c / (1 - h^2) of Q1.
 *
 * G5 solves G1 under the bordered lowering on FSILS GMRES (the field block on
 * FSILS, the multiplier through Newton's bordered correction) and compares
 * lambda and u with the one-rank in-matrix solution of Eigen's direct solver,
 * to 1e-10 relative.  FSILS takes the owner-contiguous numbering with cell
 * ownership, as in the FSILS tests (on this strip, whose cells every rank
 * sees, FSILS does not reach the solution at 4 ranks with GlobalIds and
 * vertex ownership, with or without a multiplier).
 */

#include <gtest/gtest.h>

#include <mpi.h>

#include "Assembly/Assembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Basis/LagrangeBasis.h"
#include "Elements/ReferenceElement.h"
#include "Geometry/MappingFactory.h"
#include "Spaces/ProductSpace.h"
#include "Constraints/DirichletBC.h"
#include "Constraints/VertexDirichletConstraint.h"
#include "Core/FEException.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/Vocabulary.h"
#include "Spaces/GlobalSpace.h"
#include "Spaces/H1Space.h"
#include "Systems/ExactAccumulator.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/SystemCollectives.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"
#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/TimeHistory.h"

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Backends/Utils/BackendOptions.h"
#if defined(FE_HAS_FSILS)
#include "Backends/FSILS/FsilsFactory.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::systems::test_global_multiplier_mpi {
namespace {

using forms::FormExpr;

constexpr int kCells = 12;
constexpr Real kHeight = 0.5;
constexpr Real kMean = 1.5;

[[nodiscard]] int ownerOf(int cell, int size)
{
    return (cell * size) / kCells;
}

/// The strip; every rank sees every cell, cell c belongs to ownerOf(c).
class StripMeshAccess final : public assembly::IMeshAccess {
public:
    StripMeshAccess(int rank, int size)
        : rank_(rank)
        , size_(size)
    {
        for (int c = 0; c < kCells; ++c) {
            if (ownerOf(c, size_) == rank_) {
                owned_.push_back(c);
            }
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return kCells; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return static_cast<GlobalIndex>(owned_.size()); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        return ownerOf(static_cast<int>(cell), size_) == rank_;
    }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Quad4; }

    void getCellNodes(GlobalIndex cell, std::vector<GlobalIndex>& nodes) const override
    {
        const auto c = static_cast<GlobalIndex>(cell);
        nodes = {2 * c, 2 * (c + 1), 2 * (c + 1) + 1, 2 * c + 1};
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node) const override
    {
        const auto i = node / 2;
        const auto j = node % 2;
        return {static_cast<Real>(i) / static_cast<Real>(kCells), kHeight * static_cast<Real>(j), 0.0};
    }

    void getCellCoordinates(GlobalIndex cell, std::vector<std::array<Real, 3>>& coords) const override
    {
        std::vector<GlobalIndex> nodes;
        getCellNodes(cell, nodes);
        coords.resize(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            coords[i] = getNodeCoordinates(nodes[i]);
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (int c = 0; c < kCells; ++c) {
            callback(c);
        }
    }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (const auto c : owned_) {
            callback(c);
        }
    }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    int rank_{0};
    int size_{1};
    std::vector<GlobalIndex> owned_;
};

dofs::MeshTopologyInfo stripTopology(int rank, int size)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = kCells;
    topo.n_vertices = 2 * (kCells + 1);
    topo.dim = 2;
    topo.cell2vertex_offsets.push_back(0);
    for (int c = 0; c < kCells; ++c) {
        const auto base = static_cast<MeshIndex>(2 * c);
        topo.cell2vertex_data.insert(topo.cell2vertex_data.end(),
                                     {base, static_cast<MeshIndex>(base + 2),
                                      static_cast<MeshIndex>(base + 3),
                                      static_cast<MeshIndex>(base + 1)});
        topo.cell2vertex_offsets.push_back(static_cast<MeshOffset>(topo.cell2vertex_data.size()));
        topo.cell_gids.push_back(c);
        topo.cell_owner_ranks.push_back(ownerOf(c, size));
    }
    for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
        topo.vertex_gids.push_back(v);
    }
    for (int r = 0; r < size; ++r) {
        if (r != rank) {
            topo.neighbor_ranks.push_back(r);
        }
    }
    return topo;
}

struct Problem {
    std::shared_ptr<spaces::H1Space> u_space;
    std::shared_ptr<spaces::GlobalSpace> r1;
    std::unique_ptr<FESystem> system;
    FieldId u{INVALID_FIELD_ID};
    FieldId lambda{INVALID_FIELD_ID};
};

/// G1 on @p comm; u = 0 on the vertices of x = 0 and x = 1 (DOF = vertex for Q1 here).
Problem buildProblem(MPI_Comm comm, int rank, int size,
                     MultiplierLowering lowering = MultiplierLowering::Auto,
                     bool owner_contiguous = false)
{
    Problem out;
    auto mesh = std::make_shared<StripMeshAccess>(rank, size);
    out.u_space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    out.r1 = std::make_shared<spaces::GlobalSpace>(1);
    out.system = std::make_unique<FESystem>(mesh);
    auto& sys = *out.system;
    out.u = sys.addField(FieldSpec{.name = "u", .space = out.u_space});
    out.lambda = sys.addField(FieldSpec{.name = "lambda", .space = out.r1, .scope = FieldScope::Global});
    sys.addOperator("poisson");
    const auto u = forms::StateField(out.u, *out.u_space, "u");
    const auto v = forms::TestField(out.u, *out.u_space, "v");
    const auto lam = forms::StateField(out.lambda, *out.r1, "lambda");
    const auto mu = forms::TestField(out.lambda, *out.r1, "mu");
    (void)installFormulation(sys, "poisson", {out.u, out.lambda},
                             (forms::inner(forms::grad(u), forms::grad(v)) - lam * v).dx() -
                                 (mu * u).dx() + (mu * FormExpr::constant(kMean * kHeight)).dGlobal());
    // The DOF of a wall vertex is the vertex itself under GlobalIds numbering;
    // under the owner-contiguous numbering the condition names the vertices.
    if (!owner_contiguous) {
        std::vector<GlobalIndex> wall{0, 1, 2 * kCells, 2 * kCells + 1};
        sys.addConstraint(std::make_unique<constraints::DirichletBC>(std::move(wall), 0.0));
    } else {
        std::vector<constraints::VertexDirichletValue> wall;
        for (const GlobalIndex vertex : {GlobalIndex{0}, GlobalIndex{1}, GlobalIndex{2 * kCells},
                                         GlobalIndex{2 * kCells + 1}}) {
            wall.push_back({.vertex_id = vertex, .value = 0.0});
        }
        sys.addSystemConstraint(std::make_unique<constraints::VertexDirichletConstraint>(
            out.u, std::move(wall), constraints::VertexIdMode::LocalVertexId));
    }

    SetupOptions options;
    options.dof_options.global_numbering = owner_contiguous ? dofs::GlobalNumberingMode::OwnerContiguous
                                                            : dofs::GlobalNumberingMode::GlobalIds;
    options.dof_options.ownership = owner_contiguous ? dofs::OwnershipStrategy::CellOwner
                                                     : dofs::OwnershipStrategy::VertexGID;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = size;
    options.dof_options.mpi_comm = comm;
    options.assembly_options.deterministic = true;
    options.assembly_options.overlap_communication = false;
    options.multiplier_lowering = lowering;
    SetupInputs inputs;
    inputs.topology_override = stripTopology(rank, size);
    sys.setup(options, inputs);
    if (owner_contiguous) {
        return out;
    }

    // The constrained DOFs are the wall vertices'.
    const auto* entities = sys.dofHandler().getEntityDofMap();
    EXPECT_NE(entities, nullptr);
    if (entities != nullptr) {
        for (const GlobalIndex vertex : {GlobalIndex{0}, GlobalIndex{1}, GlobalIndex{2 * kCells},
                                         GlobalIndex{2 * kCells + 1}}) {
            const auto dofs = entities->getVertexDofs(vertex);
            EXPECT_EQ(dofs.size(), 1u);
            if (!dofs.empty()) {
                EXPECT_EQ(dofs[0], vertex);
            }
        }
    }
    return out;
}

/// Dense matrix and vector in exact accumulators, combined over the ranks once.
class ExactDenseView final : public assembly::GlobalSystemView {
public:
    explicit ExactDenseView(GlobalIndex n)
        : n_(n)
        , matrix_(static_cast<std::size_t>(n * n))
        , vector_(static_cast<std::size_t>(n))
    {
    }

    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local,
                          assembly::AddMode mode) override
    {
        addMatrixEntries(dofs, dofs, local, mode);
    }
    void addMatrixEntries(std::span<const GlobalIndex> rows,
                          std::span<const GlobalIndex> cols,
                          std::span<const Real> local,
                          assembly::AddMode mode) override
    {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            for (std::size_t j = 0; j < cols.size(); ++j) {
                addMatrixEntry(rows[i], cols[j], local[i * cols.size() + j], mode);
            }
        }
    }
    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode) override
    {
        if (row < 0 || col < 0) {
            return;
        }
        apply(matrix_[static_cast<std::size_t>(row * n_ + col)], value, mode);
    }
    void setDiagonal(std::span<const GlobalIndex> dofs, std::span<const Real> values) override
    {
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            setDiagonal(dofs[i], values[i]);
        }
    }
    void setDiagonal(GlobalIndex dof, Real value) override
    {
        addMatrixEntry(dof, dof, value, assembly::AddMode::Insert);
    }
    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal) override
    {
        for (const auto row : rows) {
            for (GlobalIndex col = 0; col < n_; ++col) {
                matrix_[static_cast<std::size_t>(row * n_ + col)] = systems::ExactAccumulator{};
            }
            if (set_diagonal) {
                setDiagonal(row, 1.0);
            }
        }
    }
    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local,
                          assembly::AddMode mode) override
    {
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], local[i], mode);
        }
    }
    void addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode) override
    {
        if (dof < 0) {
            return;
        }
        apply(vector_[static_cast<std::size_t>(dof)], value, mode);
    }
    void setVectorEntries(std::span<const GlobalIndex> dofs, std::span<const Real> values) override
    {
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], values[i], assembly::AddMode::Insert);
        }
    }
    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        for (const auto dof : dofs) {
            vector_[static_cast<std::size_t>(dof)] = systems::ExactAccumulator{};
        }
    }
    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        return vector_[static_cast<std::size_t>(dof)].value();
    }
    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        return matrix_[static_cast<std::size_t>(row * n_ + col)].value();
    }
    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = assembly::AssemblyPhase::Finalized; }
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }
    [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return n_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return n_; }
    [[nodiscard]] std::string backendName() const override { return "ExactDense"; }
    void zero() override
    {
        for (auto& a : matrix_) {
            a = systems::ExactAccumulator{};
        }
        for (auto& a : vector_) {
            a = systems::ExactAccumulator{};
        }
    }

    /// The sums over the ranks of @p comm, rounded once.
    std::pair<std::vector<Real>, std::vector<Real>> reduce(const CollectiveCommunicator& comm) const
    {
        const auto matrix = collectiveExactSums(matrix_, comm);
        const auto vector = collectiveExactSums(vector_, comm);
        return {std::vector<Real>(matrix.begin(), matrix.end()),
                std::vector<Real>(vector.begin(), vector.end())};
    }

private:
    static void apply(systems::ExactAccumulator& acc, Real value, assembly::AddMode mode)
    {
        if (mode == assembly::AddMode::Insert) {
            acc = systems::ExactAccumulator{};
        }
        acc.add(value);
    }

    GlobalIndex n_{0};
    std::vector<systems::ExactAccumulator> matrix_;
    std::vector<systems::ExactAccumulator> vector_;
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

AssemblyRequest jacobianAndResidual()
{
    AssemblyRequest request;
    request.op = "poisson";
    request.want_matrix = true;
    request.want_vector = true;
    request.suppress_constraint_inhomogeneity = true;
    return request;
}

std::vector<Real> allreduceSum(std::span<const Real> local, MPI_Comm comm)
{
    std::vector<Real> global(local.size(), 0.0);
    MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

/// Solves A x = b (dense, partial pivoting).
std::vector<Real> solveDense(std::vector<Real> a, std::vector<Real> b)
{
    const auto n = b.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i * n + k]) > std::abs(a[p * n + k])) {
                p = i;
            }
        }
        if (p != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[k * n + j], a[p * n + j]);
            }
            std::swap(b[k], b[p]);
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real f = a[i * n + k] / a[k * n + k];
            for (std::size_t j = k; j < n; ++j) {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    std::vector<Real> x(n, 0.0);
    for (std::size_t k = n; k-- > 0;) {
        Real s = b[k];
        for (std::size_t j = k + 1; j < n; ++j) {
            s -= a[k * n + j] * x[j];
        }
        x[k] = s / a[k * n + k];
    }
    return x;
}

std::vector<Real> negate(std::vector<Real> v)
{
    for (auto& x : v) {
        x = -x;
    }
    return v;
}

} // namespace

TEST(GlobalMultiplierG1MPI, ResidualEntryAndLambdaAreTheOneRankValues)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    auto reference = buildProblem(MPI_COMM_SELF, 0, 1);
    auto parallel = buildProblem(comm, rank, size);
    auto& one = *reference.system;
    auto& many = *parallel.system;
    const auto n = one.dofHandler().getNumDofs();
    ASSERT_EQ(many.dofHandler().getNumDofs(), n);
    ASSERT_EQ(one.fieldDofOffset(reference.lambda), many.fieldDofOffset(parallel.lambda));
    const auto lam = static_cast<std::size_t>(one.fieldDofOffset(reference.lambda));
    const auto nn = static_cast<std::size_t>(n);

    // 1. The residual entry at a state, through the ordinary parallel path.
    std::vector<Real> state(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        state[i] = std::sin(0.37 * static_cast<Real>(i) + 0.1);
    }
    one.constraints().distribute(state);
    SystemStateView view;
    view.u = state;
    const auto request = jacobianAndResidual();

    assembly::DenseSystemView one_out(n);
    one_out.zero();
    ASSERT_TRUE(one.assemble(request, view, &one_out, &one_out).success);
    assembly::DenseSystemView many_out(n);
    many_out.zero();
    ASSERT_TRUE(many.assemble(request, view, &many_out, &many_out).success);
    const auto many_vector = allreduceSum(many_out.vectorData(), comm);
    const auto many_matrix = allreduceSum(many_out.matrixData(), comm);
    EXPECT_EQ(many_vector[lam], one_out.vectorData()[lam]);
    for (std::size_t j = 0; j < nn; ++j) {
        EXPECT_NEAR(many_matrix[lam * nn + j], one_out.matrixData()[lam * nn + j], 1e-14) << j;
    }

    // 2. lambda from exact accumulators: the one-rank value, bit for bit.
    const std::vector<Real> zero(nn, 0.0);
    SystemStateView at_zero;
    at_zero.u = zero;
    ExactDenseView one_exact(n);
    one_exact.zero();
    ASSERT_TRUE(one.assemble(request, at_zero, &one_exact, &one_exact).success);
    ExactDenseView many_exact(n);
    many_exact.zero();
    ASSERT_TRUE(many.assemble(request, at_zero, &many_exact, &many_exact).success);
    const auto [one_a, one_b] = one_exact.reduce(CollectiveCommunicator{});
    const auto [many_a, many_b] = many_exact.reduce(CollectiveCommunicator(comm));
    EXPECT_EQ(many_a, one_a);
    EXPECT_EQ(many_b, one_b);
    const auto one_lambda = solveDense(one_a, negate(one_b))[lam];
    const auto many_lambda = solveDense(many_a, negate(many_b))[lam];
    EXPECT_EQ(many_lambda, one_lambda);

    // 3. lambda through the ordinary parallel path: round-off from the one-rank
    //    value, and the Q1 closed form 12c / (1 - h^2).
    assembly::DenseSystemView many_zero(n);
    many_zero.zero();
    ASSERT_TRUE(many.assemble(request, at_zero, &many_zero, &many_zero).success);
    const auto a = allreduceSum(many_zero.matrixData(), comm);
    const auto b = allreduceSum(many_zero.vectorData(), comm);
    const auto lambda = solveDense(a, negate(b))[lam];
    const Real h = 1.0 / static_cast<Real>(kCells);
    EXPECT_NEAR(lambda, one_lambda, 1e-12 * std::abs(one_lambda));
    EXPECT_NEAR(one_lambda, 12.0 * kMean / (1.0 - h * h), 1e-11 * 12.0 * kMean);

    // Every rank holds the same values.
    std::array<double, 2> local{one_lambda, lambda};
    std::array<double, 2> lowest{};
    std::array<double, 2> highest{};
    MPI_Allreduce(local.data(), lowest.data(), 2, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(local.data(), highest.data(), 2, MPI_DOUBLE, MPI_MAX, comm);
    EXPECT_EQ(lowest, highest);
}

namespace {

/// The n x n squares of the unit square, each split into a lower and an upper
/// triangle (as test_GlobalMultiplier.cpp's buildTriangleGrid); every rank sees
/// every cell, and the cells are split into contiguous blocks over the ranks.
class TriangleGridAccess final : public assembly::IMeshAccess {
public:
    TriangleGridAccess(int n, int rank, int size)
        : n_(n)
        , rank_(rank)
        , size_(size)
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            if (isOwnedCell(c)) {
                owned_.push_back(c);
            }
        }
    }

    [[nodiscard]] int owner(GlobalIndex cell) const
    {
        return static_cast<int>((cell * size_) / numCells());
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2 * n_ * n_; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return static_cast<GlobalIndex>(owned_.size()); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override { return owner(cell) == rank_; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Triangle3; }

    void getCellNodes(GlobalIndex cell, std::vector<GlobalIndex>& nodes) const override
    {
        const auto square = cell / 2;
        const auto i = square % n_;
        const auto j = square / n_;
        const auto id = [&](GlobalIndex a, GlobalIndex b) { return b * (n_ + 1) + a; };
        if (cell % 2 == 0) {
            nodes = {id(i, j), id(i + 1, j), id(i + 1, j + 1)};
        } else {
            nodes = {id(i, j), id(i + 1, j + 1), id(i, j + 1)};
        }
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node) const override
    {
        return {static_cast<Real>(node % (n_ + 1)) / static_cast<Real>(n_),
                static_cast<Real>(node / (n_ + 1)) / static_cast<Real>(n_), 0.0};
    }

    void getCellCoordinates(GlobalIndex cell, std::vector<std::array<Real, 3>>& coords) const override
    {
        std::vector<GlobalIndex> nodes;
        getCellNodes(cell, nodes);
        coords.resize(nodes.size());
        for (std::size_t k = 0; k < nodes.size(); ++k) {
            coords[k] = getNodeCoordinates(nodes[k]);
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override
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
        for (const auto c : owned_) {
            callback(c);
        }
    }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

    [[nodiscard]] dofs::MeshTopologyInfo topology() const
    {
        dofs::MeshTopologyInfo topo;
        topo.n_cells = numCells();
        topo.n_vertices = (n_ + 1) * (n_ + 1);
        topo.dim = 2;
        topo.cell2vertex_offsets.push_back(0);
        std::vector<GlobalIndex> nodes;
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            getCellNodes(c, nodes);
            for (const auto node : nodes) {
                topo.cell2vertex_data.push_back(static_cast<MeshIndex>(node));
            }
            topo.cell2vertex_offsets.push_back(static_cast<MeshOffset>(topo.cell2vertex_data.size()));
            topo.cell_gids.push_back(c);
            topo.cell_owner_ranks.push_back(owner(c));
        }
        for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
            topo.vertex_gids.push_back(v);
        }
        // The edges as the DOF handler derives them (reference edges of each
        // cell in cell order), with their index as global ID: every rank sees
        // every cell in the same order, so the IDs agree.
        const auto reference = elements::ReferenceElement::create(ElementType::Triangle3);
        std::map<std::pair<MeshIndex, MeshIndex>, MeshIndex> edge_ids;
        topo.cell2edge_offsets.push_back(0);
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            getCellNodes(c, nodes);
            for (std::size_t e = 0; e < reference.num_edges(); ++e) {
                const auto& ends = reference.edge_nodes(e);
                const auto a = static_cast<MeshIndex>(nodes[static_cast<std::size_t>(ends[0])]);
                const auto b = static_cast<MeshIndex>(nodes[static_cast<std::size_t>(ends[1])]);
                const auto key = std::make_pair(std::min(a, b), std::max(a, b));
                auto [it, inserted] = edge_ids.emplace(key, static_cast<MeshIndex>(edge_ids.size()));
                if (inserted) {
                    topo.edge2vertex_data.push_back(key.first);
                    topo.edge2vertex_data.push_back(key.second);
                    topo.edge_gids.push_back(static_cast<dofs::gid_t>(it->second));
                }
                topo.cell2edge_data.push_back(it->second);
            }
            topo.cell2edge_offsets.push_back(static_cast<MeshOffset>(topo.cell2edge_data.size()));
        }
        topo.n_edges = static_cast<GlobalIndex>(edge_ids.size());
        for (int r = 0; r < size_; ++r) {
            if (r != rank_) {
                topo.neighbor_ranks.push_back(r);
            }
        }
        return topo;
    }

private:
    GlobalIndex n_{1};
    int rank_{0};
    int size_{1};
    std::vector<GlobalIndex> owned_;
};

SetupOptions parallelOptions(MPI_Comm comm, int rank, int size)
{
    SetupOptions options;
    options.dof_options.global_numbering = dofs::GlobalNumberingMode::GlobalIds;
    options.dof_options.ownership = dofs::OwnershipStrategy::VertexGID;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = size;
    options.dof_options.mpi_comm = comm;
    options.assembly_options.deterministic = true;
    options.assembly_options.overlap_communication = false;
    return options;
}

constexpr Real kPi = 3.14159265358979323846;

/// G2 on the triangle grid: pure Neumann Poisson with a mean-zero multiplier.
std::unique_ptr<FESystem> buildG2(int n, MPI_Comm comm, int rank, int size, FieldId& lambda,
                                  MultiplierLowering lowering = MultiplierLowering::Auto)
{
    auto mesh = std::make_shared<TriangleGridAccess>(n, rank, size);
    auto h1 = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    auto sys = std::make_unique<FESystem>(mesh);
    const auto u_id = sys->addField(FieldSpec{.name = "u", .space = h1});
    lambda = sys->addField(FieldSpec{.name = "lambda", .space = r1});
    sys->addOperator("poisson");
    const auto f = FormExpr::coefficient("f", [](Real x, Real y, Real) {
        return 2.0 * kPi * kPi * std::cos(kPi * x) * std::cos(kPi * y);
    });
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lambda, *r1, "lambda");
    const auto mu = forms::TestField(lambda, *r1, "mu");
    (void)installFormulation(*sys, "poisson", {u_id, lambda},
                             (forms::inner(forms::grad(u), forms::grad(v)) - f * v - lam * v).dx() -
                                 (mu * u).dx());
    SetupInputs inputs;
    inputs.topology_override = mesh->topology();
    auto options = parallelOptions(comm, rank, size);
    options.multiplier_lowering = lowering;
    sys->setup(options, inputs);
    return sys;
}

struct G3System {
    std::unique_ptr<FESystem> system;
    FieldId u{INVALID_FIELD_ID};
    FieldId lambda{INVALID_FIELD_ID};
};

/// G3 on the triangle grid: P2/P1 Stokes, u = 0 on @p wall (velocity DOFs), mean-zero pressure multiplier.
G3System buildG3(int n, MPI_Comm comm, int rank, int size, const std::vector<GlobalIndex>& wall)
{
    G3System out;
    auto mesh = std::make_shared<TriangleGridAccess>(n, rank, size);
    auto p2 = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 2);
    auto velocity_space = std::make_shared<spaces::ProductSpace>(p2, 2);
    auto pressure_space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    out.system = std::make_unique<FESystem>(mesh);
    auto& sys = *out.system;
    out.u = sys.addField(FieldSpec{.name = "u", .space = velocity_space, .components = 2});
    const auto p_id = sys.addField(FieldSpec{.name = "p", .space = pressure_space});
    out.lambda = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    sys.addOperator("stokes");
    const auto f = FormExpr::coefficient("f", [](Real x, Real y, Real) {
        const Real two_pi_cubed = 2.0 * kPi * kPi * kPi;
        return std::array<Real, 3>{
            -two_pi_cubed * std::sin(2.0 * kPi * y) * (2.0 * std::cos(2.0 * kPi * x) - 1.0) -
                kPi * std::sin(kPi * x) * std::cos(kPi * y),
            two_pi_cubed * std::sin(2.0 * kPi * x) * (2.0 * std::cos(2.0 * kPi * y) - 1.0) -
                kPi * std::cos(kPi * x) * std::sin(kPi * y),
            0.0};
    });
    const auto u = forms::StateField(out.u, *velocity_space, "u");
    const auto v = forms::TestField(out.u, *velocity_space, "v");
    const auto p = forms::StateField(p_id, *pressure_space, "p");
    const auto q = forms::TestField(p_id, *pressure_space, "q");
    const auto lam = forms::StateField(out.lambda, *r1, "lambda");
    const auto mu = forms::TestField(out.lambda, *r1, "mu");
    (void)installFormulation(
        sys, "stokes", {out.u, p_id, out.lambda},
        (forms::inner(forms::grad(u), forms::grad(v)) - p * forms::div(v) - forms::inner(f, v)).dx() -
            (q * forms::div(u)).dx() + (lam * q).dx() + (mu * p).dx());
    if (!wall.empty()) {
        sys.addConstraint(std::make_unique<constraints::DirichletBC>(wall, 0.0));
    }
    SetupInputs inputs;
    inputs.topology_override = mesh->topology();
    sys.setup(parallelOptions(comm, rank, size), inputs);
    return out;
}

/// The velocity DOFs of G3 on the boundary of the unit square, from the
/// coordinates of the P2 nodes of a one-rank system without them.
std::vector<GlobalIndex> wallDofs(int n)
{
    auto probe = buildG3(n, MPI_COMM_SELF, 0, 1, {});
    const auto& map = probe.system->fieldDofHandler(probe.u).getDofMap();
    const auto offset = probe.system->fieldDofOffset(probe.u);
    TriangleGridAccess mesh(n, 0, 1);
    const auto& element = probe.system->fieldRecord(probe.u).space->getElement(ElementType::Triangle3, 0);
    const auto* lagrange = dynamic_cast<const basis::LagrangeBasis*>(&element.basis());
    EXPECT_NE(lagrange, nullptr);
    std::vector<GlobalIndex> wall;
    if (lagrange == nullptr) {
        return wall;
    }
    const auto& nodes = lagrange->nodes();
    for (GlobalIndex c = 0; c < mesh.numCells(); ++c) {
        std::vector<std::array<Real, 3>> coords;
        mesh.getCellCoordinates(c, coords);
        std::vector<math::Vector<Real, 3>> node_coords;
        for (const auto& x : coords) {
            node_coords.push_back(math::Vector<Real, 3>{x[0], x[1], x[2]});
        }
        geometry::MappingRequest request;
        request.element_type = ElementType::Triangle3;
        request.geometry_order = 1;
        request.use_affine = true;
        const auto mapping = geometry::MappingFactory::create(request, node_coords);
        const auto dofs = map.getCellDofs(c);
        EXPECT_EQ(dofs.size(), 2 * nodes.size());
        for (std::size_t k = 0; k < nodes.size(); ++k) {
            const auto x = mapping->map_to_physical(math::Vector<Real, 3>{nodes[k][0], nodes[k][1], nodes[k][2]});
            const auto on_wall = [](Real t) { return std::abs(t) < 1e-12 || std::abs(t - 1.0) < 1e-12; };
            if (on_wall(x[0]) || on_wall(x[1])) {
                for (std::size_t component = 0; component < 2; ++component) {
                    wall.push_back(dofs[component * nodes.size() + k] + offset);
                }
            }
        }
    }
    std::sort(wall.begin(), wall.end());
    wall.erase(std::unique(wall.begin(), wall.end()), wall.end());
    return wall;
}

/**
 * The checks of G4 for one problem: the residual entry of the global row at
 * a state through the ordinary parallel path, bit for bit; the exactly
 * accumulated system and its solution, bit for bit; the ordinary parallel
 * solution within round-off.
 */
void expectOneRankValues(FESystem& one, FESystem& many, FieldId lambda, const OperatorTag& op, MPI_Comm comm)
{
    const auto n = one.dofHandler().getNumDofs();
    ASSERT_EQ(many.dofHandler().getNumDofs(), n);
    const auto nn = static_cast<std::size_t>(n);
    const auto lam = static_cast<std::size_t>(one.fieldDofOffset(lambda));
    AssemblyRequest request = jacobianAndResidual();
    request.op = op;

    std::vector<Real> state(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        state[i] = std::sin(0.37 * static_cast<Real>(i) + 0.1);
    }
    one.constraints().distribute(state);
    SystemStateView view;
    view.u = state;
    assembly::DenseSystemView one_out(n);
    one_out.zero();
    ASSERT_TRUE(one.assemble(request, view, &one_out, &one_out).success);
    assembly::DenseSystemView many_out(n);
    many_out.zero();
    ASSERT_TRUE(many.assemble(request, view, &many_out, &many_out).success);
    EXPECT_EQ(allreduceSum(many_out.vectorData(), comm)[lam], one_out.vectorData()[lam]);

    const std::vector<Real> zero(nn, 0.0);
    SystemStateView at_zero;
    at_zero.u = zero;
    ExactDenseView one_exact(n);
    one_exact.zero();
    ASSERT_TRUE(one.assemble(request, at_zero, &one_exact, &one_exact).success);
    ExactDenseView many_exact(n);
    many_exact.zero();
    ASSERT_TRUE(many.assemble(request, at_zero, &many_exact, &many_exact).success);
    const auto [one_a, one_b] = one_exact.reduce(CollectiveCommunicator{});
    const auto [many_a, many_b] = many_exact.reduce(CollectiveCommunicator(comm));
    EXPECT_EQ(many_a, one_a);
    EXPECT_EQ(many_b, one_b);
    const auto one_x = solveDense(one_a, negate(one_b));
    EXPECT_EQ(solveDense(many_a, negate(many_b)), one_x);

    assembly::DenseSystemView many_zero(n);
    many_zero.zero();
    ASSERT_TRUE(many.assemble(request, at_zero, &many_zero, &many_zero).success);
    const auto x = solveDense(allreduceSum(many_zero.matrixData(), comm),
                              negate(allreduceSum(many_zero.vectorData(), comm)));
    Real scale = 0.0;
    for (const auto value : one_x) {
        scale = std::max(scale, std::abs(value));
    }
    for (std::size_t i = 0; i < nn; ++i) {
        EXPECT_NEAR(x[i], one_x[i], 1e-11 * scale) << "dof " << i;
    }
}

} // namespace

TEST(GlobalMultiplierG4MPI, G2MeanZeroMultiplierIsTheOneRankSolution)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    FieldId lambda_one = INVALID_FIELD_ID;
    FieldId lambda_many = INVALID_FIELD_ID;
    auto one = buildG2(6, MPI_COMM_SELF, 0, 1, lambda_one);
    auto many = buildG2(6, comm, rank, size, lambda_many);
    // The multiplier fixes the constant of u at every rank count: no gauge.
    EXPECT_EQ(one->constraints().numConstraints(), 0);
    EXPECT_EQ(many->constraints().numConstraints(), 0);
    expectOneRankValues(*one, *many, lambda_one, "poisson", comm);
}

TEST(GlobalMultiplierG4MPI, G3PressureMultiplierIsTheOneRankSolution)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    constexpr int n = 4;
    const auto wall = wallDofs(n);
    EXPECT_EQ(wall.size(), static_cast<std::size_t>(2 * 8 * n)); // 2 components x 8n boundary P2 nodes
    auto one = buildG3(n, MPI_COMM_SELF, 0, 1, wall);
    auto many = buildG3(n, comm, rank, size, wall);
    expectOneRankValues(*one.system, *many.system, one.lambda, "stokes", comm);
}

// ---------------------------------------------------------------------------
// G5: G1 under the bordered lowering on FSILS GMRES at 1, 2 and 4 ranks
// ---------------------------------------------------------------------------

namespace {

struct SteadySolution {
    bool converged{false};
    Real lambda{0.0};
    /// The values at the vertices whose DOF this rank owns, by vertex.
    std::map<GlobalIndex, Real> owned_u;
};

/// A steady Newton solve of G1 from 0 with @p factory's solver.
SteadySolution solveSteady(Problem& problem,
                           const backends::BackendFactory& factory,
                           const backends::SolverOptions& solver_options)
{
    SteadySolution out;
    auto& sys = *problem.system;
    auto linear = factory.createLinearSolver(solver_options);
    EXPECT_NE(linear, nullptr);
    if (!linear) {
        return out;
    }
    TransientSystem transient(sys, std::make_shared<ZeroTimeDerivativeIntegrator>());
    const auto n = sys.dofHandler().getNumDofs();
    auto history = timestepping::TimeHistory::allocate(factory, n);
    history.setDt(1.0);
    history.setPrevDt(1.0);
    timestepping::NewtonOptions options;
    options.residual_op = "poisson";
    options.jacobian_op = "poisson";
    options.max_iterations = 6;
    options.abs_tolerance = 1e-12;
    options.rel_tolerance = 1e-12;
    options.use_line_search = false;
    timestepping::NewtonSolver newton(options);
    timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(sys, factory, workspace);
    history.repack(factory);
    const auto report = newton.solveStep(transient, *linear, /*solve_time=*/0.0, history, workspace);
    out.converged = report.converged;
    SystemStateView state;
    state.u_vector = &history.u();
    state.u = history.uSpan();
    const auto lambda = sys.globalFieldValues(problem.lambda, state);
    EXPECT_EQ(lambda.size(), 1u);
    if (!lambda.empty()) {
        out.lambda = lambda.front();
    }
    auto view = history.u().createGhostedReadView();
    const auto& owned = sys.dofHandler().getPartition().locallyOwned();
    const auto* map = sys.dofHandler().getEntityDofMap();
    EXPECT_NE(map, nullptr);
    if (map != nullptr) {
        for (GlobalIndex vertex = 0; vertex < 2 * (kCells + 1); ++vertex) {
            const auto dofs = map->getVertexDofs(vertex);
            if (dofs.size() == 1u && owned.contains(dofs[0])) {
                out.owned_u[vertex] = view->getVectorEntry(dofs[0]);
            }
        }
    }
    return out;
}

} // namespace

TEST(GlobalMultiplierG5MPI, G1BorderedOnFsilsGmresIsTheEigenInMatrixSolution)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // The reference: one rank, the multiplier in the matrix, Eigen's direct solve.
    auto reference = buildProblem(MPI_COMM_SELF, 0, 1, MultiplierLowering::InMatrix);
    auto eigen = backends::BackendFactory::create(backends::BackendKind::Eigen);
    ASSERT_NE(eigen, nullptr);
    backends::SolverOptions direct;
    direct.method = backends::SolverMethod::Direct;
    direct.preconditioner = backends::PreconditionerType::None;
    direct.rel_tol = 1e-14;
    direct.abs_tol = 1e-14;
    direct.max_iter = 1;
    const auto one = solveSteady(reference, *eigen, direct);
    ASSERT_TRUE(one.converged);
    // Q1 on the strip: lambda_h = 12c / (1 - h^2).
    const Real h = 1.0 / static_cast<Real>(kCells);
    EXPECT_NEAR(one.lambda, 12.0 * kMean / (1.0 - h * h), 1e-10 * 12.0 * kMean);

    // The bordered lowering on FSILS at this rank count: the field block on
    // FSILS, the multiplier through the bordered correction.
    auto bordered = buildProblem(comm, rank, size, MultiplierLowering::Bordered, /*owner_contiguous=*/true);
    auto& sys = *bordered.system;
    ASSERT_TRUE(sys.hasBorderedGlobalFields());
    ASSERT_EQ(sys.dofHandler().getNumDofs(), reference.system->dofHandler().getNumDofs() - 1);
    backends::FsilsFactory fsils(/*dof_per_node=*/1, sys.dofPermutation(), comm);
    backends::SolverOptions gmres;
    gmres.method = backends::SolverMethod::GMRES;
    gmres.preconditioner = backends::PreconditionerType::Diagonal;
    gmres.rel_tol = 1e-13;
    gmres.abs_tol = 1e-15;
    gmres.max_iter = 2000;
    const auto many = solveSteady(bordered, fsils, gmres);
    ASSERT_TRUE(many.converged);
    EXPECT_NEAR(many.lambda, one.lambda, 1e-10 * std::abs(one.lambda)) << "ranks " << size;
    // The field values agree too, vertex by vertex.
    Real scale = 0.0;
    for (const auto& [dof, value] : one.owned_u) {
        scale = std::max(scale, std::abs(value));
    }
    for (const auto& [dof, value] : many.owned_u) {
        const auto it = one.owned_u.find(dof);
        ASSERT_NE(it, one.owned_u.end()) << "vertex " << dof;
        EXPECT_NEAR(value, it->second, 1e-10 * scale) << "ranks " << size << " vertex " << dof;
    }
#endif
}

TEST(GlobalMultiplierG5MPI, G2UnderTheBorderedLoweringIsRefusedOnEveryRank)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    FieldId lambda = INVALID_FIELD_ID;
    std::string message;
    try {
        (void)buildG2(4, comm, rank, size, lambda, MultiplierLowering::Bordered);
    } catch (const FEException& e) {
        message = e.what();
    }
    EXPECT_NE(message.find("the multiplier 'lambda' is a gauge: its row alone fixes the constant of field 'u'"),
              std::string::npos)
        << "rank " << rank << ": " << message;
}

} // namespace svmp::FE::systems::test_global_multiplier_mpi

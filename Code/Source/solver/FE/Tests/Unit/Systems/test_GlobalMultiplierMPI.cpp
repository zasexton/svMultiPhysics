/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GlobalMultiplierMPI.cpp
 * @brief G1 (mean-constrained Poisson) on 1, 2 and 4 ranks
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
 */

#include <gtest/gtest.h>

#include <mpi.h>

#include "Assembly/Assembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Constraints/DirichletBC.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/Vocabulary.h"
#include "Spaces/GlobalSpace.h"
#include "Spaces/H1Space.h"
#include "Systems/ExactAccumulator.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/SystemCollectives.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
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
Problem buildProblem(MPI_Comm comm, int rank, int size)
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
    std::vector<GlobalIndex> wall{0, 1, 2 * kCells, 2 * kCells + 1};
    sys.addConstraint(std::make_unique<constraints::DirichletBC>(std::move(wall), 0.0));

    SetupOptions options;
    options.dof_options.global_numbering = dofs::GlobalNumberingMode::GlobalIds;
    options.dof_options.ownership = dofs::OwnershipStrategy::VertexGID;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = size;
    options.dof_options.mpi_comm = comm;
    options.assembly_options.deterministic = true;
    options.assembly_options.overlap_communication = false;
    SetupInputs inputs;
    inputs.topology_override = stripTopology(rank, size);
    sys.setup(options, inputs);

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

} // namespace svmp::FE::systems::test_global_multiplier_mpi

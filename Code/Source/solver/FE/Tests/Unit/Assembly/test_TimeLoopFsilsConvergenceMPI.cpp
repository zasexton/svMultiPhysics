/**
 * @file test_TimeLoopFsilsConvergenceMPI.cpp
 * @brief MPI regression: TimeLoop + NewtonSolver + FSILS backend should converge on a small distributed transient problem.
 *
 * Motivation: The new OOP solver shows nonlinear convergence failures only in multi-rank runs.
 * This test exercises the end-to-end transient path under MPI:
 * - distributed DOF numbering/ownership and ghost exchange,
 * - FSILS DOF permutation + overlap vectors,
 * - TimeHistory::repack() (history vectors allocated before the Jacobian exists),
 * - Generalized-α (first-order) special handling of algebraic (non-dt) fields.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Assembly/TimeIntegrationContext.h"

#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryModelDSL.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"

#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/Interfaces/DofPermutation.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Backends/Utils/BackendOptions.h"

#include "Core/FEException.h"
#include "Core/Types.h"

#include "Constraints/SystemConstraint.h"

#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"
#include "Forms/Vocabulary.h"

#include "Spaces/SpaceFactory.h"

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TransientSystem.h"

#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"
#include "TimeStepping/NewtonSolver.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <cstddef>
#include <limits>
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

int stripCellOwner(int cell, int n_cells, int world_size)
{
    FE_THROW_IF(n_cells < 1, InvalidArgumentException, "stripCellOwner: n_cells must be >= 1");
    FE_THROW_IF(world_size < 1, InvalidArgumentException, "stripCellOwner: world_size must be >= 1");
    const int scaled = (cell * world_size) / n_cells;
    return std::min(std::max(scaled, 0), world_size - 1);
}

// A deliberately disconnected distributed mesh: every rank owns one local
// Quad4 whose vertex and cell GIDs are unique to that rank.  This makes a
// rank-zero-only affine constraint truly rank-local while Newton and the
// distributed FSILS matrix still share one communicator.
class RankLocalOwnedQuadMeshAccess final : public IMeshAccess {
public:
    explicit RankLocalOwnedQuadMeshAccess(int rank)
        : nodes_{{
              {static_cast<Real>(2 * rank), Real{0}, Real{0}},
              {static_cast<Real>(2 * rank + 1), Real{0}, Real{0}},
              {static_cast<Real>(2 * rank + 1), Real{1}, Real{0}},
              {static_cast<Real>(2 * rank), Real{1}, Real{0}},
          }}
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return cell_id == 0;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override
    {
        FE_THROW_IF(cell_id != 0,
                    InvalidArgumentException,
                    "RankLocalOwnedQuadMeshAccess: invalid cell ID");
        return ElementType::Quad4;
    }

    void getCellNodes(GlobalIndex cell_id,
                      std::vector<GlobalIndex>& nodes) const override
    {
        FE_THROW_IF(cell_id != 0,
                    InvalidArgumentException,
                    "RankLocalOwnedQuadMeshAccess: invalid cell ID");
        nodes = {0, 1, 2, 3};
    }

    [[nodiscard]] std::array<Real, 3>
    getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        GlobalIndex cell_id,
        std::vector<std::array<Real, 3>>& coords) const override
    {
        FE_THROW_IF(cell_id != 0,
                    InvalidArgumentException,
                    "RankLocalOwnedQuadMeshAccess: invalid cell ID");
        coords.assign(nodes_.begin(), nodes_.end());
    }

    [[nodiscard]] LocalIndex
    getLocalFaceIndex(GlobalIndex /*face_id*/,
                      GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int
    getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    std::array<std::array<Real, 3>, 4> nodes_{};
};

[[nodiscard]] dofs::MeshTopologyInfo
buildRankLocalOwnedQuadTopology(int rank)
{
    dofs::MeshTopologyInfo topo;
    topo.dim = 2;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {
        static_cast<dofs::gid_t>(4 * rank + 0),
        static_cast<dofs::gid_t>(4 * rank + 1),
        static_cast<dofs::gid_t>(4 * rank + 2),
        static_cast<dofs::gid_t>(4 * rank + 3),
    };
    topo.vertex_coords = {
        static_cast<Real>(2 * rank), Real{0},
        static_cast<Real>(2 * rank + 1), Real{0},
        static_cast<Real>(2 * rank + 1), Real{1},
        static_cast<Real>(2 * rank), Real{1},
    };
    topo.cell_gids = {static_cast<dofs::gid_t>(rank)};
    topo.cell_owner_ranks = {rank};
    // No rank shares an entity with any other rank.
    topo.neighbor_ranks = {};
    return topo;
}

struct RankZeroMpcState {
    bool reverse{false};
    Real inhomogeneity{0};
};

[[nodiscard]] GlobalIndex localVertexDof(
    const systems::FESystem& system,
    FieldId field,
    GlobalIndex local_vertex)
{
    const auto* entity =
        system.fieldDofHandler(field).getEntityDofMap();
    FE_THROW_IF(entity == nullptr,
                systems::InvalidStateException,
                "localVertexDof: missing field EntityDofMap");
    const auto dofs = entity->getVertexDofs(local_vertex);
    FE_THROW_IF(dofs.size() != 1u,
                systems::InvalidStateException,
                "localVertexDof: scalar H1 vertex must have one DOF");
    return system.fieldDofOffset(field) + dofs.front();
}

class RankZeroSwitchingMpc final
    : public constraints::ISystemConstraint {
public:
    RankZeroSwitchingMpc(int rank,
                         FieldId field,
                         std::shared_ptr<RankZeroMpcState> state)
        : rank_(rank)
        , field_(field)
        , state_(std::move(state))
    {
        FE_CHECK_NOT_NULL(state_.get(),
                          "RankZeroSwitchingMpc: state");
    }

    void apply(
        const systems::FESystem& system,
        constraints::AffineConstraints& affine) override
    {
        if (rank_ != 0) {
            return;
        }
        const auto first = localVertexDof(system, field_, 0);
        const auto second = localVertexDof(system, field_, 1);
        const auto slave = state_->reverse ? second : first;
        const auto master = state_->reverse ? first : second;
        affine.addLine(slave);
        affine.addEntry(slave, master, 1.0);
        affine.setInhomogeneity(slave, state_->inhomogeneity);
    }

    bool updateValues(
        const systems::FESystem&,
        constraints::AffineConstraints&,
        double,
        double) override
    {
        return false;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override
    {
        return false;
    }

    [[nodiscard]] systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        systems::SetupStorageRequirements requirements;
        requirements.entity_dof_map = true;
        return requirements;
    }

private:
    int rank_{0};
    FieldId field_{INVALID_FIELD_ID};
    std::shared_ptr<RankZeroMpcState> state_{};
};

class ScalingLinearSolver final : public backends::LinearSolver {
public:
    ScalingLinearSolver(backends::LinearSolver& inner, Real scale)
        : inner_(inner)
        , scale_(scale)
    {
    }

    [[nodiscard]] backends::BackendKind
    backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(
        const backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const backends::SolverOptions&
    getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] backends::SolverReport solve(
        const backends::GenericMatrix& matrix,
        backends::GenericVector& solution,
        const backends::GenericVector& rhs) override
    {
        auto report = inner_.solve(matrix, solution, rhs);
        solution.scale(scale_);
        return report;
    }

private:
    backends::LinearSolver& inner_;
    Real scale_{1};
};

// The main Newton solve is delegated once.  Every later solve belongs to the
// explicit bordered K^{-1}B recovery exercised by the focused regression
// below.  Its first report deliberately disagrees by rank while remaining
// finite; subsequent reports remain finite but cannot pass residual replay or
// polish.  Newton must therefore take and leave the recovery path collectively.
class RankDivergentBorderedRecoveryLinearSolver final
    : public backends::LinearSolver {
public:
    RankDivergentBorderedRecoveryLinearSolver(
        backends::LinearSolver& inner,
        int rank)
        : inner_(inner)
        , rank_(rank)
    {
    }

    [[nodiscard]] backends::BackendKind
    backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(
        const backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const backends::SolverOptions&
    getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] backends::SolverReport solve(
        const backends::GenericMatrix& matrix,
        backends::GenericVector& solution,
        const backends::GenericVector& rhs) override
    {
        ++solve_calls;
        if (solve_calls == 1) {
            auto report = inner_.solve(matrix, solution, rhs);
            report.converged = true;
            report.numerical_breakdown = false;
            report.iterations = std::max(report.iterations, 1);
            report.initial_residual_norm = 1.0;
            report.final_residual_norm = 0.0;
            report.relative_residual = 0.0;
            report.message = "delegated main Newton solve";
            return report;
        }

        solution.zero();
        backends::SolverReport report;
        report.iterations = 1;
        report.initial_residual_norm = 1.0;
        report.final_residual_norm = 1.0;
        report.relative_residual = 1.0;
        report.message = "scripted bordered recovery miss";
        if (solve_calls == 2) {
            ++divergent_report_injections;
            report.converged = rank_ != 0;
            if (report.converged) {
                report.final_residual_norm = 0.0;
                report.relative_residual = 0.0;
                report.message = "scripted rank-local bordered success";
            }
        }
        return report;
    }

    void setRankOneUpdates(
        std::span<const backends::RankOneUpdate> updates) override
    {
        inner_.setRankOneUpdates(updates);
    }

    void setReducedFieldUpdates(
        std::span<const backends::ReducedFieldUpdate> updates) override
    {
        inner_.setReducedFieldUpdates(updates);
    }

    void setEffectiveTimeStep(double dt_eff) override
    {
        inner_.setEffectiveTimeStep(dt_eff);
    }

    void setDirichletDofs(
        std::span<const GlobalIndex> dofs) override
    {
        inner_.setDirichletDofs(dofs);
    }

    void setGroupedBorderedFieldCouplings(
        std::span<const backends::GroupedBorderedFieldCoupling> groups) override
    {
        inner_.setGroupedBorderedFieldCouplings(groups);
    }

    [[nodiscard]] bool
    supportsNativeRankOneUpdates() const noexcept override
    {
        return inner_.supportsNativeRankOneUpdates();
    }

    [[nodiscard]] bool
    supportsNativeReducedFieldUpdates() const noexcept override
    {
        return inner_.supportsNativeReducedFieldUpdates();
    }

    [[nodiscard]] bool supportsNullspace() const noexcept override
    {
        return inner_.supportsNullspace();
    }

    void setNullspaceBasis(
        std::span<const std::vector<double>> basis) override
    {
        inner_.setNullspaceBasis(basis);
    }

    int solve_calls{0};
    int divergent_report_injections{0};

private:
    backends::LinearSolver& inner_;
    int rank_{0};
};

// 2D strip of Quad4 cells, with interleaved node IDs:
// x-index i has nodes {2*i (bottom), 2*i+1 (top)}.
class StripQuadMeshAccess final : public IMeshAccess {
public:
    StripQuadMeshAccess(int n_cells, int my_rank, int world_size)
        : n_cells_(n_cells)
        , my_rank_(my_rank)
        , world_size_(world_size)
    {
        FE_THROW_IF(n_cells_ < 1, InvalidArgumentException, "StripQuadMeshAccess: n_cells must be >= 1");
        FE_THROW_IF(world_size_ < 1,
                    InvalidArgumentException,
                    "StripQuadMeshAccess: world_size must be >= 1");

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
        GlobalIndex owned = 0;
        for (int c = 0; c < n_cells_; ++c) {
            if (stripCellOwner(c, n_cells_, world_size_) == my_rank_) {
                ++owned;
            }
        }
        return owned;
    }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 2; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return stripCellOwner(static_cast<int>(cell_id), n_cells_, world_size_) == my_rank_;
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

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex cell_id) const override
    {
        if (face_id == 0 && cell_id == 0) {
            return 3; // left boundary on first cell
        }
        if (face_id == 1 && cell_id == static_cast<GlobalIndex>(n_cells_ - 1)) {
            return 1; // right boundary on last cell
        }
        return 0;
    }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        if (face_id == 0) {
            return 11;
        }
        if (face_id == 1) {
            return 12;
        }
        return -1;
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override { return {0, 0}; }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) callback(c);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (int c = 0; c < n_cells_; ++c) {
            if (stripCellOwner(c, n_cells_, world_size_) == my_rank_) {
                callback(static_cast<GlobalIndex>(c));
            }
        }
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker < 0 || marker == 11) {
            callback(/*face_id=*/0, /*cell_id=*/0);
        }
        if (marker < 0 || marker == 12) {
            callback(/*face_id=*/1, /*cell_id=*/static_cast<GlobalIndex>(n_cells_ - 1));
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int n_cells_{0};
    int my_rank_{0};
    int world_size_{1};
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
        topo.cell_owner_ranks[static_cast<std::size_t>(c)] = stripCellOwner(c, n_cells, world_size);
    }

    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

[[nodiscard]] std::shared_ptr<const backends::DofPermutation>
buildFsilsDofPermutation(const systems::FESystem& system,
                         int dof_per_node,
                         const dofs::DofDistributionOptions& dof_options)
{
    using backends::DofPermutation;

    if (dof_per_node <= 0) {
        return {};
    }

    const GlobalIndex total_dofs = system.dofHandler().getNumDofs();
    if (total_dofs <= 0) {
        return {};
    }

    // Derive node-block permutation from the field map (requires equal-order fields).
    const auto& fmap = system.fieldMap();
    const std::size_t n_fields = fmap.numFields();
    if (n_fields == 0u) {
        return {};
    }

    GlobalIndex n_nodes = -1;
    int expected_dof_per_node = 0;
    for (std::size_t f = 0; f < n_fields; ++f) {
        const auto& field = fmap.getField(f);
        expected_dof_per_node += field.n_components;
        if (field.n_components <= 0) {
            return {};
        }
        if (field.n_dofs % field.n_components != 0) {
            return {};
        }
        const GlobalIndex n_per_component = field.n_dofs / field.n_components;
        if (n_nodes < 0) {
            n_nodes = n_per_component;
        } else if (n_nodes != n_per_component) {
            return {};
        }
    }

    if (expected_dof_per_node != dof_per_node) {
        return {};
    }
    if (n_nodes <= 0) {
        return {};
    }
    if (total_dofs != static_cast<GlobalIndex>(dof_per_node) * n_nodes) {
        return {};
    }

    const bool explicit_spatial =
        dof_options.numbering == dofs::DofNumberingStrategy::Morton ||
        dof_options.numbering == dofs::DofNumberingStrategy::Hilbert;
    const bool default_spatial =
        dof_options.enable_spatial_locality_ordering &&
        dof_options.numbering == dofs::DofNumberingStrategy::Sequential;
    const bool want_spatial = explicit_spatial || default_spatial;
    const auto curve =
        explicit_spatial
            ? (dof_options.numbering == dofs::DofNumberingStrategy::Hilbert
                   ? dofs::SpatialCurveType::Hilbert
                   : dofs::SpatialCurveType::Morton)
            : dof_options.spatial_curve;

    constexpr std::uint32_t kSfcBits = 21u;
    constexpr std::uint64_t kSfcMaxCoord = (1ULL << kSfcBits) - 1ULL;

    auto morton3d = [](std::uint32_t xi, std::uint32_t yi, std::uint32_t zi) -> std::uint64_t {
        auto spread = [](std::uint64_t v) -> std::uint64_t {
            v = (v | (v << 32)) & 0x1f00000000ffffULL;
            v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
            v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
            v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
            v = (v | (v << 2)) & 0x1249249249249249ULL;
            return v;
        };
        return spread(xi) | (spread(yi) << 1) | (spread(zi) << 2);
    };

    auto hilbert_nd = [](const std::array<std::uint32_t, 3>& coords, std::uint32_t bits) -> std::uint64_t {
        std::array<std::uint32_t, 3> x = coords;
        const int n = 3;

        std::uint32_t M = 1u << (bits - 1u);
        for (std::uint32_t Q = M; Q > 1u; Q >>= 1u) {
            const std::uint32_t P = Q - 1u;
            for (int i = 0; i < n; ++i) {
                if ((x[static_cast<std::size_t>(i)] & Q) != 0u) {
                    x[0] ^= P;
                } else {
                    const std::uint32_t t = (x[0] ^ x[static_cast<std::size_t>(i)]) & P;
                    x[0] ^= t;
                    x[static_cast<std::size_t>(i)] ^= t;
                }
            }
        }
        for (int i = 1; i < n; ++i) {
            x[static_cast<std::size_t>(i)] ^= x[static_cast<std::size_t>(i - 1)];
        }
        std::uint32_t t = 0u;
        for (std::uint32_t Q = M; Q > 1u; Q >>= 1u) {
            if ((x[static_cast<std::size_t>(n - 1)] & Q) != 0u) {
                t ^= (Q - 1u);
            }
        }
        for (int i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] ^= t;
        }

        std::uint64_t index = 0;
        for (int b = static_cast<int>(bits) - 1; b >= 0; --b) {
            for (int i = 0; i < n; ++i) {
                index <<= 1u;
                index |= static_cast<std::uint64_t>((x[static_cast<std::size_t>(i)] >> static_cast<std::uint32_t>(b)) & 1u);
            }
        }
        return index;
    };

    auto sfc_code = [&](double x, double y, double z) -> std::uint64_t {
        auto normalize = [](double v) -> std::uint32_t {
            v = std::max(0.0, std::min(1.0, v));
            return static_cast<std::uint32_t>(v * static_cast<double>(kSfcMaxCoord));
        };
        const std::uint32_t xi = normalize(x);
        const std::uint32_t yi = normalize(y);
        const std::uint32_t zi = normalize(z);
        if (curve == dofs::SpatialCurveType::Hilbert) {
            return hilbert_nd(std::array<std::uint32_t, 3>{xi, yi, zi}, kSfcBits);
        }
        return morton3d(xi, yi, zi);
    };

    std::vector<int> node_owner(static_cast<std::size_t>(n_nodes), -1);
    std::vector<std::array<double, 3>> node_xyz(static_cast<std::size_t>(n_nodes),
                                                std::array<double, 3>{0.0, 0.0, 0.0});
    const auto* emap = system.dofHandler().getEntityDofMap();

    std::array<double, 3> min_xyz{std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()};
    std::array<double, 3> max_xyz{-std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity()};
    const int dim = std::max(2, system.meshAccess().dimension());

    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        const GlobalIndex fe0 = fmap.componentToGlobal(0, 0, node);
        const int owner0 = system.dofHandler().getDofMap().getDofOwner(fe0);
        if (owner0 < 0) {
            return {};
        }
        for (std::size_t f = 0; f < n_fields; ++f) {
            const auto& field = fmap.getField(f);
            for (LocalIndex c = 0; c < field.n_components; ++c) {
                const GlobalIndex fe = fmap.componentToGlobal(f, c, node);
                if (system.dofHandler().getDofMap().getDofOwner(fe) != owner0) {
                    return {};
                }
            }
        }

        node_owner[static_cast<std::size_t>(node)] = owner0;

        std::array<double, 3> xyz{0.0, 0.0, 0.0};
        bool have_xyz = false;
        if (want_spatial && emap) {
            if (const auto ent = emap->getDofEntity(fe0); ent && ent->kind == dofs::EntityKind::Vertex) {
                const auto p = system.meshAccess().getNodeCoordinates(ent->id);
                xyz = {static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
                have_xyz = true;
            }
        }
        if (!have_xyz) {
            const double d = static_cast<double>(node);
            xyz = {d, d, d};
        }
        node_xyz[static_cast<std::size_t>(node)] = xyz;
        for (int a = 0; a < dim && a < 3; ++a) {
            min_xyz[static_cast<std::size_t>(a)] = std::min(min_xyz[static_cast<std::size_t>(a)], xyz[static_cast<std::size_t>(a)]);
            max_xyz[static_cast<std::size_t>(a)] = std::max(max_xyz[static_cast<std::size_t>(a)], xyz[static_cast<std::size_t>(a)]);
        }
    }

    auto norm = [&](double v, int axis) -> double {
        const auto ax = static_cast<std::size_t>(axis);
        const double lo = min_xyz[ax];
        const double hi = max_xyz[ax];
        if (!(hi > lo)) return 0.0;
        return (v - lo) / (hi - lo);
    };

    struct NodeKey {
        int owner{0};
        std::uint64_t code{0};
        GlobalIndex node{-1};
    };
    std::vector<NodeKey> ordering;
    ordering.reserve(static_cast<std::size_t>(n_nodes));
    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        const auto& xyz = node_xyz[static_cast<std::size_t>(node)];
        const double x = norm(xyz[0], 0);
        const double y = (dim >= 2) ? norm(xyz[1], 1) : 0.0;
        const double z = (dim >= 3) ? norm(xyz[2], 2) : 0.0;
        const std::uint64_t code = want_spatial ? sfc_code(x, y, z) : 0u;
        ordering.push_back(NodeKey{node_owner[static_cast<std::size_t>(node)], code, node});
    }
    std::sort(ordering.begin(), ordering.end(), [&](const NodeKey& a, const NodeKey& b) {
        if (a.owner != b.owner) return a.owner < b.owner;
        if (a.code != b.code) return a.code < b.code;
        return a.node < b.node;
    });

    std::vector<GlobalIndex> node_to_backend(static_cast<std::size_t>(n_nodes), INVALID_GLOBAL_INDEX);
    for (GlobalIndex i = 0; i < n_nodes; ++i) {
        node_to_backend[static_cast<std::size_t>(ordering[static_cast<std::size_t>(i)].node)] = i;
    }

    auto perm = std::make_shared<DofPermutation>();
    perm->forward.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);
    perm->inverse.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);

    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        const GlobalIndex backend_node = node_to_backend[static_cast<std::size_t>(node)];
        if (backend_node < 0) {
            return {};
        }
        int comp_offset = 0;
        for (std::size_t f = 0; f < n_fields; ++f) {
            const auto& field = fmap.getField(f);
            for (LocalIndex c = 0; c < field.n_components; ++c) {
                const GlobalIndex fe_dof = fmap.componentToGlobal(f, c, node);
                const GlobalIndex fs_dof =
                    backend_node * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(comp_offset);
                if (fe_dof < 0 || fe_dof >= total_dofs) {
                    return {};
                }
                perm->forward[static_cast<std::size_t>(fe_dof)] = fs_dof;
                perm->inverse[static_cast<std::size_t>(fs_dof)] = fe_dof;
                ++comp_offset;
            }
        }
        if (comp_offset != dof_per_node) {
            return {};
        }
    }

    if (std::any_of(perm->forward.begin(), perm->forward.end(),
                    [](GlobalIndex v) { return v == INVALID_GLOBAL_INDEX; })) {
        return {};
    }
    if (std::any_of(perm->inverse.begin(), perm->inverse.end(),
                    [](GlobalIndex v) { return v == INVALID_GLOBAL_INDEX; })) {
        return {};
    }

    return perm;
}

[[nodiscard]] std::shared_ptr<const backends::DofPermutation>
getFsilsDofPermutation(const systems::FESystem& system,
                       int dof_per_node,
                       const dofs::DofDistributionOptions& dof_options)
{
    if (auto perm = system.dofPermutation()) {
        return perm;
    }
    if (dof_options.world_size <= 1) {
        return {};
    }
    return buildFsilsDofPermutation(system, dof_per_node, dof_options);
}

backends::SolverOptions fsilsGmresDiagOptions()
{
    backends::SolverOptions o;
    o.method = backends::SolverMethod::GMRES;
    o.preconditioner = backends::PreconditionerType::Diagonal;
    o.rel_tol = 1e-10;
    o.abs_tol = 1e-12;
    o.max_iter = 5000;
    return o;
}

backends::SolverOptions fsilsBlockSchurOptions(
    backends::FsilsBlockSchurSchurPreconditioner schur_pc =
        backends::FsilsBlockSchurSchurPreconditioner::DiagL,
    backends::FsilsBlockSchurMomentumApproximation momentum_hat =
        backends::FsilsBlockSchurMomentumApproximation::DiagK)
{
    backends::SolverOptions o;
    o.method = backends::SolverMethod::BlockSchur;
    o.preconditioner = backends::PreconditionerType::Diagonal;
    o.rel_tol = 1e-10;
    o.abs_tol = 1e-12;
    o.max_iter = 40;
    o.krylov_dim = 40;
    o.fsils_blockschur_gm_max_iter = 80;
    o.fsils_blockschur_cg_max_iter = 80;
    o.fsils_blockschur_gm_rel_tol = 1e-10;
    o.fsils_blockschur_cg_rel_tol = 1e-10;
    o.fsils_residual_check_policy = backends::FsilsResidualCheckPolicy::RetryOnly;
    o.fsils_blockschur_schur_preconditioner = schur_pc;
    o.fsils_blockschur_momentum_approximation = momentum_hat;

    backends::BlockLayout layout;
    layout.blocks.push_back({"u", 0, 2, backends::BlockRole::PrimaryField});
    layout.blocks.push_back({"p", 2, 1, backends::BlockRole::ConstraintField});
    layout.momentum_block = 0;
    layout.constraint_block = 1;
    o.block_layout = std::move(layout);
    return o;
}

std::unique_ptr<systems::FESystem>
buildOutletCoupledTransientSystem(MPI_Comm comm,
                                  int rank,
                                  int size,
                                  int n_cells)
{
    auto mesh = std::make_shared<StripQuadMeshAccess>(n_cells, rank, size);
    const auto topo = buildStripTopology(n_cells, rank, size);

    const auto u_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                             ElementType::Quad4,
                                             /*order=*/1,
                                             /*components=*/2);
    const auto p_space = spaces::Space(spaces::SpaceType::H1,
                                       ElementType::Quad4,
                                       /*order=*/1,
                                       /*components=*/1);

    auto sys = std::make_unique<systems::FESystem>(mesh);
    const auto u_field = sys->addField(systems::FieldSpec{.name = "u", .space = u_space, .components = 2});
    const auto p_field = sys->addField(systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys->addOperator("op");

    const auto u_state = forms::FormExpr::stateField(u_field, *u_space, "u");
    const auto p_state = forms::FormExpr::stateField(p_field, *p_space, "p");
    const auto u_disc = forms::FormExpr::discreteField(u_field, *u_space, "u_disc");
    const auto v = forms::TestFunction(*u_space, "v");
    const auto q = forms::TestFunction(*p_space, "q");
    const auto n = forms::FormExpr::normal();

    const auto Q_left = sys->boundaryIntegral(forms::inner(u_disc, n), /*marker=*/11);
    const auto Q_right = sys->boundaryIntegral(forms::inner(u_disc, n), /*marker=*/12);

    auto resistance_model = systems::aux::model("resistive_direct_only", [](systems::ModelFacade& m) {
        auto Q = m.input("Q");
        auto P = m.state("P", systems::AuxiliaryVariableKind::Algebraic);
        auto Rsum = m.param("Rsum");
        auto Pd = m.param("Pd");

        m.initialGuess("P", 0.0);
        m << systems::alg(P) == P - (Pd + Rsum * Q);
        m << systems::out("P_out") == P;
    });

    auto left_inst = sys->deploy(
        systems::use(resistance_model).name("left_outlet").boundary(11).monolithic()
            .bind("Q", Q_left)
            .param("Rsum", 80.0)
            .param("Pd", 20.0)
            .initialState({{"P", 20.0}}));
    auto right_inst = sys->deploy(
        systems::use(resistance_model).name("right_outlet").boundary(12).monolithic()
            .bind("Q", Q_right)
            .param("Rsum", 110.0)
            .param("Pd", 25.0)
            .initialState({{"P", 25.0}}));

    const auto one = forms::FormExpr::constant(Real(1.0));
    const auto lambda = forms::FormExpr::constant(Real(0.75));
    const auto nu = forms::FormExpr::constant(Real(0.05));
    const auto eps = forms::FormExpr::constant(Real(0.20));
    const auto kappa = forms::FormExpr::constant(Real(0.0));

    const auto residual =
        (forms::inner(u_state.dt(1), v) +
         lambda * forms::inner(u_state, v) +
         nu * forms::inner(forms::grad(u_state), forms::grad(v)) +
         eps * (one + forms::inner(u_state, u_state)) * forms::inner(u_state, v) -
         p_state * forms::div(v))
            .dx() +
        (q * forms::div(u_state) + kappa * p_state * q).dx() -
        (left_inst.output("P_out") * forms::inner(v, n)).ds(11) +
        (right_inst.output("P_out") * forms::inner(v, n)).ds(12);

    (void)systems::installFormulation(*sys, "op", {u_field, p_field}, residual);

    systems::SetupOptions setup_opts;
    setup_opts.assembler_name = "StandardAssembler";
    setup_opts.assembly_options.ghost_policy = GhostPolicy::ReverseScatter;
    setup_opts.assembly_options.deterministic = true;
    setup_opts.assembly_options.overlap_communication = false;
    setup_opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    setup_opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    setup_opts.dof_options.my_rank = rank;
    setup_opts.dof_options.world_size = size;
    setup_opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys->setup(setup_opts, inputs);
    sys->finalizeAuxiliaryLayout();

    return sys;
}

std::unique_ptr<systems::FESystem>
buildOutletCoupledTransientSystemRCR(MPI_Comm comm,
                                     int rank,
                                     int size,
                                     int n_cells)
{
    auto mesh = std::make_shared<StripQuadMeshAccess>(n_cells, rank, size);
    const auto topo = buildStripTopology(n_cells, rank, size);

    const auto u_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                             ElementType::Quad4,
                                             /*order=*/1,
                                             /*components=*/2);
    const auto p_space = spaces::Space(spaces::SpaceType::H1,
                                       ElementType::Quad4,
                                       /*order=*/1,
                                       /*components=*/1);

    auto sys = std::make_unique<systems::FESystem>(mesh);
    const auto u_field =
        sys->addField(systems::FieldSpec{.name = "u", .space = u_space, .components = 2});
    const auto p_field =
        sys->addField(systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys->addOperator("op");

    const auto u_state = forms::FormExpr::stateField(u_field, *u_space, "u");
    const auto p_state = forms::FormExpr::stateField(p_field, *p_space, "p");
    const auto u_disc = forms::FormExpr::discreteField(u_field, *u_space, "u_disc");
    const auto v = forms::TestFunction(*u_space, "v");
    const auto q = forms::TestFunction(*p_space, "q");
    const auto n = forms::FormExpr::normal();

    const auto Q_left = sys->boundaryIntegral(forms::inner(u_disc, n), /*marker=*/11);
    const auto Q_right = sys->boundaryIntegral(forms::inner(u_disc, n), /*marker=*/12);

    auto rcr_model = systems::aux::model("rcr_direct_probe", [](systems::ModelFacade& m) {
        auto Q = m.input("Q");
        auto X = m.state("X");
        auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");

        m << systems::ddt(X) == (Q - (X - Pd) / Rd) / C;
        m << systems::out("P_out") == X + Rp * Q;
    });

    auto left_inst = sys->deploy(
        systems::use(rcr_model).name("left_rcr").boundary(11).monolithic()
            .bind("Q", Q_left)
            .param("Rp", 20.0)
            .param("C", 0.5)
            .param("Rd", 60.0)
            .param("Pd", 20.0)
            .initialState({{"X", 20.0}}));
    auto right_inst = sys->deploy(
        systems::use(rcr_model).name("right_rcr").boundary(12).monolithic()
            .bind("Q", Q_right)
            .param("Rp", 30.0)
            .param("C", 0.35)
            .param("Rd", 80.0)
            .param("Pd", 25.0)
            .initialState({{"X", 25.0}}));

    const auto one = forms::FormExpr::constant(Real(1.0));
    const auto lambda = forms::FormExpr::constant(Real(0.75));
    const auto nu = forms::FormExpr::constant(Real(0.05));
    const auto eps = forms::FormExpr::constant(Real(0.20));
    const auto kappa = forms::FormExpr::constant(Real(0.0));

    const auto residual =
        (forms::inner(u_state.dt(1), v) +
         lambda * forms::inner(u_state, v) +
         nu * forms::inner(forms::grad(u_state), forms::grad(v)) +
         eps * (one + forms::inner(u_state, u_state)) * forms::inner(u_state, v) -
         p_state * forms::div(v))
            .dx() +
        (q * forms::div(u_state) + kappa * p_state * q).dx() -
        (left_inst.output("P_out") * forms::inner(v, n)).ds(11) +
        (right_inst.output("P_out") * forms::inner(v, n)).ds(12);

    (void)systems::installFormulation(*sys, "op", {u_field, p_field}, residual);

    systems::SetupOptions setup_opts;
    setup_opts.assembler_name = "StandardAssembler";
    setup_opts.assembly_options.ghost_policy = GhostPolicy::ReverseScatter;
    setup_opts.assembly_options.deterministic = true;
    setup_opts.assembly_options.overlap_communication = false;
    setup_opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    setup_opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    setup_opts.dof_options.my_rank = rank;
    setup_opts.dof_options.world_size = size;
    setup_opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys->setup(setup_opts, inputs);
    sys->finalizeAuxiliaryLayout();

    return sys;
}

} // namespace

TEST(TimeLoopFsilsConvergenceMPI, GeneralizedAlphaConvergesWithAlgebraicField)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    auto run_case = [&](bool deterministic_mode, bool overlap_communication) {
        const std::string trace_name =
            std::string(deterministic_mode ? "deterministic_on" : "deterministic_off") +
            (overlap_communication ? "_overlap_on" : "_overlap_off");
        SCOPED_TRACE(trace_name);

        // One owned cell per rank; all cells present as ghosts to enable OwnedRowsOnly-style assembly.
        const int n_cells = size;
        auto mesh = std::make_shared<StripQuadMeshAccess>(n_cells, rank, size);

        const auto u_space = spaces::VectorSpace(spaces::SpaceType::H1, ElementType::Quad4, /*order=*/1, /*components=*/2);
        const auto p_space = spaces::Space(spaces::SpaceType::H1, ElementType::Quad4, /*order=*/1, /*components=*/1);
        ASSERT_TRUE(u_space);
        ASSERT_TRUE(p_space);

        systems::FESystem sys(mesh);
        const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = u_space, .components = 2});
        const auto p_field = sys.addField(systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
        sys.addOperator("op");

        const auto u = forms::FormExpr::stateField(u_field, *u_space, "u");
        const auto p = forms::FormExpr::stateField(p_field, *p_space, "p");
        const auto v = forms::TestFunction(*u_space, "v");
        const auto q = forms::TestFunction(*p_space, "q");

        // Transient nonlinear reaction for velocity; algebraic pressure mass term.
        const auto one = forms::FormExpr::constant(Real(1.0));
        const auto lambda = forms::FormExpr::constant(Real(2.0));
        const auto eps = forms::FormExpr::constant(Real(0.05));
        const auto kappa = forms::FormExpr::constant(Real(1.0));

        const auto residual =
            (forms::inner(u.dt(1), v) +
             lambda * forms::inner(u, v) +
             eps * (one + forms::inner(u, u)) * forms::inner(u, v))
                .dx() +
            (kappa * p * q).dx();

        (void)systems::installFormulation(
            sys, "op",
            {u_field, p_field},
            residual);

        systems::SetupOptions setup_opts;
        setup_opts.assembler_name = "StandardAssembler";
        setup_opts.assembly_options.ghost_policy = GhostPolicy::ReverseScatter;
        setup_opts.assembly_options.deterministic = deterministic_mode;
        setup_opts.assembly_options.overlap_communication = overlap_communication;

        // FSILS distributed layout requires each rank's owned nodes to be contiguous in node space.
        // Use dense, process-count-independent global IDs (non-owner-contiguous) to force the
        // node-interleaved distributed sparsity path, and a deterministic ownership strategy that
        // yields contiguous node blocks for this strip topology.
        setup_opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
        setup_opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
        setup_opts.dof_options.my_rank = rank;
        setup_opts.dof_options.world_size = size;
        setup_opts.dof_options.mpi_comm = comm;

        systems::SetupInputs inputs;
        inputs.topology_override = buildStripTopology(n_cells, rank, size);

        sys.setup(setup_opts, inputs);
        ASSERT_TRUE(sys.isSetup());

        const GlobalIndex n_dofs = sys.dofHandler().getNumDofs();
        ASSERT_TRUE(inputs.topology_override.has_value());
        const GlobalIndex n_nodes = inputs.topology_override->n_vertices;
        ASSERT_EQ(n_dofs, n_nodes * 3);

        constexpr int dof_per_node = 3;
        auto perm = getFsilsDofPermutation(sys, dof_per_node, setup_opts.dof_options);
        if (size > 1) {
            ASSERT_TRUE(perm) << "Failed to build FSILS DOF permutation for test system";
        }

        const std::array<std::pair<backends::FsilsBlockSchurSchurPreconditioner,
                                   backends::FsilsBlockSchurMomentumApproximation>, 2> variants{{
            {backends::FsilsBlockSchurSchurPreconditioner::DiagL,
             backends::FsilsBlockSchurMomentumApproximation::DiagK},
            {backends::FsilsBlockSchurSchurPreconditioner::AlgebraicSchur,
             backends::FsilsBlockSchurMomentumApproximation::BlockDiagK},
        }};

        for (const auto& [schur_pc, momentum_hat] : variants) {
            SCOPED_TRACE(std::string(backends::fsilsBlockSchurPreconditionerToString(schur_pc)) + "/" +
                         std::string(backends::fsilsBlockSchurMomentumApproximationToString(momentum_hat)));

            backends::FsilsFactory factory(dof_per_node, perm);
            auto linear = factory.createLinearSolver(fsilsBlockSchurOptions(schur_pc, momentum_hat));
            ASSERT_TRUE(linear);

            // Allocate history before any matrix exists: this creates local-only vectors that must be repacked.
            auto history = timestepping::TimeHistory::allocate(factory, n_dofs, /*history_depth=*/2,
                                                               /*allocate_second_order_state=*/false);
            const double dt = 0.05;
            history.setTime(0.0);
            history.setDt(dt);
            history.setPrevDt(dt);
            history.setStepIndex(0);

            auto init = [&](backends::GenericVector& vec, double scale) {
                auto s = vec.localSpan();
                ASSERT_EQ(static_cast<GlobalIndex>(s.size()), n_dofs);
                for (GlobalIndex i = 0; i < n_dofs; ++i) {
                    s[static_cast<std::size_t>(i)] = static_cast<Real>(scale) * static_cast<Real>(0.01) * static_cast<Real>(i + 1);
                }
            };
            init(history.uPrev(), /*scale=*/1.0);
            init(history.uPrev2(), /*scale=*/1.0);
            history.resetCurrentToPrevious();

            auto base_integrator = std::make_shared<systems::BackwardDifferenceIntegrator>();
            systems::TransientSystem transient(sys, std::move(base_integrator));

            timestepping::TimeLoopOptions loop_opts;
            loop_opts.t0 = 0.0;
            loop_opts.t_end = 3.0 * dt;
            loop_opts.dt = dt;
            loop_opts.max_steps = 3;
            loop_opts.scheme = timestepping::SchemeKind::GeneralizedAlpha;
            loop_opts.generalized_alpha_rho_inf = 0.5;
            loop_opts.newton.residual_op = "op";
            loop_opts.newton.jacobian_op = "op";
            loop_opts.newton.max_iterations = 12;
            loop_opts.newton.abs_tolerance = 1e-12;
            loop_opts.newton.rel_tolerance = 1e-10;

            timestepping::TimeLoop loop(loop_opts);

            int nonconverged_steps = 0;
            std::vector<timestepping::NewtonReport> nonlinear_reports;
            timestepping::TimeLoopCallbacks callbacks;
            callbacks.on_nonlinear_done =
                [&nonconverged_steps, &nonlinear_reports](const timestepping::TimeHistory&,
                                                          const timestepping::NewtonReport& nr) {
                if (!nr.converged) {
                    ++nonconverged_steps;
                }
                nonlinear_reports.push_back(nr);
            };

            timestepping::TimeLoopReport rep{};
            try {
                rep = loop.run(transient, factory, *linear, history, callbacks);
            } catch (const FEException& e) {
                ADD_FAILURE() << "Rank " << rank << ": TimeLoop threw FEException: " << e.what();
                return;
            } catch (const std::exception& e) {
                ADD_FAILURE() << "Rank " << rank << ": TimeLoop threw std::exception: " << e.what();
                return;
            }

            EXPECT_TRUE(rep.success);
            EXPECT_NEAR(rep.final_time, loop_opts.t_end, 1e-12);
            EXPECT_EQ(nonconverged_steps, 0);
            EXPECT_FALSE(nonlinear_reports.empty());
            EXPECT_GE(nonlinear_reports.size(), 3u);

            bool saw_collective_activity = false;
            bool saw_blockschur_report = false;
            for (const auto& nr : nonlinear_reports) {
                EXPECT_TRUE(nr.converged);
                EXPECT_TRUE(nr.linear.converged);
                EXPECT_GE(nr.linear.iterations, 0);
                EXPECT_TRUE(std::isfinite(nr.linear.initial_residual_norm));
                EXPECT_TRUE(std::isfinite(nr.linear.final_residual_norm));
                EXPECT_TRUE(std::isfinite(nr.linear.relative_residual));
                saw_collective_activity =
                    saw_collective_activity || (nr.linear.collective_calls > 0u);

                if (nr.linear.blockschur_outer_iterations > 0) {
                    saw_blockschur_report = true;
                    EXPECT_GT(nr.linear.blockschur_momentum_solve_calls, 0);
                    EXPECT_GE(nr.linear.blockschur_momentum_iterations, 0);
                    EXPECT_GT(nr.linear.blockschur_schur_solve_calls, 0);
                    EXPECT_GE(nr.linear.blockschur_schur_iterations, 0);
                    EXPECT_LE(nr.linear.blockschur_momentum_restart_cycles,
                              nr.linear.blockschur_momentum_solve_calls);
                    EXPECT_GE(nr.linear.blockschur_schur_setup_time_seconds, 0.0);
                    EXPECT_GE(nr.linear.blockschur_schur_solve_time_seconds, 0.0);
                    EXPECT_GE(nr.linear.blockschur_collective_calls_max_per_outer, 0u);
                    EXPECT_GE(nr.linear.blockschur_collective_time_max_per_outer, 0.0);
                }
            }
            EXPECT_TRUE(saw_collective_activity);
            EXPECT_TRUE(saw_blockschur_report);
        }
    };

    run_case(/*deterministic_mode=*/true, /*overlap_communication=*/false);
    run_case(/*deterministic_mode=*/true, /*overlap_communication=*/true);
#endif
}

TEST(TimeLoopFsilsConvergenceMPI,
     RankLocalConstraintRefreshReinitializesJacobianCollectively)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks to enable this test";
    }

    auto mesh = std::make_shared<RankLocalOwnedQuadMeshAccess>(rank);
    const auto space = spaces::Space(spaces::SpaceType::H1,
                                     ElementType::Quad4,
                                     /*order=*/1,
                                     /*components=*/1);
    ASSERT_TRUE(space);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        systems::FieldSpec{.name = "u",
                           .space = space,
                           .components = 1});
    sys.addOperator("op");

    const auto u =
        forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::TestFunction(*space, "v");
    (void)systems::installFormulation(
        sys, "op", {u_field}, (u * v).dx());

    auto mpc_state = std::make_shared<RankZeroMpcState>();
    sys.addSystemConstraint(std::make_unique<RankZeroSwitchingMpc>(
        rank, u_field, mpc_state));

    systems::SetupOptions setup_opts;
    setup_opts.assembler_name = "StandardAssembler";
    setup_opts.assembly_options.ghost_policy =
        GhostPolicy::ReverseScatter;
    setup_opts.assembly_options.deterministic = true;
    setup_opts.assembly_options.overlap_communication = false;
    setup_opts.dof_options.global_numbering =
        dofs::GlobalNumberingMode::OwnerContiguous;
    setup_opts.dof_options.ownership =
        dofs::OwnershipStrategy::CellOwner;
    setup_opts.dof_options.my_rank = rank;
    setup_opts.dof_options.world_size = size;
    setup_opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = buildRankLocalOwnedQuadTopology(rank);
    ASSERT_NO_THROW(sys.setup(setup_opts, inputs));
    ASSERT_TRUE(sys.isSetup());

    EXPECT_EQ(sys.constraints().numConstraints(), rank == 0 ? 1u : 0u);
    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4 * size);

    backends::FsilsFactory factory(
        /*dof_per_node=*/1, sys.dofPermutation(), comm);
    auto inner_linear =
        factory.createLinearSolver(fsilsGmresDiagOptions());
    ASSERT_TRUE(inner_linear);

    // Allocate history before a distributed matrix exists, then repack it
    // after Newton configures the FSILS owned-row layout.
    auto history = timestepping::TimeHistory::allocate(
        factory,
        n_dofs,
        /*history_depth=*/2,
        /*allocate_second_order_state=*/false);
    history.setTime(0.0);
    history.setDt(0.1);
    history.setPrevDt(0.1);
    history.setStepIndex(0);
    auto fill_one = [](backends::GenericVector& vector) {
        auto values = vector.localSpan();
        std::fill(values.begin(), values.end(), Real{1});
    };
    fill_one(history.uPrev());
    fill_one(history.uPrev2());
    history.resetCurrentToPrevious();

    auto integrator =
        std::make_shared<systems::BackwardDifferenceIntegrator>();
    systems::TransientSystem transient(sys, std::move(integrator));

    using SyncPoint =
        timestepping::NewtonOptions::StateSynchronizationPoint;
    timestepping::NewtonWorkspace workspace;
    int line_search_trial_callbacks = 0;
    int restored_callbacks = 0;
    bool trial_semantics_active = false;
    bool trial_revision_recorded = false;
    bool restored_revision_recorded = false;
    bool accepted_trial_sync_observed = false;
    std::uint64_t trial_revision = 0;
    std::uint64_t restored_revision = 0;
    std::uint64_t trial_layout_revision = 0;
    const backends::GenericMatrix* trial_jacobian = nullptr;

    const auto base_revision = sys.sparsityPatternRevision();
    timestepping::NewtonOptions newton_opts;
    newton_opts.residual_op = "op";
    newton_opts.jacobian_op = "op";
    newton_opts.max_iterations = 1;
    newton_opts.abs_tolerance = 0.0;
    newton_opts.rel_tolerance = 0.0;
    newton_opts.step_tolerance = 0.0;
    newton_opts.stagnation_tolerance = 0.0;
    newton_opts.assemble_both_when_possible = false;
    newton_opts.use_line_search = true;
    newton_opts.line_search_max_iterations = 1;
    newton_opts.line_search_alpha_min = 1.0;
    newton_opts.line_search_shrink = 0.5;
    newton_opts.line_search_fail_on_no_reduction = true;
    newton_opts.synchronize_state =
        [&](const systems::SystemStateView&, SyncPoint point) {
            if (point == SyncPoint::LineSearchTrialResidual) {
                ++line_search_trial_callbacks;
                if (!trial_semantics_active) {
                    trial_semantics_active = true;
                    mpc_state->reverse = true;
                    mpc_state->inhomogeneity = Real{100};
                    // All ranks enter the rebuild even though only rank zero
                    // contributes an affine line.
                    sys.rebuildConstraintState();
                    trial_revision = sys.sparsityPatternRevision();
                    trial_revision_recorded = true;
                }
                return;
            }

            if (point == SyncPoint::AcceptedNonlinearState &&
                trial_semantics_active) {
                accepted_trial_sync_observed = true;
                trial_jacobian = workspace.jacobian.get();
                const auto view = workspace.jacobian->createAssemblyView();
                trial_layout_revision = view->matrixLayoutRevision();
                return;
            }

            if (point == SyncPoint::RestoredNonlinearState) {
                ++restored_callbacks;
                if (trial_semantics_active) {
                    trial_semantics_active = false;
                    mpc_state->reverse = false;
                    mpc_state->inhomogeneity = Real{0};
                    // The first restored pass rebuilds the accepted
                    // constraint set; the fixed-point callback is a no-op.
                    sys.rebuildConstraintState();
                    restored_revision = sys.sparsityPatternRevision();
                    restored_revision_recorded = true;
                }
            }
        };

    timestepping::NewtonSolver newton(newton_opts);
    newton.allocateWorkspace(sys, factory, workspace);
    ASSERT_NE(workspace.jacobian, nullptr);
    const auto* initial_jacobian = workspace.jacobian.get();
    const auto initial_layout_revision =
        workspace.jacobian->createAssemblyView()->matrixLayoutRevision();
    history.repack(factory);
    const std::vector<Real> accepted_base_state(
        history.u().localSpan().begin(),
        history.u().localSpan().end());

    ScalingLinearSolver ascent_solver(*inner_linear, Real{-1});
    timestepping::NewtonReport report{};
    ASSERT_NO_THROW(report = newton.solveStep(
                        transient,
                        ascent_solver,
                        /*solve_time=*/history.dt(),
                        history,
                        workspace));

    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.iterations, 1);
    EXPECT_TRUE(report.linear.converged);
    EXPECT_TRUE(trial_revision_recorded);
    EXPECT_TRUE(restored_revision_recorded);
    EXPECT_TRUE(accepted_trial_sync_observed);
    ASSERT_NE(trial_jacobian, nullptr);
    EXPECT_EQ(trial_jacobian, initial_jacobian)
        << "A compatible distributed refresh should retain matrix identity so "
           "existing vector layouts and bound references remain valid";
    EXPECT_GT(trial_layout_revision, initial_layout_revision)
        << "A rank-local sparsity change must advance the resolved-insertion "
           "layout generation on every communicator rank";

    if (rank == 0) {
        EXPECT_GT(trial_revision, base_revision);
        EXPECT_GT(restored_revision, trial_revision);
    } else {
        EXPECT_EQ(trial_revision, base_revision);
        EXPECT_EQ(restored_revision, base_revision);
    }

    int trial_min = 0;
    int trial_max = 0;
    int restored_min = 0;
    int restored_max = 0;
    MPI_Allreduce(&line_search_trial_callbacks,
                  &trial_min,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  comm);
    MPI_Allreduce(&line_search_trial_callbacks,
                  &trial_max,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm);
    MPI_Allreduce(&restored_callbacks,
                  &restored_min,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  comm);
    MPI_Allreduce(&restored_callbacks,
                  &restored_max,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm);
    EXPECT_EQ(trial_min, 2);
    EXPECT_EQ(trial_max, 2);
    EXPECT_EQ(restored_min, 2);
    EXPECT_EQ(restored_max, 2);

    EXPECT_FALSE(trial_semantics_active);
    EXPECT_FALSE(mpc_state->reverse);
    EXPECT_DOUBLE_EQ(mpc_state->inhomogeneity, Real{0});
    EXPECT_EQ(sys.constraints().numConstraints(), rank == 0 ? 1u : 0u);
    if (rank == 0) {
        const auto slave = localVertexDof(sys, u_field, 0);
        const auto master = localVertexDof(sys, u_field, 1);
        const auto line = sys.constraints().getConstraint(slave);
        ASSERT_TRUE(line.has_value());
        EXPECT_DOUBLE_EQ(line->inhomogeneity, 0.0);
        ASSERT_EQ(line->entries.size(), 1u);
        EXPECT_EQ(line->entries.front().master_dof, master);
        EXPECT_DOUBLE_EQ(line->entries.front().weight, 1.0);
    }

    const auto final_state = history.u().localSpan();
    ASSERT_EQ(final_state.size(), accepted_base_state.size());
    for (std::size_t i = 0; i < final_state.size(); ++i) {
        EXPECT_DOUBLE_EQ(final_state[i], accepted_base_state[i]);
    }
#endif
}

TEST(TimeLoopFsilsConvergenceMPI,
     PseudoTransientContinuationIncludesRankWithNoOwnedDtRows)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks to enable this test";
    }

    // Both ranks retain the same local cell topology, but cell-owner DOF
    // ownership assigns the sole cell and all four scalar rows to rank zero.
    // Rank one is therefore a real distributed FSILS participant with no
    // locally owned dt row.
    auto mesh = std::make_shared<StripQuadMeshAccess>(
        /*n_cells=*/1, rank, size);
    const auto space = spaces::Space(spaces::SpaceType::H1,
                                     ElementType::Quad4,
                                     /*order=*/1,
                                     /*components=*/1);
    ASSERT_TRUE(space);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        systems::FieldSpec{.name = "u",
                           .space = space,
                           .components = 1});
    sys.addOperator("op");

    const auto u =
        forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::TestFunction(*space, "v");
    const auto reaction = forms::FormExpr::constant(Real{1});
    const auto residual =
        (u.dt(1) * v + reaction * u * v).dx();
    (void)systems::installFormulation(
        sys, "op", {u_field}, residual);

    systems::SetupOptions setup_opts;
    setup_opts.assembler_name = "StandardAssembler";
    setup_opts.assembly_options.ghost_policy =
        GhostPolicy::ReverseScatter;
    setup_opts.assembly_options.deterministic = true;
    setup_opts.assembly_options.overlap_communication = false;
    setup_opts.dof_options.global_numbering =
        dofs::GlobalNumberingMode::OwnerContiguous;
    setup_opts.dof_options.ownership =
        dofs::OwnershipStrategy::CellOwner;
    setup_opts.dof_options.my_rank = rank;
    setup_opts.dof_options.world_size = size;
    setup_opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = buildStripTopology(
        /*n_cells=*/1, rank, size);
    ASSERT_NO_THROW(sys.setup(setup_opts, inputs));
    ASSERT_TRUE(sys.isSetup());

    const auto dt_fields = sys.timeDerivativeFields("op");
    ASSERT_EQ(dt_fields.size(), 1u);
    EXPECT_EQ(dt_fields.front(), u_field);
    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);

    backends::FsilsFactory factory(
        /*dof_per_node=*/1, sys.dofPermutation(), comm);
    auto linear = factory.createLinearSolver(
        fsilsGmresDiagOptions());
    ASSERT_TRUE(linear);

    auto history = timestepping::TimeHistory::allocate(
        factory,
        n_dofs,
        /*history_depth=*/2,
        /*allocate_second_order_state=*/false);
    constexpr double dt = 0.1;
    history.setTime(0.0);
    history.setDt(dt);
    history.setPrevDt(dt);
    history.setStepIndex(0);
    auto fill_one = [](backends::GenericVector& vector) {
        auto values = vector.localSpan();
        std::fill(values.begin(), values.end(), Real{1});
    };
    fill_one(history.uPrev());
    fill_one(history.uPrev2());
    history.resetCurrentToPrevious();

    auto integrator =
        std::make_shared<systems::BackwardDifferenceIntegrator>();
    systems::TransientSystem transient(sys, std::move(integrator));

    using SyncPoint =
        timestepping::NewtonOptions::StateSynchronizationPoint;
    int dt_only_jacobian_syncs = 0;
    int all_jacobian_syncs = 0;

    timestepping::NewtonOptions newton_opts;
    newton_opts.residual_op = "op";
    newton_opts.jacobian_op = "op";
    newton_opts.max_iterations = 1;
    newton_opts.abs_tolerance = 0.0;
    newton_opts.rel_tolerance = 0.0;
    newton_opts.step_tolerance = 0.0;
    newton_opts.stagnation_tolerance = 0.0;
    newton_opts.assemble_both_when_possible = false;
    newton_opts.use_line_search = false;
    newton_opts.pseudo_transient.enabled = true;
    newton_opts.pseudo_transient.activate_on_linear_failure = false;
    newton_opts.pseudo_transient.gamma_initial = 5.0;
    newton_opts.pseudo_transient.gamma_growth = 2.0;
    newton_opts.pseudo_transient.gamma_max = 100.0;
    newton_opts.pseudo_transient.max_linear_retries = 2;
    newton_opts.pseudo_transient.update_from_residual_ratio = false;
    newton_opts.synchronize_state =
        [&](const systems::SystemStateView& state, SyncPoint point) {
            if (point != SyncPoint::JacobianAssembly) {
                return;
            }
            ++all_jacobian_syncs;
            const auto* time = state.time_integration;
            if (time != nullptr &&
                time->time_derivative_term_weight == Real{1} &&
                time->non_time_derivative_term_weight == Real{0}) {
                ++dt_only_jacobian_syncs;
            }
        };

    timestepping::NewtonSolver newton(newton_opts);
    timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(sys, factory, workspace);
    ASSERT_NE(workspace.jacobian, nullptr);
    ASSERT_NE(workspace.ptc_mass_lumped, nullptr);
    history.repack(factory);

    const auto* fsils_jacobian =
        dynamic_cast<const backends::FsilsMatrix*>(
            workspace.jacobian.get());
    ASSERT_NE(fsils_jacobian, nullptr);
    ASSERT_TRUE(fsils_jacobian->usesOwnedRowOperator());
    int locally_owned_dt_rows = 0;
    for (GlobalIndex dof = 0; dof < n_dofs; ++dof) {
        if (fsils_jacobian->ownsFeDofRow(dof)) {
            ++locally_owned_dt_rows;
        }
    }
    EXPECT_EQ(locally_owned_dt_rows, rank == 0 ? 4 : 0);

    int ownerless_rank_count = locally_owned_dt_rows == 0 ? 1 : 0;
    int global_ownerless_rank_count = 0;
    int global_owned_dt_rows = 0;
    MPI_Allreduce(&ownerless_rank_count,
                  &global_ownerless_rank_count,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  comm);
    MPI_Allreduce(&locally_owned_dt_rows,
                  &global_owned_dt_rows,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  comm);
    ASSERT_EQ(global_ownerless_rank_count, 1);
    ASSERT_EQ(global_owned_dt_rows, n_dofs);

    timestepping::NewtonReport report{};
    ASSERT_NO_THROW(report = newton.solveStep(
                        transient,
                        *linear,
                        /*solve_time=*/dt,
                        history,
                        workspace));

    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.iterations, 1);
    EXPECT_TRUE(report.linear.converged);
    EXPECT_GT(report.linear.collective_calls, 0u);
    EXPECT_TRUE(std::isfinite(report.residual_norm0));
    EXPECT_TRUE(std::isfinite(report.residual_norm));
    EXPECT_GT(report.residual_norm0, 0.0);
    // A nonzero always-on PTC shift deliberately prevents the one-step
    // linear reaction solve from being the exact unshifted Newton update.
    EXPECT_GT(report.residual_norm, 1e-8);
    EXPECT_LT(report.residual_norm, report.residual_norm0);

    int dt_sync_min = 0;
    int dt_sync_max = 0;
    int jacobian_sync_min = 0;
    int jacobian_sync_max = 0;
    MPI_Allreduce(&dt_only_jacobian_syncs,
                  &dt_sync_min,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  comm);
    MPI_Allreduce(&dt_only_jacobian_syncs,
                  &dt_sync_max,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm);
    MPI_Allreduce(&all_jacobian_syncs,
                  &jacobian_sync_min,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  comm);
    MPI_Allreduce(&all_jacobian_syncs,
                  &jacobian_sync_max,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm);
    EXPECT_EQ(dt_sync_min, 1);
    EXPECT_EQ(dt_sync_max, 1);
    EXPECT_EQ(jacobian_sync_min, 3);
    EXPECT_EQ(jacobian_sync_max, 3);

    // The global norm is collective.  It must be nonzero on both ranks even
    // though rank one contributed no local entry to the PTC diagonal loop.
    const auto lumped_norm = workspace.ptc_mass_lumped->norm();
    EXPECT_TRUE(std::isfinite(lumped_norm));
    EXPECT_GT(lumped_norm, 0.0);

    auto lumped_view =
        workspace.ptc_mass_lumped->createAssemblyView();
    ASSERT_NE(lumped_view, nullptr);
    int local_positive_lumped_entries = 0;
    for (GlobalIndex dof = 0; dof < n_dofs; ++dof) {
        if (!fsils_jacobian->ownsFeDofRow(dof)) {
            continue;
        }
        const auto value = lumped_view->getVectorEntry(dof);
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GT(value, Real{0});
        if (value > Real{0}) {
            ++local_positive_lumped_entries;
        }
    }
    int global_positive_lumped_entries = 0;
    MPI_Allreduce(&local_positive_lumped_entries,
                  &global_positive_lumped_entries,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  comm);
    EXPECT_EQ(global_positive_lumped_entries, n_dofs);
#endif
}

TEST(TimeLoopFsilsConvergenceMPI,
     ConcurrentSplitSubcommunicatorsIsolateNewtonCollectives)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 4) {
        GTEST_SKIP() << "Run with exactly 4 MPI ranks to enable this test";
    }

    // Run two independent, concurrent two-rank Newton solves.  No world
    // collective is permitted below the split: color zero starts from the
    // exact solution and returns before a linear solve, while color one must
    // take a Newton update.  An accidental production MPI_COMM_WORLD
    // collective therefore either contaminates the color-local norm and
    // convergence assertions or blocks once the control paths diverge.
    const int color = world_rank / 2;
    MPI_Comm subcomm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &subcomm);
    ASSERT_NE(subcomm, MPI_COMM_NULL);
    struct SubcommunicatorGuard {
        MPI_Comm& communicator;
        ~SubcommunicatorGuard()
        {
            if (communicator != MPI_COMM_NULL) {
                MPI_Comm_free(&communicator);
            }
        }
    } subcomm_guard{subcomm};

    const int rank = mpiRank(subcomm);
    const int size = mpiSize(subcomm);
    ASSERT_EQ(size, 2);

    auto mesh = std::make_shared<RankLocalOwnedQuadMeshAccess>(rank);
    const auto space = spaces::Space(spaces::SpaceType::H1,
                                     ElementType::Quad4,
                                     /*order=*/1,
                                     /*components=*/1);
    ASSERT_TRUE(space);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        systems::FieldSpec{.name = "u",
                           .space = space,
                           .components = 1});
    sys.addOperator("op");

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::TestFunction(*space, "v");
    (void)systems::installFormulation(sys, "op", {u_field}, (u * v).dx());

    systems::SetupOptions setup_opts;
    setup_opts.assembler_name = "StandardAssembler";
    setup_opts.assembly_options.ghost_policy = GhostPolicy::ReverseScatter;
    setup_opts.assembly_options.deterministic = true;
    setup_opts.assembly_options.overlap_communication = false;
    setup_opts.dof_options.global_numbering =
        dofs::GlobalNumberingMode::OwnerContiguous;
    setup_opts.dof_options.ownership = dofs::OwnershipStrategy::CellOwner;
    setup_opts.dof_options.my_rank = rank;
    setup_opts.dof_options.world_size = size;
    setup_opts.dof_options.mpi_comm = subcomm;

    systems::SetupInputs inputs;
    inputs.topology_override = buildRankLocalOwnedQuadTopology(rank);
    ASSERT_NO_THROW(sys.setup(setup_opts, inputs));
    ASSERT_TRUE(sys.isSetup());

    int communicator_relation = MPI_UNEQUAL;
    MPI_Comm_compare(sys.activeMpiCommunicator(),
                     subcomm,
                     &communicator_relation);
    EXPECT_TRUE(communicator_relation == MPI_IDENT ||
                communicator_relation == MPI_CONGRUENT);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4 * size);
    backends::FsilsFactory factory(
        /*dof_per_node=*/1, sys.dofPermutation(), subcomm);
    auto linear = factory.createLinearSolver(fsilsGmresDiagOptions());
    ASSERT_TRUE(linear);

    auto history = timestepping::TimeHistory::allocate(
        factory,
        n_dofs,
        /*history_depth=*/2,
        /*allocate_second_order_state=*/false);
    constexpr double dt = 0.1;
    history.setTime(0.0);
    history.setDt(dt);
    history.setPrevDt(dt);
    history.setStepIndex(0);
    const Real initial_value = color == 0 ? Real{0} : Real{1};
    auto fill_initial = [initial_value](backends::GenericVector& vector) {
        auto values = vector.localSpan();
        std::fill(values.begin(), values.end(), initial_value);
    };
    fill_initial(history.uPrev());
    fill_initial(history.uPrev2());
    history.resetCurrentToPrevious();

    auto integrator =
        std::make_shared<systems::BackwardDifferenceIntegrator>();
    systems::TransientSystem transient(sys, std::move(integrator));

    timestepping::NewtonOptions newton_opts;
    newton_opts.residual_op = "op";
    newton_opts.jacobian_op = "op";
    newton_opts.max_iterations = 2;
    newton_opts.abs_tolerance = 1e-11;
    newton_opts.rel_tolerance = 0.0;
    newton_opts.step_tolerance = 0.0;
    newton_opts.stagnation_tolerance = 0.0;
    newton_opts.use_line_search = false;

    timestepping::NewtonSolver newton(newton_opts);
    timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(sys, factory, workspace);
    history.repack(factory);

    timestepping::NewtonReport report{};
    ASSERT_NO_THROW(report = newton.solveStep(
                        transient,
                        *linear,
                        /*solve_time=*/dt,
                        history,
                        workspace));
    EXPECT_TRUE(report.converged);
    if (color == 0) {
        EXPECT_EQ(report.iterations, 0);
        EXPECT_DOUBLE_EQ(report.residual_norm0, 0.0);
        EXPECT_DOUBLE_EQ(report.residual_norm, 0.0);
    } else {
        EXPECT_EQ(report.iterations, 1);
        EXPECT_GT(report.residual_norm0, 0.0);
        EXPECT_LT(report.residual_norm, report.residual_norm0);
        EXPECT_LE(report.residual_norm, newton_opts.abs_tolerance);
        EXPECT_TRUE(report.linear.converged);
        EXPECT_GT(report.linear.collective_calls, 0u);
    }
#endif
}

TEST(TimeLoopFsilsConvergenceMPI, DISABLED_GeneralizedAlphaMonolithicResistanceOutletsProbe)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    constexpr int n_cells = 4;
    if (size > n_cells) {
        GTEST_SKIP() << "Probe uses a fixed 4-cell strip; run with at most 4 MPI ranks";
    }

    auto sys = buildOutletCoupledTransientSystem(comm, rank, size, n_cells);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto* gauge_reg = sys->gaugeRegistryIfPresent();
    const std::size_t gauge_candidates = gauge_reg ? gauge_reg->candidates().size() : 0u;
    const std::size_t gauge_resolved = gauge_reg ? gauge_reg->resolvedModes().size() : 0u;

    const auto n_dofs = sys->dofHandler().getNumDofs();
    constexpr int dof_per_node = 3;

    dofs::DofDistributionOptions dof_options;
    dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    dof_options.my_rank = rank;
    dof_options.world_size = size;
    dof_options.mpi_comm = comm;

    auto perm = getFsilsDofPermutation(*sys, dof_per_node, dof_options);
    if (size > 1) {
        ASSERT_TRUE(perm) << "Failed to build FSILS permutation for outlet-coupled probe";
    }

    backends::FsilsFactory factory(dof_per_node, perm);
    auto linear = factory.createLinearSolver(fsilsBlockSchurOptions());
    ASSERT_TRUE(linear);

    auto history = timestepping::TimeHistory::allocate(factory,
                                                       n_dofs,
                                                       /*history_depth=*/2,
                                                       /*allocate_second_order_state=*/false);
    const double dt = 0.05;
    history.setTime(0.0);
    history.setDt(dt);
    history.setPrevDt(dt);
    history.setStepIndex(0);

    auto init = [&](backends::GenericVector& vec, double scale) {
        auto s = vec.localSpan();
        ASSERT_EQ(static_cast<GlobalIndex>(s.size()), n_dofs);
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const double sign = ((i % 3) == 1) ? -1.0 : 1.0;
            s[static_cast<std::size_t>(i)] =
                static_cast<Real>(scale * sign * 0.035 * static_cast<double>(i + 1));
        }
    };
    init(history.uPrev(), /*scale=*/1.0);
    init(history.uPrev2(), /*scale=*/0.9);
    history.resetCurrentToPrevious();

    auto integrator = std::make_shared<systems::BackwardDifferenceIntegrator>();
    systems::TransientSystem transient(*sys, std::move(integrator));

    timestepping::TimeLoopOptions loop_opts;
    loop_opts.t0 = 0.0;
    loop_opts.t_end = dt;
    loop_opts.dt = dt;
    loop_opts.max_steps = 1;
    loop_opts.scheme = timestepping::SchemeKind::GeneralizedAlpha;
    loop_opts.generalized_alpha_rho_inf = 0.5;
    loop_opts.newton.residual_op = "op";
    loop_opts.newton.jacobian_op = "op";
    loop_opts.newton.max_iterations = 12;
    loop_opts.newton.abs_tolerance = 1e-12;
    loop_opts.newton.rel_tolerance = 1e-10;

    timestepping::TimeLoop loop(loop_opts);

    std::vector<timestepping::NewtonReport> nonlinear_reports;
    timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done =
        [&nonlinear_reports](const timestepping::TimeHistory&,
                             const timestepping::NewtonReport& nr) {
            nonlinear_reports.push_back(nr);
        };

    timestepping::TimeLoopReport rep{};
    bool caught_exception = false;
    std::string caught_message{};
    try {
        rep = loop.run(transient, factory, *linear, history, callbacks);
    } catch (const FEException& e) {
        caught_exception = true;
        caught_message = e.what();
    }

    ASSERT_FALSE(nonlinear_reports.empty());
    if (rank == 0) {
        const auto& nr = nonlinear_reports.back();
        std::cout << "[mpi-gap-probe] ranks=" << size
                  << " has_gauge=" << (sys->hasGaugeRegistry() ? 1 : 0)
                  << " gauge_candidates=" << gauge_candidates
                  << " gauge_resolved=" << gauge_resolved
                  << " constraints=" << sys->constraints().numConstraints()
                  << " reports=" << nonlinear_reports.size()
                  << " caught=" << (caught_exception ? 1 : 0)
                  << " success=" << (rep.success ? 1 : 0)
                  << " converged=" << (nr.converged ? 1 : 0)
                  << " newton_iters=" << nr.iterations
                  << " linear_iters=" << nr.linear.iterations
                  << " linear_rel=" << nr.linear.relative_residual
                  << " outer=" << nr.linear.blockschur_outer_iterations
                  << " schur_iters=" << nr.linear.blockschur_schur_iterations
                  << " momentum_iters=" << nr.linear.blockschur_momentum_iterations
                  << " residual=" << nr.residual_norm
                  << " message=\"" << rep.message << "\""
                  << " caught_message=\"" << caught_message << "\""
                  << std::endl;
    }

    EXPECT_TRUE(std::isfinite(nonlinear_reports.back().residual_norm));
    EXPECT_TRUE(std::isfinite(nonlinear_reports.back().linear.relative_residual));
#endif
}

TEST(TimeLoopFsilsConvergenceMPI, DISABLED_GeneralizedAlphaMonolithicRCROutletsProbe)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    constexpr int n_cells = 4;
    if (size > n_cells) {
        GTEST_SKIP() << "Probe uses a fixed 4-cell strip; run with at most 4 MPI ranks";
    }

    auto sys = buildOutletCoupledTransientSystemRCR(comm, rank, size, n_cells);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto* gauge_reg = sys->gaugeRegistryIfPresent();
    const std::size_t gauge_candidates = gauge_reg ? gauge_reg->candidates().size() : 0u;
    const std::size_t gauge_resolved = gauge_reg ? gauge_reg->resolvedModes().size() : 0u;

    const auto n_dofs = sys->dofHandler().getNumDofs();
    constexpr int dof_per_node = 3;

    dofs::DofDistributionOptions dof_options;
    dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    dof_options.my_rank = rank;
    dof_options.world_size = size;
    dof_options.mpi_comm = comm;

    auto perm = getFsilsDofPermutation(*sys, dof_per_node, dof_options);
    if (size > 1) {
        ASSERT_TRUE(perm) << "Failed to build FSILS permutation for outlet-coupled RCR probe";
    }

    backends::FsilsFactory factory(dof_per_node, perm);
    auto linear = factory.createLinearSolver(fsilsBlockSchurOptions());
    ASSERT_TRUE(linear);

    auto history = timestepping::TimeHistory::allocate(factory,
                                                       n_dofs,
                                                       /*history_depth=*/2,
                                                       /*allocate_second_order_state=*/false);
    const double dt = 0.05;
    history.setTime(0.0);
    history.setDt(dt);
    history.setPrevDt(dt);
    history.setStepIndex(0);

    auto init = [&](backends::GenericVector& vec, double scale) {
        auto s = vec.localSpan();
        ASSERT_EQ(static_cast<GlobalIndex>(s.size()), n_dofs);
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const double sign = ((i % 3) == 1) ? -1.0 : 1.0;
            s[static_cast<std::size_t>(i)] =
                static_cast<Real>(scale * sign * 0.035 * static_cast<double>(i + 1));
        }
    };
    init(history.uPrev(), /*scale=*/1.0);
    init(history.uPrev2(), /*scale=*/0.9);
    history.resetCurrentToPrevious();

    auto integrator = std::make_shared<systems::BackwardDifferenceIntegrator>();
    systems::TransientSystem transient(*sys, std::move(integrator));

    timestepping::TimeLoopOptions loop_opts;
    loop_opts.t0 = 0.0;
    loop_opts.t_end = dt;
    loop_opts.dt = dt;
    loop_opts.max_steps = 1;
    loop_opts.scheme = timestepping::SchemeKind::GeneralizedAlpha;
    loop_opts.generalized_alpha_rho_inf = 0.5;
    loop_opts.newton.residual_op = "op";
    loop_opts.newton.jacobian_op = "op";
    loop_opts.newton.max_iterations = 12;
    loop_opts.newton.abs_tolerance = 1e-12;
    loop_opts.newton.rel_tolerance = 1e-10;

    timestepping::TimeLoop loop(loop_opts);

    std::vector<timestepping::NewtonReport> nonlinear_reports;
    timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done =
        [&nonlinear_reports](const timestepping::TimeHistory&,
                             const timestepping::NewtonReport& nr) {
            nonlinear_reports.push_back(nr);
        };

    timestepping::TimeLoopReport rep{};
    bool caught_exception = false;
    std::string caught_message{};
    try {
        rep = loop.run(transient, factory, *linear, history, callbacks);
    } catch (const FEException& e) {
        caught_exception = true;
        caught_message = e.what();
    }

    ASSERT_FALSE(nonlinear_reports.empty());
    if (rank == 0) {
        const auto& nr = nonlinear_reports.back();
        std::cout << "[mpi-gap-probe-rcr] ranks=" << size
                  << " has_gauge=" << (sys->hasGaugeRegistry() ? 1 : 0)
                  << " gauge_candidates=" << gauge_candidates
                  << " gauge_resolved=" << gauge_resolved
                  << " constraints=" << sys->constraints().numConstraints()
                  << " reports=" << nonlinear_reports.size()
                  << " caught=" << (caught_exception ? 1 : 0)
                  << " success=" << (rep.success ? 1 : 0)
                  << " converged=" << (nr.converged ? 1 : 0)
                  << " newton_iters=" << nr.iterations
                  << " linear_iters=" << nr.linear.iterations
                  << " linear_rel=" << nr.linear.relative_residual
                  << " outer=" << nr.linear.blockschur_outer_iterations
                  << " schur_iters=" << nr.linear.blockschur_schur_iterations
                  << " momentum_iters=" << nr.linear.blockschur_momentum_iterations
                  << " residual=" << nr.residual_norm
                  << " caught_message=\"" << caught_message << "\""
                  << std::endl;
        EXPECT_FALSE(caught_exception);
        EXPECT_TRUE(rep.success);
        EXPECT_TRUE(nr.converged);
        EXPECT_TRUE(nr.linear.converged);
    }
#endif
}

TEST(TimeLoopFsilsConvergenceMPI,
     RankDivergentExplicitBorderedRecoveryFailsCollectivelyAndRestoresState)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    constexpr int n_cells = 4;
    if (size < 2 || size > n_cells) {
        GTEST_SKIP() << "Run with 2-4 MPI ranks to enable this test";
    }

    auto sys = buildOutletCoupledTransientSystemRCR(
        comm, rank, size, n_cells);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());

    const auto n_dofs = sys->dofHandler().getNumDofs();
    constexpr int dof_per_node = 3;
    dofs::DofDistributionOptions dof_options;
    dof_options.global_numbering =
        dofs::GlobalNumberingMode::DenseGlobalIds;
    dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    dof_options.my_rank = rank;
    dof_options.world_size = size;
    dof_options.mpi_comm = comm;

    auto permutation =
        getFsilsDofPermutation(*sys, dof_per_node, dof_options);
    ASSERT_TRUE(permutation)
        << "Failed to build the distributed RCR test permutation";

    backends::FsilsFactory factory(dof_per_node, permutation);
    auto inner_linear =
        factory.createLinearSolver(fsilsGmresDiagOptions());
    ASSERT_TRUE(inner_linear);

    auto history = timestepping::TimeHistory::allocate(
        factory,
        n_dofs,
        /*history_depth=*/2,
        /*allocate_second_order_state=*/false);
    constexpr double dt = 0.05;
    history.setTime(0.0);
    history.setDt(dt);
    history.setPrevDt(dt);
    history.setStepIndex(0);
    history.uPrev().zero();
    history.uPrev2().zero();
    history.resetCurrentToPrevious();

    auto integrator =
        std::make_shared<systems::BackwardDifferenceIntegrator>();
    systems::TransientSystem transient(*sys, std::move(integrator));

    timestepping::NewtonOptions options;
    options.residual_op = "op";
    options.jacobian_op = "op";
    options.max_iterations = 1;
    options.abs_tolerance = 1e-14;
    options.rel_tolerance = 0.0;
    options.step_tolerance = 0.0;
    options.stagnation_tolerance = 0.0;
    options.use_line_search = false;

    timestepping::NewtonSolver newton(options);
    timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(*sys, factory, workspace);
    history.repack(factory);

    auto snapshot = [](const backends::GenericVector& vector) {
        const auto values = vector.localSpan();
        return std::vector<Real>(values.begin(), values.end());
    };
    const auto entry_u = snapshot(history.u());
    const auto entry_u_prev = snapshot(history.uPrev());
    const auto entry_u_prev2 = snapshot(history.uPrev2());
    const auto entry_auxiliary = sys->checkpointAuxiliaryState();

    RankDivergentBorderedRecoveryLinearSolver scripted(
        *inner_linear, rank);
    bool caught = false;
    std::string caught_message;
    try {
        (void)newton.solveStep(
            transient,
            scripted,
            /*solve_time=*/dt,
            history,
            workspace);
    } catch (const FEException& error) {
        caught = true;
        caught_message = error.what();
    }

    const int local_caught = caught ? 1 : 0;
    int caught_min = 0;
    int caught_max = 0;
    int solve_calls_min = 0;
    int solve_calls_max = 0;
    MPI_Allreduce(&local_caught,
                  &caught_min,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  comm);
    MPI_Allreduce(&local_caught,
                  &caught_max,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm);
    MPI_Allreduce(&scripted.solve_calls,
                  &solve_calls_min,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  comm);
    MPI_Allreduce(&scripted.solve_calls,
                  &solve_calls_max,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm);

    EXPECT_EQ(caught_min, 1);
    EXPECT_EQ(caught_max, 1);
    EXPECT_NE(caught_message.find("bordered K^{-1}B solve did not converge"),
              std::string::npos)
        << caught_message;
    EXPECT_EQ(scripted.divergent_report_injections, 1);
    EXPECT_EQ(solve_calls_min, solve_calls_max);
    EXPECT_EQ(solve_calls_min, 4)
        << "All ranks must run the main solve, one bordered column solve, "
           "and both polish attempts";

    EXPECT_EQ(snapshot(history.u()), entry_u);
    EXPECT_EQ(snapshot(history.uPrev()), entry_u_prev);
    EXPECT_EQ(snapshot(history.uPrev2()), entry_u_prev2);
    EXPECT_EQ(sys->checkpointAuxiliaryState(), entry_auxiliary);
    ASSERT_NE(workspace.delta, nullptr);
    for (const auto value : workspace.delta->localSpan()) {
        EXPECT_EQ(value, Real{0});
    }
#endif
}

TEST(TimeLoopFsilsConvergenceMPI, FixedStepRejectsNonconvergedNewtonStep)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const int n_cells = size;
    auto mesh = std::make_shared<StripQuadMeshAccess>(n_cells, rank, size);
    const auto space = spaces::Space(spaces::SpaceType::H1, ElementType::Quad4, /*order=*/1, /*components=*/1);
    ASSERT_TRUE(space);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::TestFunction(*space, "v");
    const auto one = forms::FormExpr::constant(Real(1.0));
    const auto lambda = forms::FormExpr::constant(Real(1.0));
    const auto beta = forms::FormExpr::constant(Real(0.5));
    const auto residual = (u.dt(1) * v + lambda * u * v + beta * (one + u * u) * u * v).dx();

    (void)systems::installFormulation(sys, "op", {u_field}, residual);

    systems::SetupOptions setup_opts;
    setup_opts.assembler_name = "StandardAssembler";
    setup_opts.assembly_options.ghost_policy = GhostPolicy::ReverseScatter;
    setup_opts.assembly_options.deterministic = true;
    setup_opts.assembly_options.overlap_communication = false;
    setup_opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    setup_opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    setup_opts.dof_options.my_rank = rank;
    setup_opts.dof_options.world_size = size;
    setup_opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = buildStripTopology(n_cells, rank, size);

    sys.setup(setup_opts, inputs);
    ASSERT_TRUE(sys.isSetup());

    const GlobalIndex n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_TRUE(inputs.topology_override.has_value());
    const GlobalIndex n_nodes = inputs.topology_override->n_vertices;
    ASSERT_EQ(n_dofs, n_nodes);

    constexpr int dof_per_node = 1;
    auto perm = getFsilsDofPermutation(sys, dof_per_node, setup_opts.dof_options);
    if (size > 1) {
        ASSERT_TRUE(perm) << "Failed to build FSILS DOF permutation for scalar test system";
    }

    backends::FsilsFactory factory(dof_per_node, perm);
    auto linear = factory.createLinearSolver(fsilsGmresDiagOptions());
    ASSERT_TRUE(linear);

    auto history = timestepping::TimeHistory::allocate(factory, n_dofs, /*history_depth=*/2,
                                                       /*allocate_second_order_state=*/false);
    const double dt = 0.1;
    history.setTime(0.0);
    history.setDt(dt);
    history.setPrevDt(dt);
    history.setStepIndex(0);

    auto init = [&](backends::GenericVector& vec, double scale) {
        auto s = vec.localSpan();
        ASSERT_EQ(static_cast<GlobalIndex>(s.size()), n_dofs);
        for (GlobalIndex node = 0; node < n_nodes; ++node) {
            s[static_cast<std::size_t>(node)] =
                static_cast<Real>(scale * 0.05 * static_cast<double>(node + 1));
        }
    };
    init(history.uPrev(), /*scale=*/1.0);
    init(history.uPrev2(), /*scale=*/1.0);
    history.resetCurrentToPrevious();

    auto integrator = std::make_shared<systems::BackwardDifferenceIntegrator>();
    systems::TransientSystem transient(sys, std::move(integrator));

    timestepping::TimeLoopOptions loop_opts;
    loop_opts.t0 = 0.0;
    loop_opts.t_end = dt;
    loop_opts.dt = dt;
    loop_opts.max_steps = 1;
    loop_opts.scheme = timestepping::SchemeKind::BackwardEuler;
    loop_opts.newton.residual_op = "op";
    loop_opts.newton.jacobian_op = "op";
    loop_opts.newton.max_iterations = 1;
    loop_opts.newton.abs_tolerance = 1e-12;
    loop_opts.newton.rel_tolerance = 0.0;

    timestepping::TimeLoop loop(loop_opts);

    int nonconverged_steps = 0;
    int accepted_steps = 0;
    timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&nonconverged_steps](const timestepping::TimeHistory&,
                                                        const timestepping::NewtonReport& nr) {
        if (!nr.converged) {
            ++nonconverged_steps;
        }
    };
    callbacks.on_step_accepted = [&accepted_steps](const timestepping::TimeHistory&) {
        ++accepted_steps;
    };

    EXPECT_THROW((void)loop.run(transient, factory, *linear, history, callbacks), FEException);
    EXPECT_EQ(nonconverged_steps, 1);
    EXPECT_EQ(accepted_steps, 0);
#endif
}

} // namespace svmp::FE::assembly::testing

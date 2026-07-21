/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_SmallCutAggregationConstraintMPI.cpp
 * @brief MPI regressions for communicator-global small-cut aggregation.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Constraints/VertexDirichletConstraint.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Geometry/CutQuadrature.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

namespace {

constexpr int kInterfaceMarker = 7;
constexpr int kWallMarker = 11;

class TwoQuadAggregationMeshAccess final : public assembly::IMeshAccess {
public:
    TwoQuadAggregationMeshAccess(int rank,
                                 bool full_view,
                                 bool expose_left_wall,
                                 bool rank_one_empty = false,
                                 bool all_cells_owned_by_rank_zero = false)
        : rank_(rank),
          full_view_(full_view),
          expose_left_wall_(expose_left_wall),
          rank_one_empty_(rank_one_empty),
          all_cells_owned_by_rank_zero_(all_cells_owned_by_rank_zero)
    {
        if (rank_one_empty_ && rank_ == 1) {
            return;
        }
        if (full_view_) {
            nodes_ = {
                {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
                {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0},
                {2.0, 0.0, 0.0}, {2.0, 1.0, 0.0},
            };
            cells_ = {
                std::array<GlobalIndex, 4>{0, 2, 3, 1},
                std::array<GlobalIndex, 4>{2, 4, 5, 3},
            };
        } else {
            nodes_ = {
                {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
            };
            cells_ = {std::array<GlobalIndex, 4>{0, 1, 2, 3}};
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }
    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        if (rank_one_empty_ && rank_ == 1) {
            return 0;
        }
        if (all_cells_owned_by_rank_zero_) {
            return rank_ == 0 ? numCells() : 0;
        }
        return 1;
    }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override
    {
        return expose_left_wall_ ? 1 : 0;
    }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override
    {
        return full_view_ ? 1 : 0;
    }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        if (all_cells_owned_by_rank_zero_) {
            return rank_ == 0;
        }
        return full_view_ ? static_cast<int>(cell) == rank_ : true;
    }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Quad4;
    }
    void getCellNodes(GlobalIndex cell,
                      std::vector<GlobalIndex>& nodes) const override
    {
        const auto& connectivity = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(connectivity.begin(), connectivity.end());
    }
    [[nodiscard]] std::array<Real, 3>
    getNodeCoordinates(GlobalIndex node) const override
    {
        return nodes_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(
        GlobalIndex cell,
        std::vector<std::array<Real, 3>>& coordinates) const override
    {
        const auto& connectivity = cells_.at(static_cast<std::size_t>(cell));
        coordinates.resize(connectivity.size());
        for (std::size_t i = 0; i < connectivity.size(); ++i) {
            coordinates[i] =
                nodes_.at(static_cast<std::size_t>(connectivity[i]));
        }
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex,
                                               GlobalIndex cell) const override
    {
        return cell == 0 ? LocalIndex{3} : INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override
    {
        return kWallMarker;
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex) const override
    {
        return {0, 1};
    }
    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < numCells(); ++cell) {
            callback(cell);
        }
    }
    void forEachOwnedCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        if (rank_one_empty_ && rank_ == 1) {
            return;
        }
        if (all_cells_owned_by_rank_zero_) {
            if (rank_ == 0) {
                for (GlobalIndex cell = 0; cell < numCells(); ++cell) {
                    callback(cell);
                }
            }
            return;
        }
        callback(full_view_ ? static_cast<GlobalIndex>(rank_) : 0);
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (expose_left_wall_ && (marker < 0 || marker == kWallMarker)) {
            callback(0, 0);
        }
    }
    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback)
        const override
    {
        if (full_view_) {
            callback(0, 0, 1);
        }
    }

private:
    int rank_{0};
    bool full_view_{true};
    bool expose_left_wall_{false};
    bool rank_one_empty_{false};
    bool all_cells_owned_by_rank_zero_{false};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

class FiveQuadAggregationMeshAccess final : public assembly::IMeshAccess {
public:
    explicit FiveQuadAggregationMeshAccess(int rank) : rank_(rank)
    {
        const int first_x = rank_ == 0 ? 0 : 1;
        const int last_x = 5;
        for (int x = first_x; x <= last_x; ++x) {
            nodes_.push_back(
                {static_cast<Real>(x), Real{0}, Real{0}});
            nodes_.push_back(
                {static_cast<Real>(x), Real{1}, Real{0}});
        }
        const int cell_count = rank_ == 0 ? 5 : 4;
        for (int cell = 0; cell < cell_count; ++cell) {
            const auto left = static_cast<GlobalIndex>(2 * cell);
            cells_.push_back({left, left + 2, left + 3, left + 1});
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }
    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        return rank_ == 0 ? 2 : 3;
    }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override
    {
        return static_cast<GlobalIndex>(cells_.size() - 1u);
    }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        const auto global_cell = cell + (rank_ == 0 ? 0 : 1);
        return rank_ == 0 ? global_cell < 2 : global_cell >= 2;
    }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Quad4;
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex,
                                               GlobalIndex) const override
    {
        return INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override
    {
        return -1;
    }
    void getCellNodes(GlobalIndex cell,
                      std::vector<GlobalIndex>& nodes) const override
    {
        const auto& connectivity = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(connectivity.begin(), connectivity.end());
    }
    [[nodiscard]] std::array<Real, 3>
    getNodeCoordinates(GlobalIndex node) const override
    {
        return nodes_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(
        GlobalIndex cell,
        std::vector<std::array<Real, 3>>& coordinates) const override
    {
        const auto& connectivity = cells_.at(static_cast<std::size_t>(cell));
        coordinates.resize(connectivity.size());
        for (std::size_t i = 0; i < connectivity.size(); ++i) {
            coordinates[i] =
                nodes_.at(static_cast<std::size_t>(connectivity[i]));
        }
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face) const override
    {
        return {face, face + 1};
    }
    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        if (rank_ == 1) {
            // Enumerate only rank-1-owned cells. The rank-0 view includes the
            // whole band so it can propose both the nearer unavailable root
            // and a farther fallback; global owner face signatures still have
            // to connect the split ownership boundary without mutual halos.
            for (GlobalIndex cell = 1; cell < numCells(); ++cell) {
                callback(cell);
            }
            return;
        }
        for (GlobalIndex cell = 0; cell < numCells(); ++cell) {
            callback(cell);
        }
    }
    void forEachOwnedCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < numCells(); ++cell) {
            if (isOwnedCell(cell)) {
                callback(cell);
            }
        }
    }
    void forEachBoundaryFace(
        int,
        std::function<void(GlobalIndex, GlobalIndex)>) const override
    {
    }
    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback)
        const override
    {
        for (GlobalIndex face = 0; face < numInteriorFaces(); ++face) {
            callback(face, face, face + 1);
        }
    }

private:
    int rank_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

dofs::MeshTopologyInfo fullTopology()
{
    dofs::MeshTopologyInfo topology;
    topology.dim = 2;
    topology.n_cells = 2;
    topology.n_vertices = 6;
    topology.cell2vertex_offsets = {0, 4, 8};
    topology.cell2vertex_data = {0, 2, 3, 1, 2, 4, 5, 3};
    topology.vertex_gids = {0, 1, 2, 3, 4, 5};
    topology.vertex_coords = {
        0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 2.0, 0.0, 2.0, 1.0,
    };
    topology.cell_gids = {0, 1};
    topology.cell_owner_ranks = {0, 1};
    topology.neighbor_ranks = {0, 1};
    return topology;
}

dofs::MeshTopologyInfo leftCellOnlyTopology()
{
    dofs::MeshTopologyInfo topology;
    topology.dim = 2;
    topology.n_cells = 1;
    topology.n_vertices = 4;
    topology.cell2vertex_offsets = {0, 4};
    topology.cell2vertex_data = {0, 1, 2, 3};
    // Local node order is (gid 0, gid 2, gid 3, gid 1).
    topology.vertex_gids = {0, 2, 3, 1};
    topology.vertex_coords = {
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
    };
    topology.cell_gids = {0};
    topology.cell_owner_ranks = {0};
    topology.neighbor_ranks = {1};
    return topology;
}

dofs::MeshTopologyInfo allCellsOnRankZeroTopology(int rank)
{
    // FESystem setup requires a nonempty topology on every rank. Rank 1 keeps
    // an all-ghost DOF view so setup collectives remain well formed, while its
    // IMeshAccess below enumerates zero cells/rules/support for aggregation.
    auto topology = fullTopology();
    topology.cell_owner_ranks = {0, 0};
    topology.neighbor_ranks = {rank == 0 ? 1 : 0};
    return topology;
}

dofs::MeshTopologyInfo fiveQuadTopology(int rank)
{
    dofs::MeshTopologyInfo topology;
    topology.dim = 2;
    const int first_x = rank == 0 ? 0 : 1;
    const int n_cells = rank == 0 ? 5 : 4;
    const int n_vertices = rank == 0 ? 12 : 10;
    topology.n_cells = n_cells;
    topology.n_vertices = n_vertices;
    topology.cell2vertex_offsets.reserve(
        static_cast<std::size_t>(n_cells + 1));
    topology.cell2vertex_offsets.push_back(0);
    for (int cell = 0; cell < n_cells; ++cell) {
        const int left = 2 * cell;
        topology.cell2vertex_data.insert(
            topology.cell2vertex_data.end(),
            {left, left + 2, left + 3, left + 1});
        topology.cell2vertex_offsets.push_back(
            static_cast<GlobalIndex>(topology.cell2vertex_data.size()));
    }
    for (int local_x = 0; local_x <= n_cells; ++local_x) {
        const auto global_x = first_x + local_x;
        topology.vertex_gids.push_back(2 * global_x);
        topology.vertex_gids.push_back(2 * global_x + 1);
        topology.vertex_coords.insert(
            topology.vertex_coords.end(),
            {static_cast<Real>(global_x), Real{0},
             static_cast<Real>(global_x), Real{1}});
    }
    for (int local_cell = 0; local_cell < n_cells; ++local_cell) {
        const auto global_cell = first_x + local_cell;
        topology.cell_gids.push_back(global_cell);
        topology.cell_owner_ranks.push_back(global_cell < 2 ? 0 : 1);
    }
    topology.neighbor_ranks = {rank == 0 ? 1 : 0};
    return topology;
}

systems::SetupOptions setupOptions(int rank,
                                  int world_size,
                                  MPI_Comm comm = MPI_COMM_WORLD)
{
    systems::SetupOptions options;
    options.dof_options.global_numbering =
        dofs::GlobalNumberingMode::OwnerContiguous;
    options.dof_options.ownership = dofs::OwnershipStrategy::CellOwner;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = world_size;
    options.dof_options.mpi_comm = comm;
    return options;
}

struct CellRule {
    GlobalIndex cell{-1};
    Real fraction{0};
    bool full{false};
    geometry::CutIntegrationSide side{
        geometry::CutIntegrationSide::Negative};
};

std::shared_ptr<assembly::CutIntegrationContext> cutContext(
    std::initializer_list<CellRule> cell_rules)
{
    auto context = std::make_shared<assembly::CutIntegrationContext>();
    for (const auto& cell_rule : cell_rules) {
        assembly::CutCellAssemblyMetadata metadata{};
        metadata.cell = cell_rule.cell;
        metadata.parent_entity = cell_rule.cell;
        metadata.side = cell_rule.side;
        metadata.volume_fraction = cell_rule.fraction;

        geometry::CutQuadratureRule rule{};
        rule.kind = geometry::CutQuadratureKind::Volume;
        rule.side = cell_rule.side;
        rule.measure = cell_rule.fraction;
        rule.parent_measure = Real{1};
        rule.volume_fraction = cell_rule.fraction;
        rule.full_cell_equivalent = cell_rule.full;
        context->addGeneratedVolumeRule(kInterfaceMarker, metadata, rule);
    }
    return context;
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* key, const char* value) : key_(key)
    {
        if (const char* prior = std::getenv(key_)) {
            prior_ = std::string(prior);
        }
        ::setenv(key_, value, 1);
    }

    ~ScopedEnvVar()
    {
        if (prior_.has_value()) {
            ::setenv(key_, prior_->c_str(), 1);
        } else {
            ::unsetenv(key_);
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* key_;
    std::optional<std::string> prior_;
};

struct CollectiveOutcome {
    int minimum_threw{0};
    int maximum_threw{0};
    std::string local_message{};

    [[nodiscard]] bool allSucceeded() const noexcept
    {
        return maximum_threw == 0;
    }
    [[nodiscard]] bool allThrew() const noexcept
    {
        return minimum_threw == 1;
    }
};

template <typename Callable>
CollectiveOutcome invokeCollectively(MPI_Comm comm, Callable&& callable)
{
    int local_threw = 0;
    std::string message;
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception& error) {
        local_threw = 1;
        message = error.what();
    } catch (...) {
        local_threw = 1;
        message = "non-std exception";
    }

    CollectiveOutcome outcome;
    outcome.local_message = std::move(message);
    MPI_Allreduce(&local_threw,
                  &outcome.minimum_threw,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  comm);
    MPI_Allreduce(&local_threw,
                  &outcome.maximum_threw,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm);
    return outcome;
}

bool anyMessageContains(MPI_Comm comm,
                        const CollectiveOutcome& outcome,
                        const std::string& needle)
{
    const int local = outcome.local_message.find(needle) != std::string::npos
                          ? 1
                          : 0;
    int any = 0;
    MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, comm);
    return any != 0;
}

GlobalIndex vertexDof(const systems::FESystem& system,
                      FieldId field,
                      GlobalIndex local_vertex)
{
    const auto* entity_map = system.fieldDofHandler(field).getEntityDofMap();
    EXPECT_NE(entity_map, nullptr);
    if (entity_map == nullptr) {
        return -1;
    }
    const auto dofs = entity_map->getVertexDofs(local_vertex);
    EXPECT_EQ(dofs.size(), 1u);
    return dofs.empty() ? -1 : system.fieldDofOffset(field) + dofs.front();
}

std::vector<std::pair<GlobalIndex, double>> lineEntries(
    const systems::FESystem& system,
    GlobalIndex slave)
{
    std::vector<std::pair<GlobalIndex, double>> entries;
    const auto line = system.constraints().getConstraint(slave);
    EXPECT_TRUE(line.has_value());
    if (!line.has_value()) {
        return entries;
    }
    for (const auto& entry : line->entries) {
        entries.emplace_back(entry.master_dof, entry.weight);
    }
    std::sort(entries.begin(), entries.end());
    return entries;
}

void expectEveryMasterLineFiniteAndPartitionOfUnity(
    const systems::FESystem& system)
{
    std::size_t master_lines = 0u;
    system.constraints().forEach(
        [&](const AffineConstraints::ConstraintView& line) {
            if (line.entries.empty()) {
                return;
            }
            ++master_lines;
            long double sum = 0.0L;
            long double l1 = 0.0L;
            for (const auto& entry : line.entries) {
                EXPECT_GE(entry.master_dof, 0);
                EXPECT_TRUE(std::isfinite(entry.weight));
                sum += static_cast<long double>(entry.weight);
                l1 += std::abs(static_cast<long double>(entry.weight));
            }
            const auto tolerance =
                1.0e-10L * std::max(1.0L, l1) +
                64.0L * std::numeric_limits<double>::epsilon() *
                    std::max(1.0L, l1);
            EXPECT_NEAR(static_cast<double>(sum),
                        1.0,
                        static_cast<double>(tolerance))
                << "slave " << line.slave_dof;
        });
    EXPECT_GT(master_lines, 0u);
}

} // namespace

TEST(SmallCutAggregationConstraintMPI,
     FullOverlapOwnerNonCandidateImportsCanonicalGhostRoot)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }

    auto mesh = std::make_shared<TwoQuadAggregationMeshAccess>(
        rank, /*full_view=*/true, /*expose_left_wall=*/false);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));

    systems::SetupInputs inputs;
    inputs.topology_override = fullTopology();
    const auto setup_outcome = invokeCollectively(MPI_COMM_WORLD, [&] {
        system.setup(setupOptions(rank, world_size), inputs);
    });
    ASSERT_TRUE(setup_outcome.allSucceeded())
        << setup_outcome.local_message;

    // Rank 0 owns the two left slave vertices but has no local cut
    // declaration for them. Rank 1 sees the cut and computes the root line.
    system.setCutIntegrationContext(
        rank == 0
            ? cutContext({{1, Real{1}, true}})
            : cutContext({{0, Real{0.25}, false}, {1, Real{1}, true}}));
    const auto rebuild_outcome = invokeCollectively(
        MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
    ASSERT_TRUE(rebuild_outcome.allSucceeded())
        << rebuild_outcome.local_message;

    const auto bottom_slave = vertexDof(system, pressure, 0);
    const auto top_slave = vertexDof(system, pressure, 1);
    ASSERT_TRUE(system.dofHandler().getPartition().isOwned(bottom_slave) ==
                (rank == 0));
    EXPECT_TRUE(system.constraints().isConstrained(bottom_slave));
    EXPECT_TRUE(system.constraints().isConstrained(top_slave));

    const auto bottom_interface = vertexDof(system, pressure, 2);
    const auto bottom_far = vertexDof(system, pressure, 4);
    const auto top_interface = vertexDof(system, pressure, 3);
    const auto top_far = vertexDof(system, pressure, 5);
    EXPECT_EQ(lineEntries(system, bottom_slave),
              (std::vector<std::pair<GlobalIndex, double>>{
                  {bottom_interface, 2.0}, {bottom_far, -1.0}}));
    EXPECT_EQ(lineEntries(system, top_slave),
              (std::vector<std::pair<GlobalIndex, double>>{
                  {top_interface, 2.0}, {top_far, -1.0}}));
    expectEveryMasterLineFiniteAndPartitionOfUnity(system);
#endif
}

TEST(SmallCutAggregationConstraintMPI,
     CanonicalRootFailsClosedWhenSlaveRankLacksMaster)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }

    const bool full_view = rank == 1;
    auto mesh = std::make_shared<TwoQuadAggregationMeshAccess>(
        rank, full_view, /*expose_left_wall=*/false);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));

    systems::SetupInputs inputs;
    inputs.topology_override =
        full_view ? fullTopology() : leftCellOnlyTopology();
    const auto setup_outcome = invokeCollectively(MPI_COMM_WORLD, [&] {
        system.setup(setupOptions(rank, world_size), inputs);
    });
    ASSERT_TRUE(setup_outcome.allSucceeded())
        << setup_outcome.local_message;
    system.setCutIntegrationContext(
        full_view
            ? cutContext({{0, Real{0.25}, false}, {1, Real{1}, true}})
            : cutContext({{0, Real{0.25}, false}}));

    const auto rebuild_outcome = invokeCollectively(
        MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
    EXPECT_TRUE(rebuild_outcome.allThrew());
    EXPECT_TRUE(anyMessageContains(
        MPI_COMM_WORLD,
        rebuild_outcome,
        "incomplete_distributed_aggregation_halo"));
    EXPECT_TRUE(anyMessageContains(
        MPI_COMM_WORLD, rebuild_outcome, "canonical_master_not_relevant"));
    EXPECT_EQ(system.constraints().numConstraints(), 0u);
#endif
}

TEST(SmallCutAggregationConstraintMPI,
     BoundaryExclusionOnOwnerSuppressesGhostProposalGlobally)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }

    auto mesh = std::make_shared<TwoQuadAggregationMeshAccess>(
        rank, /*full_view=*/true, /*expose_left_wall=*/rank == 0);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<VertexDirichletConstraint>(
        pressure,
        std::vector<VertexDirichletValue>{{.vertex_id = 0,
                                           .value = Real{5}}},
        VertexIdMode::LocalVertexId));
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker,
        std::vector<int>{kWallMarker}));

    systems::SetupInputs inputs;
    inputs.topology_override = fullTopology();
    const auto setup_outcome = invokeCollectively(MPI_COMM_WORLD, [&] {
        system.setup(setupOptions(rank, world_size), inputs);
    });
    ASSERT_TRUE(setup_outcome.allSucceeded())
        << setup_outcome.local_message;
    system.setCutIntegrationContext(
        cutContext({{0, Real{0.25}, false}, {1, Real{1}, true}}));
    const auto rebuild_outcome = invokeCollectively(
        MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
    ASSERT_TRUE(rebuild_outcome.allSucceeded())
        << rebuild_outcome.local_message;

    const auto strong_dof = vertexDof(system, pressure, 0);
    const auto strong_line = system.constraints().getConstraint(strong_dof);
    ASSERT_TRUE(strong_line.has_value());
    EXPECT_TRUE(strong_line->isDirichlet());
    EXPECT_NEAR(strong_line->inhomogeneity, 5.0, 1.0e-15);
    EXPECT_FALSE(system.constraints().isConstrained(
        vertexDof(system, pressure, 1)));
#endif
}

TEST(SmallCutAggregationConstraintMPI,
     UnavailableLowerKeyRootFallsBackToGloballyVisibleRoot)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }

    auto mesh = std::make_shared<FiveQuadAggregationMeshAccess>(rank);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));

    systems::SetupInputs inputs;
    inputs.topology_override = fiveQuadTopology(rank);
    const auto setup_outcome = invokeCollectively(MPI_COMM_WORLD, [&] {
        system.setup(setupOptions(rank, world_size), inputs);
    });
    ASSERT_TRUE(setup_outcome.allSucceeded())
        << setup_outcome.local_message;
    system.setCutIntegrationContext(
        rank == 0
            ? cutContext({{0, Real{1}, true},
                          {1, Real{0.25}, false},
                          {2, Real{0.25}, false},
                          {3, Real{0.25}, false},
                          {4, Real{1}, true}})
            : cutContext({{1, Real{0.25}, false},
                          {2, Real{0.25}, false},
                          {3, Real{1}, true}}));
    const auto rebuild_outcome = invokeCollectively(
        MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
    ASSERT_TRUE(rebuild_outcome.allSucceeded())
        << rebuild_outcome.local_message;

    // The two x=2 vertices touch only cut cells. Rank 0 can see both full
    // roots and locally prefers the lower-key x=[0,1] cell at distance one,
    // but rank 1 does not carry its x=0 master. Candidate-level validation
    // must therefore continue to rank 0's farther x=[4,5] proposal (distance
    // two), which is visible on every slave-relevant rank.
    const auto candidate_bottom_local = rank == 0 ? 4 : 2;
    const auto candidate_top_local = rank == 0 ? 5 : 3;
    const auto right_near_bottom_local = rank == 0 ? 8 : 6;
    const auto right_near_top_local = rank == 0 ? 9 : 7;
    const auto right_far_bottom_local = rank == 0 ? 10 : 8;
    const auto right_far_top_local = rank == 0 ? 11 : 9;
    const auto bottom_slave =
        vertexDof(system, pressure, candidate_bottom_local);
    const auto top_slave = vertexDof(system, pressure, candidate_top_local);
    EXPECT_TRUE(system.constraints().isConstrained(bottom_slave));
    EXPECT_TRUE(system.constraints().isConstrained(top_slave));
    EXPECT_EQ(
        lineEntries(system, bottom_slave),
        (std::vector<std::pair<GlobalIndex, double>>{
            {vertexDof(system, pressure, right_near_bottom_local), 3.0},
            {vertexDof(system, pressure, right_far_bottom_local), -2.0}}));
    EXPECT_EQ(
        lineEntries(system, top_slave),
        (std::vector<std::pair<GlobalIndex, double>>{
            {vertexDof(system, pressure, right_near_top_local), 3.0},
            {vertexDof(system, pressure, right_far_top_local), -2.0}}));
    expectEveryMasterLineFiniteAndPartitionOfUnity(system);
#endif
}

TEST(SmallCutAggregationConstraintMPI,
     ConflictingPositiveCellClassFactsFailCollectivelyAndRecover)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }

    auto mesh = std::make_shared<TwoQuadAggregationMeshAccess>(
        rank, /*full_view=*/true, /*expose_left_wall=*/false);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));

    systems::SetupInputs inputs;
    inputs.topology_override = fullTopology();
    const auto setup_outcome = invokeCollectively(MPI_COMM_WORLD, [&] {
        system.setup(setupOptions(rank, world_size), inputs);
    });
    ASSERT_TRUE(setup_outcome.allSucceeded())
        << setup_outcome.local_message;

    // Cell 0 is positively declared cut on rank 0 but inactive-full (flags=0)
    // on rank 1. Absence would be legal for an owner-filtered context; two
    // different declarations for the same stable cell key are not.
    system.setCutIntegrationContext(
        rank == 0
            ? cutContext({{0, Real{0.25}, false}, {1, Real{1}, true}})
            : cutContext({
                  {0,
                   Real{1},
                   true,
                   geometry::CutIntegrationSide::Positive},
                  {1, Real{1}, true}}));
    const auto failed = invokeCollectively(
        MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
    EXPECT_TRUE(failed.allThrew());
    EXPECT_TRUE(anyMessageContains(
        MPI_COMM_WORLD,
        failed,
        "inconsistent_distributed_cell_classification"));
    EXPECT_EQ(system.constraints().numConstraints(), 0u);

    // The failed transaction must leave no partial aggregation line and must
    // not poison a subsequent valid rebuild on the same constraint instance.
    system.setCutIntegrationContext(
        cutContext({{0, Real{0.25}, false}, {1, Real{1}, true}}));
    const auto recovered = invokeCollectively(
        MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
    ASSERT_TRUE(recovered.allSucceeded()) << recovered.local_message;
    expectEveryMasterLineFiniteAndPartitionOfUnity(system);
#endif
}

TEST(SmallCutAggregationConstraintMPI,
     RuntimeEnvironmentMismatchAndMalformedValuesFailThenRecover)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }

    ScopedEnvVar baseline_slave("SVMP_AGGREGATION_SLAVE_ALL_CUT", "0");
    ScopedEnvVar baseline_linear("SVMP_AGGREGATION_LINEAR_EXTENSION", "0");
    ScopedEnvVar baseline_allow("SVMP_AGGREGATION_ALLOW_UNAGGREGATED", "0");
    const auto unlimited_lines =
        std::to_string(std::numeric_limits<std::size_t>::max());
    ScopedEnvVar baseline_max("SVMP_AGGREGATION_MAX_LINES",
                              unlimited_lines.c_str());

    auto mesh = std::make_shared<TwoQuadAggregationMeshAccess>(
        rank, /*full_view=*/true, /*expose_left_wall=*/false);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));

    systems::SetupInputs inputs;
    inputs.topology_override = fullTopology();
    const auto setup_outcome = invokeCollectively(MPI_COMM_WORLD, [&] {
        system.setup(setupOptions(rank, world_size), inputs);
    });
    ASSERT_TRUE(setup_outcome.allSucceeded())
        << setup_outcome.local_message;
    system.setCutIntegrationContext(
        cutContext({{0, Real{0.25}, false}, {1, Real{1}, true}}));

    {
        ScopedEnvVar mismatch("SVMP_AGGREGATION_MAX_LINES",
                              rank == 0 ? "10" : "11");
        const auto failed = invokeCollectively(
            MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
        EXPECT_TRUE(failed.allThrew());
        EXPECT_TRUE(anyMessageContains(
            MPI_COMM_WORLD,
            failed,
            "inconsistent_distributed_runtime_options"));
        EXPECT_EQ(system.constraints().numConstraints(), 0u);
    }
    {
        ScopedEnvVar malformed("SVMP_AGGREGATION_ALLOW_UNAGGREGATED",
                               rank == 0 ? "garbage" : "0");
        const auto failed = invokeCollectively(
            MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
        EXPECT_TRUE(failed.allThrew());
        EXPECT_TRUE(anyMessageContains(
            MPI_COMM_WORLD, failed, "must be exactly 0 or 1"));
        EXPECT_EQ(system.constraints().numConstraints(), 0u);
    }
    {
        ScopedEnvVar malformed("SVMP_AGGREGATION_MAX_LINES",
                               rank == 0 ? "not-a-number"
                                         : unlimited_lines.c_str());
        const auto failed = invokeCollectively(
            MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
        EXPECT_TRUE(failed.allThrew());
        EXPECT_TRUE(anyMessageContains(
            MPI_COMM_WORLD, failed, "unsigned decimal integer"));
        EXPECT_EQ(system.constraints().numConstraints(), 0u);
    }

    const auto recovered = invokeCollectively(
        MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
    ASSERT_TRUE(recovered.allSucceeded()) << recovered.local_message;
    expectEveryMasterLineFiniteAndPartitionOfUnity(system);
#endif
}

TEST(SmallCutAggregationConstraintMPI,
     EmptyLocalWorkRankParticipatesInGlobalAggregation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }

    auto mesh = std::make_shared<TwoQuadAggregationMeshAccess>(
        rank,
        /*full_view=*/rank == 0,
        /*expose_left_wall=*/false,
        /*rank_one_empty=*/true,
        /*all_cells_owned_by_rank_zero=*/true);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));

    systems::SetupInputs inputs;
    inputs.topology_override = allCellsOnRankZeroTopology(rank);
    const auto setup_outcome = invokeCollectively(MPI_COMM_WORLD, [&] {
        system.setup(setupOptions(rank, world_size), inputs);
    });
    ASSERT_TRUE(setup_outcome.allSucceeded())
        << setup_outcome.local_message;

    system.setCutIntegrationContext(
        rank == 0
            ? cutContext({{0, Real{0.25}, false}, {1, Real{1}, true}})
            : cutContext(std::initializer_list<CellRule>{}));
    const auto rebuild_outcome = invokeCollectively(
        MPI_COMM_WORLD, [&] { system.rebuildConstraintState(); });
    ASSERT_TRUE(rebuild_outcome.allSucceeded())
        << rebuild_outcome.local_message;
    // Rank 1 contributed no mesh cell, cut rule, candidate, or root proposal,
    // but its all-ghost DOF view is relevant to the canonical slave/master
    // lines and must receive the same valid constraints.
    expectEveryMasterLineFiniteAndPartitionOfUnity(system);
#endif
}

TEST(SmallCutAggregationConstraintMPI,
     SplitSubcommunicatorsDoNotCrossContaminateAggregation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 4) {
        GTEST_SKIP() << "Run with exactly four MPI ranks";
    }

    MPI_Comm subcomm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD,
                   /*color=*/world_rank / 2,
                   /*key=*/world_rank,
                   &subcomm);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(subcomm, &rank);
    MPI_Comm_size(subcomm, &size);

    bool setup_ok = false;
    bool rebuild_ok = false;
    std::string failure_message;
    {
        auto mesh = std::make_shared<TwoQuadAggregationMeshAccess>(
            rank, /*full_view=*/true, /*expose_left_wall=*/false);
        auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
        systems::FESystem system(mesh);
        const auto pressure = system.addField(systems::FieldSpec{
            .name = "p", .space = space, .components = 1});
        system.addOperator("pressure");
        system.addSystemConstraint(
            std::make_unique<SmallCutAggregationConstraint>(
                pressure,
                geometry::CutIntegrationSide::Negative,
                kInterfaceMarker));

        systems::SetupInputs inputs;
        inputs.topology_override = fullTopology();
        const auto setup_outcome = invokeCollectively(subcomm, [&] {
            system.setup(setupOptions(rank, size, subcomm), inputs);
        });
        setup_ok = setup_outcome.allSucceeded();
        failure_message = setup_outcome.local_message;
        if (setup_ok) {
            system.setCutIntegrationContext(cutContext(
                {{0, Real{0.25}, false}, {1, Real{1}, true}}));
            const auto rebuild_outcome = invokeCollectively(
                subcomm, [&] { system.rebuildConstraintState(); });
            rebuild_ok = rebuild_outcome.allSucceeded();
            failure_message = rebuild_outcome.local_message;
            if (rebuild_ok) {
                expectEveryMasterLineFiniteAndPartitionOfUnity(system);
            }
        }
    }
    MPI_Comm_free(&subcomm);

    EXPECT_TRUE(setup_ok) << failure_message;
    EXPECT_TRUE(rebuild_ok) << failure_message;
#endif
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

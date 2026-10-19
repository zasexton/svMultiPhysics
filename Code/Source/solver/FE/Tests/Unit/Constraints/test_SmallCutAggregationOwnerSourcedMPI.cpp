/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_SmallCutAggregationOwnerSourcedMPI.cpp
 * @brief The owner-sourced aggregation refresh against the
 *        communicator-global one on partitioned quad grids.
 *
 * Every scenario builds the same distributed problem twice, once with each
 * refresh, and requires bitwise identical constraint lines on every rank,
 * identical canonical refresh reports and prolongation digests, and that the
 * owner-sourced refresh actually completed (or, where the halo is too
 * shallow, that both refreshes fail with the same halo diagnostic).
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

namespace {

constexpr int kMarker = 7;

using LevelSet = std::function<Real(Real, Real)>;

/**
 * nx x ny unit quads; cell (i, j) covers [i, i+1] x [j, j+1] and has global
 * ID j*nx + i.  Column i is owned by `column_owner[i]`; a rank stores every
 * cell within `ghost_layers` columns of one of its columns, i.e. the face
 * layers of a column partition.
 */
struct GridLayout {
    int nx{0};
    int ny{0};
    std::vector<int> column_owner{};
    int ghost_layers{0};

    [[nodiscard]] bool storedColumn(int column, int rank) const
    {
        for (int other = 0; other < nx; ++other) {
            if (column_owner[static_cast<std::size_t>(other)] == rank &&
                std::abs(other - column) <= ghost_layers) {
                return true;
            }
        }
        return false;
    }
};

class PartitionedGridMeshAccess final : public assembly::IMeshAccess {
public:
    PartitionedGridMeshAccess(GridLayout layout, int rank)
        : layout_(std::move(layout)), rank_(rank)
    {
        std::map<GlobalIndex, GlobalIndex> node_local;
        for (int j = 0; j < layout_.ny; ++j) {
            for (int i = 0; i < layout_.nx; ++i) {
                if (!layout_.storedColumn(i, rank_)) {
                    continue;
                }
                cell_ij_.push_back({i, j});
                for (const auto node : cornerGids(i, j)) {
                    node_local.emplace(node, 0);
                }
            }
        }
        GlobalIndex next = 0;
        for (auto& [gid, local] : node_local) {
            local = next++;
            node_gids_.push_back(gid);
            const auto x = gid % (layout_.nx + 1);
            const auto y = gid / (layout_.nx + 1);
            node_xyz_.push_back({static_cast<Real>(x),
                                 static_cast<Real>(y),
                                 Real{0}});
        }
        for (const auto& [i, j] : cell_ij_) {
            std::array<GlobalIndex, 4> nodes{};
            const auto corners = cornerGids(i, j);
            for (std::size_t k = 0; k < 4u; ++k) {
                nodes[k] = node_local.at(corners[k]);
            }
            cells_.push_back(nodes);
        }
        for (std::size_t a = 0; a < cell_ij_.size(); ++a) {
            for (std::size_t b = a + 1u; b < cell_ij_.size(); ++b) {
                const auto di = std::abs(cell_ij_[a][0] - cell_ij_[b][0]);
                const auto dj = std::abs(cell_ij_[a][1] - cell_ij_[b][1]);
                if (di + dj == 1) {
                    faces_.push_back({static_cast<GlobalIndex>(a),
                                      static_cast<GlobalIndex>(b)});
                }
            }
        }
    }

    [[nodiscard]] std::array<GlobalIndex, 4> cornerGids(int i, int j) const
    {
        const auto row = static_cast<GlobalIndex>(layout_.nx + 1);
        const auto base = static_cast<GlobalIndex>(j) * row + i;
        return {base, base + 1, base + row + 1, base + row};
    }

    [[nodiscard]] int columnOf(GlobalIndex cell) const
    {
        return cell_ij_.at(static_cast<std::size_t>(cell))[0];
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }
    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        GlobalIndex owned = 0;
        for (GlobalIndex cell = 0; cell < numCells(); ++cell) {
            owned += isOwnedCell(cell) ? 1 : 0;
        }
        return owned;
    }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override
    {
        return static_cast<GlobalIndex>(faces_.size());
    }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }
    [[nodiscard]] GlobalIndex getCellGlobalId(GlobalIndex cell) const override
    {
        const auto& ij = cell_ij_.at(static_cast<std::size_t>(cell));
        return static_cast<GlobalIndex>(ij[1]) * layout_.nx + ij[0];
    }
    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override
    {
        return 1 + *std::max_element(layout_.column_owner.begin(),
                                     layout_.column_owner.end());
    }
    [[nodiscard]] int getCellOwnerRank(GlobalIndex cell) const override
    {
        return layout_.column_owner.at(
            static_cast<std::size_t>(columnOf(cell)));
    }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        return getCellOwnerRank(cell) == rank_;
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
        return node_xyz_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(
        GlobalIndex cell,
        std::vector<std::array<Real, 3>>& coordinates) const override
    {
        const auto& connectivity = cells_.at(static_cast<std::size_t>(cell));
        coordinates.resize(connectivity.size());
        for (std::size_t i = 0; i < connectivity.size(); ++i) {
            coordinates[i] =
                node_xyz_.at(static_cast<std::size_t>(connectivity[i]));
        }
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
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face) const override
    {
        return faces_.at(static_cast<std::size_t>(face));
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
        for (std::size_t face = 0; face < faces_.size(); ++face) {
            callback(static_cast<GlobalIndex>(face),
                     faces_[face].first,
                     faces_[face].second);
        }
    }

    [[nodiscard]] dofs::MeshTopologyInfo topology() const
    {
        dofs::MeshTopologyInfo topology;
        topology.dim = 2;
        topology.n_cells = numCells();
        topology.n_vertices = static_cast<GlobalIndex>(node_gids_.size());
        topology.cell2vertex_offsets.push_back(0);
        for (const auto& cell : cells_) {
            topology.cell2vertex_data.insert(topology.cell2vertex_data.end(),
                                             cell.begin(), cell.end());
            topology.cell2vertex_offsets.push_back(
                static_cast<GlobalIndex>(topology.cell2vertex_data.size()));
        }
        for (std::size_t node = 0; node < node_gids_.size(); ++node) {
            topology.vertex_gids.push_back(node_gids_[node]);
            topology.vertex_coords.push_back(node_xyz_[node][0]);
            topology.vertex_coords.push_back(node_xyz_[node][1]);
        }
        std::set<int> neighbors;
        for (GlobalIndex cell = 0; cell < numCells(); ++cell) {
            topology.cell_gids.push_back(getCellGlobalId(cell));
            topology.cell_owner_ranks.push_back(getCellOwnerRank(cell));
            if (!isOwnedCell(cell)) {
                neighbors.insert(getCellOwnerRank(cell));
            }
        }
        // Ranks that store one of this rank's columns.
        for (int other = 0; other < parallelSize(); ++other) {
            if (other == rank_) {
                continue;
            }
            for (int column = 0; column < layout_.nx; ++column) {
                if (layout_.column_owner[static_cast<std::size_t>(column)] ==
                        rank_ &&
                    layout_.storedColumn(column, other)) {
                    neighbors.insert(other);
                }
            }
        }
        topology.neighbor_ranks.assign(neighbors.begin(), neighbors.end());
        return topology;
    }

private:
    GridLayout layout_{};
    int rank_{0};
    std::vector<std::array<int, 2>> cell_ij_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
    std::vector<GlobalIndex> node_gids_{};
    std::vector<std::array<Real, 3>> node_xyz_{};
    std::vector<std::pair<GlobalIndex, GlobalIndex>> faces_{};
};

void addRule(assembly::CutIntegrationContext& context,
             GlobalIndex cell,
             GlobalIndex gid,
             geometry::CutIntegrationSide side,
             Real fraction,
             bool full)
{
    const auto stable_id = interfaces::cutVolumeStableId(
        kMarker, gid, /*local_region_index=*/0, side, /*source_revision=*/1u);
    assembly::CutCellAssemblyMetadata metadata{};
    metadata.cell = cell;
    metadata.parent_entity = cell;
    metadata.side = side;
    metadata.volume_fraction = fraction;
    metadata.revision_key = stable_id;
    metadata.cut_topology_revision = stable_id;

    geometry::CutQuadratureRule rule{};
    rule.kind = geometry::CutQuadratureKind::Volume;
    rule.side = side;
    rule.measure = fraction;
    rule.parent_measure = Real{1};
    rule.volume_fraction = fraction;
    rule.full_cell_equivalent = full;
    rule.frame = geometry::CutGeometryFrame::Current;
    rule.provenance.parent_entity = cell;
    rule.provenance.parent_entity_global_id = gid;
    rule.provenance.marker = kMarker;
    rule.provenance.cut_topology_revision = stable_id;
    context.addGeneratedVolumeRule(kMarker, metadata, rule);
}

/**
 * Two-sided classification of the stored cells by the corner values of the
 * level set (negative = active side): all corners negative is full active,
 * all nonnegative is inactive full, anything else is cut with the active
 * fraction (negative corners)/4.  `owned_only` keeps only the rules of
 * owned cells (an owner-filtered context).
 */
std::shared_ptr<assembly::CutIntegrationContext> gridCutContext(
    const PartitionedGridMeshAccess& mesh,
    const LevelSet& phi,
    bool owned_only)
{
    auto context = std::make_shared<assembly::CutIntegrationContext>();
    std::vector<std::array<Real, 3>> xyz;
    for (GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
        if (owned_only && !mesh.isOwnedCell(cell)) {
            continue;
        }
        mesh.getCellCoordinates(cell, xyz);
        int negative = 0;
        for (const auto& point : xyz) {
            negative += phi(point[0], point[1]) < Real{0} ? 1 : 0;
        }
        const auto gid = mesh.getCellGlobalId(cell);
        if (negative == 4) {
            addRule(*context, cell, gid,
                    geometry::CutIntegrationSide::Negative, Real{1}, true);
        } else if (negative == 0) {
            addRule(*context, cell, gid,
                    geometry::CutIntegrationSide::Positive, Real{1}, true);
        } else {
            const Real fraction = static_cast<Real>(negative) / Real{4};
            addRule(*context, cell, gid,
                    geometry::CutIntegrationSide::Negative, fraction, false);
            addRule(*context, cell, gid,
                    geometry::CutIntegrationSide::Positive,
                    Real{1} - fraction, false);
        }
    }
    return context;
}

systems::SetupOptions gridSetupOptions(int rank, int world_size)
{
    systems::SetupOptions options;
    options.dof_options.global_numbering =
        dofs::GlobalNumberingMode::OwnerContiguous;
    options.dof_options.ownership = dofs::OwnershipStrategy::CellOwner;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = world_size;
    options.dof_options.mpi_comm = MPI_COMM_WORLD;
    return options;
}

// Strong value on every stored vertex at x = 0 (on every rank that stores
// it), so that candidate rows there are preconstrained.
class LeftColumnDirichlet final : public ISystemConstraint {
public:
    LeftColumnDirichlet(FieldId field,
                        std::shared_ptr<const PartitionedGridMeshAccess> mesh)
        : field_(field), mesh_(std::move(mesh))
    {
    }

    void apply(const systems::FESystem& system,
               AffineConstraints& constraints) override
    {
        const auto* entity_map =
            system.fieldDofHandler(field_).getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::logic_error("left-column Dirichlet needs vertex DOFs");
        }
        const auto offset = system.fieldDofOffset(field_);
        for (GlobalIndex vertex = 0; vertex < entity_map->numVertices();
             ++vertex) {
            if (mesh_->getNodeCoordinates(vertex)[0] != Real{0}) {
                continue;
            }
            for (const auto dof : entity_map->getVertexDofs(vertex)) {
                if (!constraints.isConstrained(offset + dof)) {
                    constraints.addDirichlet(offset + dof, 0.5);
                }
            }
        }
    }

    bool updateValues(const systems::FESystem&,
                      AffineConstraints&,
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
    FieldId field_{INVALID_FIELD_ID};
    std::shared_ptr<const PartitionedGridMeshAccess> mesh_{};
};

struct Scenario {
    GridLayout layout{};
    LevelSet phi{};
    bool owned_rules_only{false};
    bool left_dirichlet{false};
    bool rootless_fallback{false};
};

struct RefreshSnapshot {
    bool threw{false};
    std::string message{};
    // Rank-local constraint lines: [slave, inhomogeneity bits, n,
    // {master, weight bits}...] in slave order.
    std::vector<std::uint64_t> lines{};
    // Communicator-canonical report and prolongation words.
    std::vector<std::uint64_t> canonical{};
    std::uint64_t owner_sourced_completed{0u};
    std::uint64_t owner_sourced_returned{0u};
};

std::uint64_t bits(double value)
{
    return std::bit_cast<std::uint64_t>(value);
}

RefreshSnapshot runScenario(const Scenario& scenario, bool owner_sourced)
{
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    detail::setSmallCutAggregationOwnerSourcedRefresh(owner_sourced);
    const auto counters_before =
        detail::smallCutAggregationOwnerSourcedCounters();

    RefreshSnapshot snapshot;
    auto mesh = std::make_shared<PartitionedGridMeshAccess>(scenario.layout,
                                                            rank);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    if (scenario.left_dirichlet) {
        system.addSystemConstraint(
            std::make_unique<LeftColumnDirichlet>(pressure, mesh));
    }
    SmallCutAggregationGuardOptions guards;
    guards.rootless_fallback = scenario.rootless_fallback;
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kMarker,
        std::vector<int>{},
        std::vector<GlobalIndex>{},
        guards));

    systems::SetupInputs inputs;
    inputs.topology_override = mesh->topology();
    int local_threw = 0;
    try {
        system.setup(gridSetupOptions(rank, world_size), inputs);
        system.setCutIntegrationContext(
            gridCutContext(*mesh, scenario.phi, scenario.owned_rules_only));
        system.rebuildConstraintState();
    } catch (const std::exception& error) {
        local_threw = 1;
        snapshot.message = error.what();
    }
    int any_threw = 0;
    MPI_Allreduce(&local_threw, &any_threw, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    snapshot.threw = any_threw != 0;
    const auto counters_after =
        detail::smallCutAggregationOwnerSourcedCounters();
    snapshot.owner_sourced_completed =
        counters_after.completed - counters_before.completed;
    snapshot.owner_sourced_returned =
        counters_after.returned_to_global - counters_before.returned_to_global;
    detail::setSmallCutAggregationOwnerSourcedRefresh(true);
    if (snapshot.threw) {
        return snapshot;
    }

    std::vector<std::pair<GlobalIndex, std::vector<std::uint64_t>>> lines;
    system.constraints().forEach(
        [&](const AffineConstraints::ConstraintView& line) {
            std::vector<std::uint64_t> words;
            words.push_back(bits(line.inhomogeneity));
            words.push_back(line.entries.size());
            for (const auto& entry : line.entries) {
                words.push_back(static_cast<std::uint64_t>(entry.master_dof));
                words.push_back(bits(entry.weight));
            }
            lines.emplace_back(line.slave_dof, std::move(words));
        });
    std::sort(lines.begin(), lines.end());
    for (const auto& [slave, words] : lines) {
        snapshot.lines.push_back(static_cast<std::uint64_t>(slave));
        snapshot.lines.insert(snapshot.lines.end(), words.begin(), words.end());
    }

    const auto reports = system.completedSmallCutAggregationRefreshReports();
    for (const auto& report : reports) {
        auto& c = snapshot.canonical;
        c.push_back(report.canonical_feature_class_fingerprint);
        c.push_back(report.canonical_slave_set_fingerprint);
        c.push_back(report.maximum_observed_root_path);
        c.push_back(report.root_path_guard_rejections);
        c.push_back(bits(report.maximum_observed_reference_extrapolation));
        c.push_back(report.extrapolation_guard_rejections);
        c.push_back(bits(report.maximum_observed_absolute_coefficient));
        c.push_back(bits(report.maximum_observed_row_l1_norm));
        c.push_back(report.line_guard_rejections);
        c.push_back(report.canonical_candidate_vertices);
        c.push_back(report.canonical_rooted_candidate_vertices);
        c.push_back(report.canonical_rootless_candidate_vertices);
        c.push_back(report.canonical_owned_aggregate_dofs);
        c.push_back(report.canonical_owned_pinned_dofs);
        c.push_back(report.canonical_strong_suppressed_dofs);
        c.push_back(report.canonical_active_feature_count);
        c.push_back(report.canonical_rooted_active_feature_count);
        c.push_back(report.canonical_rootless_active_feature_count);
        c.push_back(bits(report.canonical_rootless_active_physical_volume));
        for (const auto& feature : report.canonical_active_features) {
            c.push_back(static_cast<std::uint64_t>(feature.stable_feature_id));
            c.push_back(feature.canonical_cell_gid_digest);
            c.push_back(feature.canonical_full_active_cell_gid_digest);
            c.push_back(feature.canonical_cut_cell_gid_digest);
            c.push_back(static_cast<std::uint64_t>(feature.disposition));
            c.push_back(feature.canonical_cell_count);
            c.push_back(feature.canonical_full_active_cell_count);
            c.push_back(feature.canonical_cut_cell_count);
            c.push_back(bits(feature.canonical_retained_physical_volume));
        }
    }
    for (const auto& prolongation :
         system.finalizedSmallCutAggregationProlongations()) {
        if (!prolongation) {
            continue;
        }
        auto& c = snapshot.canonical;
        c.push_back(prolongation->canonical_content_digest);
        c.push_back(prolongation->rows.size());
        c.push_back(prolongation->active_cells.size());
        c.push_back(prolongation->patches.size());
        for (const auto& row : prolongation->rows) {
            c.push_back(static_cast<std::uint64_t>(row.slave_dof));
            c.push_back(static_cast<std::uint64_t>(row.root_cell_gid));
            c.push_back(static_cast<std::uint64_t>(
                static_cast<std::int64_t>(row.root_provider_rank)));
            c.push_back(static_cast<std::uint64_t>(row.provisional_kind));
        }
        for (const auto& cell : prolongation->active_cells) {
            c.push_back(static_cast<std::uint64_t>(cell.cell_gid));
            c.push_back(static_cast<std::uint64_t>(
                static_cast<std::int64_t>(cell.owner_rank)));
            c.push_back(static_cast<std::uint64_t>(
                static_cast<std::int64_t>(cell.retained_measure_provider_rank)));
        }
    }
    return snapshot;
}

// True on every rank when `local` holds on every rank.
bool allRanks(bool local)
{
    const int value = local ? 1 : 0;
    int minimum = 0;
    MPI_Allreduce(&value, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return minimum != 0;
}

std::uint64_t wordsDigest(const std::vector<std::uint64_t>& words)
{
    std::uint64_t digest = 1469598103934665603ull;
    for (const auto word : words) {
        digest ^= word;
        digest *= 1099511628211ull;
    }
    return digest;
}

// Both refreshes, compared; returns the owner-sourced snapshot.
RefreshSnapshot expectIdenticalRefreshes(const Scenario& scenario)
{
    const auto global = runScenario(scenario, /*owner_sourced=*/false);
    const auto owner = runScenario(scenario, /*owner_sourced=*/true);
    EXPECT_FALSE(global.threw) << global.message;
    EXPECT_FALSE(owner.threw) << owner.message;
    EXPECT_EQ(global.owner_sourced_completed, 0u);
    EXPECT_EQ(global.owner_sourced_returned, 0u);
    EXPECT_TRUE(allRanks(owner.lines == global.lines));
    EXPECT_TRUE(allRanks(owner.canonical == global.canonical));
    EXPECT_FALSE(owner.lines.empty());
    EXPECT_FALSE(owner.canonical.empty());
    // The canonical ledger is the same on every rank.
    const auto digest = wordsDigest(owner.canonical);
    std::uint64_t minimum = 0u;
    std::uint64_t maximum = 0u;
    MPI_Allreduce(&digest, &minimum, 1, MPI_UINT64_T, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&digest, &maximum, 1, MPI_UINT64_T, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_EQ(minimum, maximum);
    return owner;
}

std::vector<int> contiguousColumns(int nx, int world_size)
{
    std::vector<int> owner(static_cast<std::size_t>(nx));
    for (int column = 0; column < nx; ++column) {
        owner[static_cast<std::size_t>(column)] =
            std::min(world_size - 1, column * world_size / nx);
    }
    return owner;
}

// Circle of radius 3.3 centered at (6.1, 4.7): a rooted drop.
Real dropLevelSet(Real x, Real y)
{
    return std::hypot(x - Real{6.1}, y - Real{4.7}) - Real{3.3};
}

// A film on (3.2, oo) x [0, 0.6) joined to a block of full cells at
// x > 4.5, y < 2.4: the film vertex at x = 3 roots in cell (5, 0), two cells
// away (the largest distance the extrapolation guard admits), and every
// other film vertex nearer.
Real filmLevelSet(Real x, Real y)
{
    return std::min(std::max(Real{3.2} - x, y - Real{0.6}),
                    std::max(Real{4.5} - x, y - Real{2.4}));
}

} // namespace

TEST(SmallCutAggregationOwnerSourcedMPI, RootedDropMatchesGlobalRefresh)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2 || world_size > 4) {
        GTEST_SKIP() << "Run with two to four MPI ranks";
    }
    Scenario scenario;
    scenario.layout.nx = 12;
    scenario.layout.ny = 10;
    scenario.layout.column_owner = contiguousColumns(12, world_size);
    scenario.layout.ghost_layers = 4;
    scenario.phi = dropLevelSet;
    const auto owner = expectIdenticalRefreshes(scenario);
    EXPECT_GE(owner.owner_sourced_completed, 1u);
    EXPECT_EQ(owner.owner_sourced_returned, 0u);
#endif
}

TEST(SmallCutAggregationOwnerSourcedMPI,
     OwnerFilteredRulesAndPreconstrainedRowsMatchGlobalRefresh)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2 || world_size > 4) {
        GTEST_SKIP() << "Run with two to four MPI ranks";
    }
    Scenario scenario;
    scenario.layout.nx = 12;
    scenario.layout.ny = 10;
    scenario.layout.column_owner = contiguousColumns(12, world_size);
    scenario.layout.ghost_layers = 4;
    // The drop touches x = 0, whose vertices carry a strong value.
    scenario.phi = [](Real x, Real y) {
        return std::hypot(x - Real{2.2}, y - Real{4.9}) - Real{3.1};
    };
    scenario.owned_rules_only = true;
    scenario.left_dirichlet = true;
    const auto owner = expectIdenticalRefreshes(scenario);
    EXPECT_GE(owner.owner_sourced_completed, 1u);
    EXPECT_EQ(owner.owner_sourced_returned, 0u);
#endif
}

TEST(SmallCutAggregationOwnerSourcedMPI,
     RootlessIslandAndRootedDropMatchGlobalRefresh)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2 || world_size > 4) {
        GTEST_SKIP() << "Run with two to four MPI ranks";
    }
    Scenario scenario;
    scenario.layout.nx = 14;
    scenario.layout.ny = 9;
    scenario.layout.column_owner = contiguousColumns(14, world_size);
    scenario.layout.ghost_layers = 4;
    // A rooted drop and a sub-cell droplet (cut cells only) far from it.
    scenario.phi = [](Real x, Real y) {
        return std::min(
            std::hypot(x - Real{9.2}, y - Real{4.4}) - Real{3.0},
            std::hypot(x - Real{2.0}, y - Real{2.0}) - Real{0.3});
    };
    const auto owner = expectIdenticalRefreshes(scenario);
    EXPECT_GE(owner.owner_sourced_completed, 1u);
    EXPECT_EQ(owner.owner_sourced_returned, 0u);
#endif
}

// With four ranks owning two columns each, rank 0 assembles the film vertex
// at x = 3 (a face neighbour of its column 1), whose only admissible root is
// cell (5, 0), owned by rank 2.  With four ghost layers column 5 is rank 0's
// outermost stored column: the root lies on the halo boundary of a rank that
// shares no face with the root's owner.
TEST(SmallCutAggregationOwnerSourcedMPI,
     RootOnNonNeighbourHaloBoundaryMatchesGlobalRefresh)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2 || world_size > 4) {
        GTEST_SKIP() << "Run with two to four MPI ranks";
    }
    Scenario scenario;
    scenario.layout.nx = 8;
    scenario.layout.ny = 4;
    scenario.layout.column_owner = contiguousColumns(8, world_size);
    scenario.layout.ghost_layers = 4;
    scenario.phi = filmLevelSet;
    const auto owner = expectIdenticalRefreshes(scenario);
    EXPECT_GE(owner.owner_sourced_completed, 1u);
    EXPECT_EQ(owner.owner_sourced_returned, 0u);
#endif
}

// The same film with three ghost layers: a rank that assembles a film
// vertex does not store its roots.  Both refreshes fail closed with the
// same halo diagnostic (the owner-sourced refresh returns to the
// communicator-global one, which reports it).
TEST(SmallCutAggregationOwnerSourcedMPI,
     ShallowHaloFailsClosedLikeGlobalRefresh)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 4) {
        GTEST_SKIP() << "Run with exactly four MPI ranks";
    }
    Scenario scenario;
    scenario.layout.nx = 8;
    scenario.layout.ny = 4;
    scenario.layout.column_owner = contiguousColumns(8, world_size);
    scenario.layout.ghost_layers = 3;
    scenario.phi = filmLevelSet;
    const auto global = runScenario(scenario, /*owner_sourced=*/false);
    const auto owner = runScenario(scenario, /*owner_sourced=*/true);
    EXPECT_TRUE(global.threw);
    EXPECT_TRUE(owner.threw);
    EXPECT_EQ(owner.owner_sourced_completed, 0u);
    EXPECT_GE(owner.owner_sourced_returned, 1u);
    const bool local_same = owner.message == global.message;
    EXPECT_TRUE(allRanks(local_same));
    const bool halo_message =
        owner.message.find("incomplete_distributed_aggregation_halo") !=
            std::string::npos ||
        owner.message.find("another communicator rank") != std::string::npos;
    EXPECT_TRUE(halo_message) << owner.message;
    int local_halo = owner.message.find(
                         "incomplete_distributed_aggregation_halo") !=
                             std::string::npos
                         ? 1
                         : 0;
    int any_halo = 0;
    MPI_Allreduce(&local_halo, &any_halo, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_EQ(any_halo, 1);
#endif
}

// Non-contiguous ownership (columns dealt round-robin) gives every rank
// several separate halos and roots owned by ranks that are not face
// neighbours of the candidate's owner.
TEST(SmallCutAggregationOwnerSourcedMPI,
     RoundRobinColumnsMatchGlobalRefresh)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2 || world_size > 4) {
        GTEST_SKIP() << "Run with two to four MPI ranks";
    }
    Scenario scenario;
    scenario.layout.nx = 12;
    scenario.layout.ny = 10;
    scenario.layout.column_owner.resize(12);
    for (int column = 0; column < 12; ++column) {
        scenario.layout.column_owner[static_cast<std::size_t>(column)] =
            (column / 2) % world_size;
    }
    scenario.layout.ghost_layers = 6;
    scenario.phi = dropLevelSet;
    const auto owner = expectIdenticalRefreshes(scenario);
    EXPECT_GE(owner.owner_sourced_completed, 1u);
    EXPECT_EQ(owner.owner_sourced_returned, 0u);
#endif
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

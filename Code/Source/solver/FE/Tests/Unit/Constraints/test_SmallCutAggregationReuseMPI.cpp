/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_SmallCutAggregationReuseMPI.cpp
 * @brief MPI checks that the small-cut aggregation reuse decision is
 *        collective and that a reused refresh publishes the same content as
 *        a full recomputation.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Dofs/DofHandler.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

namespace {

constexpr int kReuseInterfaceMarker = 7;

// Two unit quads, both visible on both ranks; rank r owns cell r. Unlike the
// fixtures of test_SmallCutAggregationConstraintMPI.cpp it reports revision
// tracking, which the aggregation reuse requires.
class TwoQuadRevisionTrackedMeshAccess final : public assembly::IMeshAccess {
public:
    explicit TwoQuadRevisionTrackedMeshAccess(int rank) : rank_(rank) {}

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override
    {
        return true;
    }
    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }
    [[nodiscard]] GlobalIndex getCellGlobalId(GlobalIndex cell) const override
    {
        return cell;
    }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        return static_cast<int>(cell) == rank_;
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
                                               GlobalIndex) const override
    {
        return INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override
    {
        return -1;
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex) const override
    {
        return {0, 1};
    }
    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }
    void forEachOwnedCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        callback(static_cast<GlobalIndex>(rank_));
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
        callback(0, 0, 1);
    }

private:
    int rank_{0};
    std::vector<std::array<Real, 3>> nodes_{
        {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0},
        {2.0, 0.0, 0.0}, {2.0, 1.0, 0.0},
    };
    std::vector<std::array<GlobalIndex, 4>> cells_{
        std::array<GlobalIndex, 4>{0, 2, 3, 1},
        std::array<GlobalIndex, 4>{2, 4, 5, 3},
    };
};

dofs::MeshTopologyInfo twoQuadTopology()
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

systems::SetupOptions twoRankSetupOptions(int rank, int world_size)
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

// Cell 0 is cut with the given retained fraction, cell 1 is full.
std::shared_ptr<assembly::CutIntegrationContext> cutLeftContext(
    Real cut_fraction)
{
    auto context = std::make_shared<assembly::CutIntegrationContext>();
    for (GlobalIndex cell = 0; cell < 2; ++cell) {
        const bool full = cell == 1;
        const Real fraction = full ? Real{1} : cut_fraction;
        const auto stable_id = interfaces::cutVolumeStableId(
            kReuseInterfaceMarker,
            cell,
            /*local_region_index=*/0,
            geometry::CutIntegrationSide::Negative,
            /*source_revision=*/1u);
        assembly::CutCellAssemblyMetadata metadata{};
        metadata.cell = cell;
        metadata.parent_entity = cell;
        metadata.side = geometry::CutIntegrationSide::Negative;
        metadata.volume_fraction = fraction;
        metadata.revision_key = stable_id;
        metadata.cut_topology_revision = stable_id;

        geometry::CutQuadratureRule rule{};
        rule.kind = geometry::CutQuadratureKind::Volume;
        rule.side = geometry::CutIntegrationSide::Negative;
        rule.measure = fraction;
        rule.parent_measure = Real{1};
        rule.volume_fraction = fraction;
        rule.full_cell_equivalent = full;
        rule.frame = geometry::CutGeometryFrame::Current;
        rule.provenance.parent_entity = cell;
        rule.provenance.parent_entity_global_id = cell;
        rule.provenance.marker = kReuseInterfaceMarker;
        rule.provenance.cut_topology_revision = stable_id;
        context->addGeneratedVolumeRule(
            kReuseInterfaceMarker, metadata, rule);
    }
    return context;
}

class ScopedReuseEnvVar {
public:
    ScopedReuseEnvVar(const char* key, const char* value) : key_(key)
    {
        if (const char* prior = std::getenv(key_)) {
            prior_ = std::string(prior);
        }
        ::setenv(key_, value, 1);
    }
    ~ScopedReuseEnvVar()
    {
        if (prior_.has_value()) {
            ::setenv(key_, prior_->c_str(), 1);
        } else {
            ::unsetenv(key_);
        }
    }
    ScopedReuseEnvVar(const ScopedReuseEnvVar&) = delete;
    ScopedReuseEnvVar& operator=(const ScopedReuseEnvVar&) = delete;

private:
    const char* key_;
    std::optional<std::string> prior_;
};

struct ReuseRebuildOutcome {
    bool all_succeeded{false};
    std::string log{};
    std::uint64_t digest{0u};
    bool digest_consistent{false};
    std::uint64_t cut_cell_volume_bits{0u};
};

// Rebuilds collectively and returns the local log, the communicator-checked
// canonical prolongation digest and the retained volume of cell 0.
ReuseRebuildOutcome rebuildCollectively(systems::FESystem& system)
{
    ReuseRebuildOutcome outcome;
    int local_threw = 0;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    try {
        system.rebuildConstraintState();
    } catch (...) {
        local_threw = 1;
    }
    outcome.log = testing::internal::GetCapturedStdout();
    outcome.log += testing::internal::GetCapturedStderr();
    int any_threw = 0;
    MPI_Allreduce(&local_threw, &any_threw, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    outcome.all_succeeded = any_threw == 0;
    if (!outcome.all_succeeded) {
        return outcome;
    }
    const auto prolongations =
        system.finalizedSmallCutAggregationProlongations();
    if (prolongations.size() == 1u && prolongations.front()) {
        outcome.digest = prolongations.front()->canonical_content_digest;
        for (const auto& cell : prolongations.front()->active_cells) {
            if (cell.cell_gid == 0) {
                outcome.cut_cell_volume_bits = std::bit_cast<std::uint64_t>(
                    static_cast<double>(cell.retained_physical_volume));
            }
        }
    }
    std::array<std::uint64_t, 2> local{{outcome.digest, ~outcome.digest}};
    std::array<std::uint64_t, 2> minimum{};
    MPI_Allreduce(local.data(), minimum.data(), 2, MPI_UINT64_T, MPI_MIN,
                  MPI_COMM_WORLD);
    outcome.digest_consistent =
        outcome.digest != 0u && minimum[0] == ~minimum[1];
    return outcome;
}

} // namespace

TEST(SmallCutAggregationReuseMPI, ReuseDecisionIsCollective)
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

    auto mesh = std::make_shared<TwoQuadRevisionTrackedMeshAccess>(rank);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kReuseInterfaceMarker));
    systems::SetupInputs inputs;
    inputs.topology_override = twoQuadTopology();
    system.setup(twoRankSetupOptions(rank, world_size), inputs);

    const auto refresh = [&](Real cut_fraction) {
        system.setCutIntegrationContext(cutLeftContext(cut_fraction));
        return rebuildCollectively(system);
    };

    const auto first = refresh(Real{0.25});
    ASSERT_TRUE(first.all_succeeded);
    ASSERT_TRUE(first.digest_consistent);
    EXPECT_NE(first.log.find("decision=rebuilt reason=no_previous_refresh"),
              std::string::npos)
        << first.log;

    const auto unchanged = refresh(Real{0.25});
    ASSERT_TRUE(unchanged.all_succeeded);
    ASSERT_TRUE(unchanged.digest_consistent);
    EXPECT_NE(unchanged.log.find("decision=reused"), std::string::npos)
        << unchanged.log;
    EXPECT_EQ(unchanged.digest, first.digest);

    // Only rank 1 cannot reuse. Rank 0 still has a matching record, but must
    // follow the collective decision instead of returning early, otherwise
    // the ranks would enter different collectives and deadlock.
    {
        std::optional<ScopedReuseEnvVar> disable;
        if (rank == 1) {
            disable.emplace("SVMP_DISABLE_SMALL_CUT_AGGREGATION_REUSE", "1");
        }
        const auto split = refresh(Real{0.25});
        ASSERT_TRUE(split.all_succeeded);
        ASSERT_TRUE(split.digest_consistent);
        EXPECT_EQ(split.digest, first.digest);
        EXPECT_NE(split.log.find(
                      rank == 0
                          ? "decision=rebuilt reason=another_rank_changed"
                          : "decision=rebuilt reason=not_admissible"),
                  std::string::npos)
            << split.log;
        EXPECT_EQ(split.log.find("decision=reused"), std::string::npos)
            << split.log;
    }

    // Rank 1 recorded nothing while reuse was disabled there.
    const auto recovering = refresh(Real{0.25});
    ASSERT_TRUE(recovering.all_succeeded);
    EXPECT_NE(recovering.log.find(
                  rank == 0 ? "decision=rebuilt reason=another_rank_changed"
                            : "decision=rebuilt reason=no_previous_refresh"),
              std::string::npos)
        << recovering.log;
    const auto reused_again = refresh(Real{0.25});
    ASSERT_TRUE(reused_again.all_succeeded);
    EXPECT_NE(reused_again.log.find("decision=reused"), std::string::npos)
        << reused_again.log;
    EXPECT_EQ(reused_again.digest, first.digest);
#endif
}

TEST(SmallCutAggregationReuseMPI,
     ReusedVolumeRefreshMatchesFullRecomputation)
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

    auto mesh = std::make_shared<TwoQuadRevisionTrackedMeshAccess>(rank);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kReuseInterfaceMarker));
    systems::SetupInputs inputs;
    inputs.topology_override = twoQuadTopology();
    system.setup(twoRankSetupOptions(rank, world_size), inputs);

    system.setCutIntegrationContext(cutLeftContext(Real{0.25}));
    const auto first = rebuildCollectively(system);
    ASSERT_TRUE(first.all_succeeded);

    // Same cut topology, new retained volume: reused, with the volumes
    // gathered from their canonical providers.
    system.setCutIntegrationContext(cutLeftContext(Real{0.375}));
    const auto moved = rebuildCollectively(system);
    ASSERT_TRUE(moved.all_succeeded);
    ASSERT_TRUE(moved.digest_consistent);
    EXPECT_NE(moved.log.find("decision=reused"), std::string::npos)
        << moved.log;
    EXPECT_NE(moved.digest, first.digest);
    EXPECT_EQ(moved.cut_cell_volume_bits,
              std::bit_cast<std::uint64_t>(0.375));

    ReuseRebuildOutcome recomputed;
    {
        ScopedReuseEnvVar disable("SVMP_DISABLE_SMALL_CUT_AGGREGATION_REUSE",
                                  "1");
        system.setCutIntegrationContext(cutLeftContext(Real{0.375}));
        recomputed = rebuildCollectively(system);
    }
    ASSERT_TRUE(recomputed.all_succeeded);
    ASSERT_TRUE(recomputed.digest_consistent);
    EXPECT_NE(recomputed.log.find("decision=rebuilt reason=not_admissible"),
              std::string::npos)
        << recomputed.log;
    EXPECT_EQ(recomputed.digest, moved.digest);
    EXPECT_EQ(recomputed.cut_cell_volume_bits, moved.cut_cell_volume_bits);
#endif
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ContactPointVelocityConstraintMPI.cpp
 * @brief Two-rank checks of the contact-point velocity constraint: a contact
 *        edge whose wet and dry vertices are owned by different ranks, and
 *        the fail-closed path when a master is not locally relevant.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Constraints/ContactPointVelocityConstraint.h"
#include "Core/FEException.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {
namespace {

constexpr int kWallMarker = 11;

// Strip [0,2]x[0,1] in four triangles; global vertices
//   0 (0,0)  1 (1,0)  2 (2,0)  3 (0,1)  4 (1,1)  5 (2,1),
// cells T0 (0,1,4), T1 (0,4,3) on rank 0 and T2 (1,2,5), T3 (1,5,4) on rank 1.
// The bottom faces (0,1) of T0 and (1,2) of T2 carry the wall marker.
// With left_only the rank sees T0 and T1 only (local vertices 0,1,3,4).
class StripMeshAccess final : public assembly::IMeshAccess {
public:
    StripMeshAccess(int rank, bool left_only) : rank_(rank), left_only_(left_only)
    {
        const std::array<std::array<Real, 3>, 6> all{{
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {2.0, 1.0, 0.0},
        }};
        if (left_only_) {
            gids_ = {0, 1, 3, 4};
            cells_ = {{0, 1, 3}, {0, 3, 2}};
            cell_gids_ = {0, 1};
        } else {
            gids_ = {0, 1, 2, 3, 4, 5};
            cells_ = {{0, 1, 4}, {0, 4, 3}, {1, 2, 5}, {1, 5, 4}};
            cell_gids_ = {0, 1, 2, 3};
        }
        for (const auto g : gids_) {
            nodes_.push_back(all[static_cast<std::size_t>(g)]);
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }
    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        GlobalIndex n = 0;
        forEachOwnedCell([&](GlobalIndex) { ++n; });
        return n;
    }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return left_only_ ? 1 : 2; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override { return true; }
    [[nodiscard]] GlobalIndex getCellGlobalId(GlobalIndex cell) const override
    {
        return cell_gids_.at(static_cast<std::size_t>(cell));
    }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        return ownerOf(cell) == rank_;
    }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Triangle3;
    }
    void getCellNodes(GlobalIndex cell, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(c.begin(), c.end());
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node) const override
    {
        return nodes_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(GlobalIndex cell,
                            std::vector<std::array<Real, 3>>& coordinates) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell));
        coordinates.clear();
        for (const auto n : c) {
            coordinates.push_back(nodes_.at(static_cast<std::size_t>(n)));
        }
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override
    {
        return LocalIndex{0};  // the bottom edge is local edge (0,1)
    }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return kWallMarker; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override
    {
        return {-1, -1};
    }
    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            if (ownerOf(c) == rank_) {
                callback(c);
            }
        }
    }
    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker >= 0 && marker != kWallMarker) {
            return;
        }
        callback(0, 0);  // T0
        if (!left_only_) {
            callback(1, 2);  // T2
        }
    }
    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override
    {
    }

    [[nodiscard]] const std::vector<GlobalIndex>& gids() const noexcept { return gids_; }

private:
    [[nodiscard]] int ownerOf(GlobalIndex cell) const
    {
        return cell_gids_.at(static_cast<std::size_t>(cell)) < 2 ? 0 : 1;
    }

    int rank_{0};
    bool left_only_{false};
    std::vector<GlobalIndex> gids_{};
    std::vector<std::array<GlobalIndex, 3>> cells_{};
    std::vector<GlobalIndex> cell_gids_{};
    std::vector<std::array<Real, 3>> nodes_{};
};

dofs::MeshTopologyInfo topologyFor(const StripMeshAccess& mesh, bool left_only, int rank)
{
    dofs::MeshTopologyInfo t;
    t.dim = 2;
    t.n_cells = static_cast<GlobalIndex>(mesh.numCells());
    t.n_vertices = static_cast<GlobalIndex>(mesh.gids().size());
    t.cell2vertex_offsets.push_back(0);
    std::vector<GlobalIndex> nodes;
    for (GlobalIndex c = 0; c < mesh.numCells(); ++c) {
        mesh.getCellNodes(c, nodes);
        for (const auto n : nodes) {
            t.cell2vertex_data.push_back(static_cast<MeshIndex>(n));
        }
        t.cell2vertex_offsets.push_back(static_cast<MeshOffset>(t.cell2vertex_data.size()));
        t.cell_gids.push_back(mesh.getCellGlobalId(c));
        t.cell_owner_ranks.push_back(mesh.getCellGlobalId(c) < 2 ? 0 : 1);
    }
    for (std::size_t v = 0; v < mesh.gids().size(); ++v) {
        t.vertex_gids.push_back(mesh.gids()[v]);
        const auto x = mesh.getNodeCoordinates(static_cast<GlobalIndex>(v));
        t.vertex_coords.insert(t.vertex_coords.end(), {x[0], x[1]});
    }
    t.neighbor_ranks = left_only ? std::vector<int>{1} : std::vector<int>{rank == 0 ? 1 : 0};
    return t;
}

systems::SetupOptions setupOptions(int rank, int world_size)
{
    systems::SetupOptions options;
    options.dof_options.global_numbering = dofs::GlobalNumberingMode::OwnerContiguous;
    options.dof_options.ownership = dofs::OwnershipStrategy::CellOwner;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = world_size;
    options.dof_options.mpi_comm = MPI_COMM_WORLD;
    return options;
}

// phi = (x - 1.3) - 0.2 y: the contact point is at x = 1.3 on edge (1,2),
// s = 0.3 from the wet vertex 1; cos(theta) = -n.n_wall = n_y.
Real phiAt(const std::array<Real, 3>& x) { return (x[0] - 1.3) - 0.2 * x[1]; }
Real law(Real cos_theta) { return 0.5 - cos_theta; }

struct Outcome {
    int threw{0};
    int any_threw{0};
    std::string message{};
};

template <typename F>
Outcome collectively(F&& f)
{
    Outcome o;
    try {
        f();
    } catch (const std::exception& error) {
        o.threw = 1;
        o.message = error.what();
    }
    MPI_Allreduce(&o.threw, &o.any_threw, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return o;
}

struct Fixture {
    std::shared_ptr<StripMeshAccess> mesh;
    std::unique_ptr<systems::FESystem> system;
    FieldId velocity{INVALID_FIELD_ID};
};

Fixture makeFixture(int rank, int world_size, bool left_only)
{
    Fixture f;
    f.mesh = std::make_shared<StripMeshAccess>(rank, left_only);
    f.system = std::make_unique<systems::FESystem>(f.mesh);
    auto space = spaces::VectorSpace(spaces::SpaceType::H1, ElementType::Triangle3, 1, 2);
    f.velocity = f.system->addField(systems::FieldSpec{
        .name = "u", .space = space, .components = space->value_dimension()});
    f.system->addOperator("momentum");
    ContactPointVelocityLaw contact;
    contact.wall_boundary_markers = {kWallMarker};
    contact.wall_normal = {0.0, -1.0, 0.0};
    contact.level_set_field_name = "phi";
    auto mesh = f.mesh;
    contact.level_set_vertex_value = [mesh](GlobalIndex v) {
        return phiAt(mesh->getNodeCoordinates(v));
    };
    contact.contact_speed = law;
    f.system->addSystemConstraint(
        std::make_unique<ContactPointVelocityConstraint>(f.velocity, std::move(contact)));
    systems::SetupInputs inputs;
    inputs.topology_override = topologyFor(*f.mesh, left_only, rank);
    f.system->setup(setupOptions(rank, world_size), inputs);
    return f;
}

GlobalIndex tangentialDof(const Fixture& f, GlobalIndex global_vertex)
{
    const auto& gids = f.mesh->gids();
    for (std::size_t v = 0; v < gids.size(); ++v) {
        if (gids[v] == global_vertex) {
            const auto vdofs = f.system->fieldDofHandler(f.velocity)
                                   .getEntityDofMap()
                                   ->getVertexDofs(static_cast<GlobalIndex>(v));
            return f.system->fieldDofOffset(f.velocity) + vdofs[0];
        }
    }
    return INVALID_GLOBAL_INDEX;
}

} // namespace

TEST(ContactPointVelocityConstraintMPI, CrossRankContactEdgeInstallsTheSerialLine)
{
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }
    Fixture f;
    const auto setup = collectively([&] { f = makeFixture(rank, world_size, false); });
    ASSERT_EQ(setup.any_threw, 0) << setup.message;
    const auto rebuild = collectively([&] { f.system->rebuildConstraintState(); });
    ASSERT_EQ(rebuild.any_threw, 0) << rebuild.message;

    const auto wet = tangentialDof(f, 1);
    const auto dry = tangentialDof(f, 2);
    const auto& partition = f.system->dofHandler().getPartition();
    // The wet and the dry vertex are owned by different ranks.
    int wet_owned = partition.isOwned(wet) ? rank : -1;
    int dry_owned = partition.isOwned(dry) ? rank : -1;
    int wet_owner = -1;
    int dry_owner = -1;
    MPI_Allreduce(&wet_owned, &wet_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&dry_owned, &dry_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_NE(wet_owner, dry_owner);

    // Serial expectation: 0.7 u_w + 0.3 u_d = law(n_y), slave u_w.
    const Real cos_theta = -0.2 / std::sqrt(1.04);
    const Real target = law(cos_theta);
    const auto line = f.system->constraints().getConstraint(wet);
    ASSERT_TRUE(line.has_value()) << "rank " << rank;
    ASSERT_EQ(line->entries.size(), 1u);
    EXPECT_EQ(line->entries[0].master_dof, dry);
    EXPECT_NEAR(line->entries[0].weight, -0.3 / 0.7, 1.0e-14);
    EXPECT_NEAR(line->inhomogeneity, target / 0.7, 1.0e-14);
    EXPECT_FALSE(f.system->constraints().isConstrained(dry));
}

TEST(ContactPointVelocityConstraintMPI, MissingMasterFailsClosedOnEveryRank)
{
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }
    // Rank 0 sees only the left cells: the wet slave is relevant there, the
    // dry master is not.
    Fixture f;
    const auto setup = collectively([&] { f = makeFixture(rank, world_size, rank == 0); });
    ASSERT_EQ(setup.any_threw, 0) << setup.message;
    const auto rebuild = collectively([&] { f.system->rebuildConstraintState(); });
    EXPECT_EQ(rebuild.threw, 1) << "rank " << rank;
    EXPECT_NE(rebuild.message.find("contact_point_line_not_installable"), std::string::npos)
        << rebuild.message;
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

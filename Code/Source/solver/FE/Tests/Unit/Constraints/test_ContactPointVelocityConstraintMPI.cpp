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

#include "ContactPointVelocityTestMesh.h"
#include "Constraints/ContactPointVelocityConstraint.h"
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
    ContactPointVelocityConstraint* constraint{};
};

Fixture makeFixture(int rank, int world_size, bool owned_only,
                    std::function<Real(const std::array<Real, 3>&)> phi = phiAt)
{
    Fixture f;
    f.mesh = std::make_shared<StripMeshAccess>(rank, world_size, owned_only);
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
    contact.level_set_vertex_value = [mesh, phi](GlobalIndex v) {
        return phi(mesh->getNodeCoordinates(v));
    };
    contact.contact_speed = law;
    auto owned = std::make_unique<ContactPointVelocityConstraint>(f.velocity, std::move(contact));
    f.constraint = owned.get();
    f.system->addSystemConstraint(std::move(owned));
    systems::SetupInputs inputs;
    inputs.topology_override = topologyFor(*f.mesh);
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

TEST(ContactPointVelocityConstraintMPI, ResolvesOneSharedVertexUsingBothOwnedWallEdges)
{
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly two MPI ranks";
    }
    // Neither rank has the other wall edge. Global vertex 1 is local vertex
    // 1 on rank 0 and local vertex 0 on rank 1. The two cell angles differ.
    std::array<Real, 3> wall_phi{};
    std::array<Real, 3> vertical_slope{-0.2, -0.2, 0.8};
    Fixture f;
    const auto setup = collectively([&] {
        f = makeFixture(rank, world_size, true, [&](const auto& x) {
            const auto i = static_cast<std::size_t>(x[0]);
            return wall_phi[i] + vertical_slope[i] * x[1];
        });
    });
    ASSERT_EQ(setup.any_threw, 0) << setup.message;
    const auto root = tangentialDof(f, 1);
    for (const Real direction : {1.0, -1.0}) {
        wall_phi = {-direction, 0, direction};
        const auto rebuild = collectively([&] { f.system->rebuildConstraintState(); });
        ASSERT_EQ(rebuild.any_threw, 0) << rebuild.message;
        const auto& samples = f.constraint->samples();
        // For +x the contour enters T3 on rank 1, from (1,0) to
        // (1.1,1); both wall triangles only touch the vertex. For -x it
        // enters T0 on rank 0 with gradient (-1,-0.2).
        const Real cos_theta = -0.2 / std::sqrt(direction > 0.0 ? 4.04 : 1.04);
        EXPECT_EQ(samples.size(), 1u) << "rank " << rank;
        if (samples.size() == 1u) {
            EXPECT_EQ(samples[0].slave_dof, root);
            EXPECT_EQ(samples[0].point[0], 1.0);
            EXPECT_EQ(samples[0].fraction, direction > 0.0 ? 1.0 : 0.0);
            EXPECT_NEAR(samples[0].cos_theta, cos_theta, 1.0e-14);
        }
        const auto line = f.system->constraints().getConstraint(root);
        EXPECT_TRUE(line.has_value()) << "rank " << rank;
        if (line) {
            EXPECT_TRUE(line->isDirichlet());
            EXPECT_NEAR(line->inhomogeneity, direction * law(cos_theta), 1.0e-14);
        }
    }
    for (const std::array<Real, 3>& values : std::vector<std::array<Real, 3>>{
             {-1, 0, -1}, {1, 0, 1}, {-1, 0, 0}, {0, 0, 1}}) {
        wall_phi = values;
        const auto rebuild = collectively([&] { f.system->rebuildConstraintState(); });
        ASSERT_EQ(rebuild.any_threw, 0) << rebuild.message;
        EXPECT_TRUE(f.constraint->samples().empty()) << "rank " << rank;
        EXPECT_FALSE(f.system->constraints().isConstrained(root)) << "rank " << rank;
    }
    // The contour follows the radial edge (1,0)--(1,1), shared by T0
    // and T3 on different ranks. Its two traces are one physical segment.
    wall_phi = {-1, 0, 1};
    vertical_slope[1] = 0;
    auto rebuilt = collectively([&] { f.system->rebuildConstraintState(); });
    ASSERT_EQ(rebuilt.any_threw, 0) << rebuilt.message;
    const auto& samples = f.constraint->samples();
    EXPECT_EQ(samples.size(), 1u);
    if (samples.size() == 1u) {
        EXPECT_NEAR(samples[0].cos_theta, 0., 1.0e-14);
    }

    // Alternating signs around the star give three distinct segments.
    // The strong law needs a unique through-liquid angle on every rank.
    vertical_slope[1] = 1;
    vertical_slope[2] = -2;
    rebuilt = collectively([&] { f.system->rebuildConstraintState(); });
    EXPECT_EQ(rebuilt.threw, 1);
    EXPECT_NE(rebuilt.message.find("unique interface segment"), std::string::npos);

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

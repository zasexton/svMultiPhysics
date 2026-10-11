/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ContactPointVelocityConstraint.cpp
 * @brief Serial checks of the contact-point velocity constraint: line
 *        structure, expansion of a pre-existing master-bearing line of the
 *        dry vertex, and no clash with strong wall-normal Dirichlet DOFs.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Constraints/ContactPointVelocityConstraint.h"
#include "Core/FEException.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"


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
        (void)cell;
        return 0;
    }

    int rank_{0};
    bool left_only_{false};
    std::vector<GlobalIndex> gids_{};
    std::vector<std::array<GlobalIndex, 3>> cells_{};
    std::vector<GlobalIndex> cell_gids_{};
    std::vector<std::array<Real, 3>> nodes_{};
};

dofs::MeshTopologyInfo serialTopology(const StripMeshAccess& mesh)
{
    dofs::MeshTopologyInfo t;
    t.dim = 2;
    t.n_cells = mesh.numCells();
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
        t.cell_owner_ranks.push_back(0);
    }
    for (std::size_t v = 0; v < mesh.gids().size(); ++v) {
        t.vertex_gids.push_back(mesh.gids()[v]);
        const auto x = mesh.getNodeCoordinates(static_cast<GlobalIndex>(v));
        t.vertex_coords.insert(t.vertex_coords.end(), {x[0], x[1]});
    }
    return t;
}

// phi = (x - 1.3) - 0.2 y: the contact point is at x = 1.3 on edge (1,2),
// s = 0.3 from the wet vertex 1; cos(theta) = -n.n_wall = n_y.
Real phiAt(const std::array<Real, 3>& x) { return (x[0] - 1.3) - 0.2 * x[1]; }
Real law(Real cos_theta) { return 0.5 - cos_theta; }

GlobalIndex componentDof(const systems::FESystem& system, FieldId field, GlobalIndex vertex, int c)
{
    const auto vdofs = system.fieldDofHandler(field).getEntityDofMap()->getVertexDofs(vertex);
    return system.fieldDofOffset(field) + vdofs[static_cast<std::size_t>(c)];
}

// Pre-existing constraints of the field, applied before the contact line:
// strong zero normal velocity (y) on the bottom vertices, and a master-bearing
// extrapolation line for the tangential DOF of the dry vertex 2,
// u_2x = 2 u_1x - u_0x, as small-cut aggregation would install.
class WallAndExtrapolation final : public ISystemConstraint {
public:
    explicit WallAndExtrapolation(FieldId field) : field_(field) {}
    void apply(const systems::FESystem& system, AffineConstraints& constraints) override
    {
        for (GlobalIndex v : {0, 1, 2}) {
            constraints.addDirichlet(componentDof(system, field_, v, 1), 0.0);
        }
        const auto slave = componentDof(system, field_, 2, 0);
        constraints.addLine(slave);
        constraints.addEntry(slave, componentDof(system, field_, 1, 0), 2.0);
        constraints.addEntry(slave, componentDof(system, field_, 0, 0), -1.0);
    }
    bool updateValues(const systems::FESystem&, AffineConstraints&, double, double) override
    {
        return false;
    }
    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }
    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override
    {
        systems::SetupStorageRequirements r;
        r.entity_dof_map = true;
        return r;
    }

private:
    FieldId field_{INVALID_FIELD_ID};
};

} // namespace

TEST(ContactPointVelocityConstraint, ExpandsTheDryVertexLineAndLeavesNormalDirichletAlone)
{
    auto mesh = std::make_shared<StripMeshAccess>(0, false);
    systems::FESystem system(mesh);
    auto space = spaces::VectorSpace(spaces::SpaceType::H1, ElementType::Triangle3, 1, 2);
    const auto u = system.addField(systems::FieldSpec{
        .name = "u", .space = space, .components = space->value_dimension()});
    system.addOperator("momentum");
    system.addSystemConstraint(std::make_unique<WallAndExtrapolation>(u));
    ContactPointVelocityLaw contact;
    contact.wall_boundary_markers = {kWallMarker};
    contact.wall_normal = {0.0, -1.0, 0.0};
    contact.level_set_field_name = "phi";
    contact.level_set_vertex_value = [mesh](GlobalIndex v) {
        return phiAt(mesh->getNodeCoordinates(v));
    };
    contact.contact_speed = law;
    auto owned = std::make_unique<ContactPointVelocityConstraint>(u, std::move(contact));
    auto* constraint = owned.get();
    system.addSystemConstraint(std::move(owned));
    systems::SetupInputs inputs;
    inputs.topology_override = serialTopology(*mesh);
    ASSERT_NO_THROW(system.setup(systems::SetupOptions{}, inputs));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    // 0.7 u_1 + 0.3 (2 u_1 - u_0) = 1.3 u_1 - 0.3 u_0 = target: slave u_1x.
    const Real target = law(-0.2 / std::sqrt(1.04));
    const auto u0 = componentDof(system, u, 0, 0);
    const auto u1 = componentDof(system, u, 1, 0);
    const auto u2 = componentDof(system, u, 2, 0);
    const auto& c = system.constraints();
    const auto line = c.getConstraint(u1);
    ASSERT_TRUE(line.has_value());
    ASSERT_EQ(line->entries.size(), 1u);
    EXPECT_EQ(line->entries[0].master_dof, u0);
    EXPECT_NEAR(line->entries[0].weight, 0.3 / 1.3, 1.0e-14);
    EXPECT_NEAR(line->inhomogeneity, target / 1.3, 1.0e-14);
    ASSERT_EQ(constraint->samples().size(), 1u);
    EXPECT_EQ(constraint->samples()[0].slave_dof, u1);
    EXPECT_NEAR(constraint->samples()[0].fraction, 0.3, 1.0e-14);
    // The wall-normal DOFs keep their strong zero lines.
    for (GlobalIndex v : {0, 1, 2}) {
        const auto normal = c.getConstraint(componentDof(system, u, v, 1));
        ASSERT_TRUE(normal.has_value());
        EXPECT_TRUE(normal->isDirichlet());
        EXPECT_EQ(normal->inhomogeneity, 0.0);
    }
    // Distributed state: the interpolated contact velocity equals the law.
    std::vector<double> state(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (std::size_t i = 0; i < state.size(); ++i) {
        state[i] = 0.1 + 0.01 * static_cast<double>(i);
    }
    c.distribute(state);
    const double interpolated = 0.7 * state[static_cast<std::size_t>(u1)] +
                                0.3 * state[static_cast<std::size_t>(u2)];
    EXPECT_NEAR(interpolated, target, 1.0e-13);
}

TEST(ContactPointVelocityConstraint, RejectsANonAxisWallNormal)
{
    ContactPointVelocityLaw contact;
    contact.wall_boundary_markers = {kWallMarker};
    contact.wall_normal = {0.6, -0.8, 0.0};
    contact.level_set_field_name = "phi";
    contact.contact_speed = law;
    EXPECT_THROW(ContactPointVelocityConstraint(0, std::move(contact)), std::invalid_argument);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

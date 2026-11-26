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

#include "ContactPointVelocityTestMesh.h"
#include "Constraints/ContactPointVelocityConstraint.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {
namespace {

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
    explicit WallAndExtrapolation(FieldId field, std::array<GlobalIndex, 3> vertices)
        : field_(field), vertices_(vertices)
    {
    }
    void apply(const systems::FESystem& system, AffineConstraints& constraints) override
    {
        for (GlobalIndex v : {0, 1, 2}) {
            constraints.addDirichlet(componentDof(system, field_, v, 1), 0.0);
        }
        const auto slave = componentDof(system, field_, vertices_[0], 0);
        constraints.addLine(slave);
        constraints.addEntry(slave, componentDof(system, field_, vertices_[1], 0), 2.0);
        constraints.addEntry(slave, componentDof(system, field_, vertices_[2], 0), -1.0);
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
    std::array<GlobalIndex, 3> vertices_;
};

struct ConstrainedStrip {
    std::shared_ptr<StripMeshAccess> mesh = std::make_shared<StripMeshAccess>();
    systems::FESystem system{mesh};
    FieldId u{INVALID_FIELD_ID};
    ContactPointVelocityConstraint* constraint{};

    explicit ConstrainedStrip(std::function<Real(const std::array<Real, 3>&)> phi,
                              std::array<GlobalIndex, 3> extrapolation = {2, 1, 0},
                              std::vector<int> wall_markers = {kWallMarker},
                              LevelSetConstraintSide liquid_side = LevelSetConstraintSide::Negative)
    {
        auto space = spaces::VectorSpace(spaces::SpaceType::H1, ElementType::Triangle3, 1, 2);
        u = system.addField(systems::FieldSpec{
            .name = "u", .space = space, .components = space->value_dimension()});
        system.addOperator("momentum");
        system.addSystemConstraint(std::make_unique<WallAndExtrapolation>(u, extrapolation));
        ContactPointVelocityLaw contact;
        contact.wall_boundary_markers = std::move(wall_markers);
        contact.level_set_field_name = "phi";
        contact.level_set_vertex_value = [mesh = mesh, phi](GlobalIndex v) {
            return phi(mesh->getNodeCoordinates(v));
        };
        contact.contact_speed = law;
        contact.liquid_side = liquid_side;
        auto owned = std::make_unique<ContactPointVelocityConstraint>(u, std::move(contact));
        constraint = owned.get();
        system.addSystemConstraint(std::move(owned));
        systems::SetupInputs inputs;
        inputs.topology_override = topologyFor(*mesh);
        system.setup(systems::SetupOptions{}, inputs);
    }
};

} // namespace

TEST(ContactPointVelocityConstraint, ExpandsTheDryVertexLineAndLeavesNormalDirichletAlone)
{
    ConstrainedStrip strip(phiAt);
    auto& system = strip.system;
    const auto u = strip.u;
    const auto* constraint = strip.constraint;
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

// Reversing both the level set and the liquid side describes the same liquid,
// so the contact point keeps its Cartesian velocity (free-surface audit A34:
// the lower level set was taken as wet, and a positive-side liquid moved the
// contact point at -0.5 instead of +0.5).
TEST(ContactPointVelocityConstraint, ReversingTheLevelSetAndTheLiquidSideKeepsTheContactVelocity)
{
    const auto phi = [](const std::array<Real, 3>& x) { return x[0] - Real{0.5}; };
    std::vector<Real> velocities;
    for (const bool positive_liquid : {false, true}) {
        ConstrainedStrip strip(
            positive_liquid ? std::function<Real(const std::array<Real, 3>&)>(
                                  [phi](const std::array<Real, 3>& x) { return -phi(x); })
                            : std::function<Real(const std::array<Real, 3>&)>(phi),
            {2, 1, 0}, {kWallMarker},
            positive_liquid ? LevelSetConstraintSide::Positive : LevelSetConstraintSide::Negative);
        ASSERT_NO_THROW(strip.system.rebuildConstraintState());
        ASSERT_EQ(strip.constraint->samples().size(), 1u);
        EXPECT_NEAR(strip.constraint->samples().front().cos_theta, 0.0, 1e-14);
        // u_x at the contact point x = 1/2 of the wall edge (0, 1), every free DOF at zero.
        const auto& c = strip.system.constraints();
        const auto value = [&](GlobalIndex vertex) {
            const auto line = c.getConstraint(componentDof(strip.system, strip.u, vertex, 0));
            return line.has_value() ? line->inhomogeneity : Real{0.0};
        };
        velocities.push_back(Real{0.5} * value(0) + Real{0.5} * value(1));
    }
    EXPECT_NEAR(velocities[0], law(0.0), 1e-14);   // toward the dry side, +x
    EXPECT_NEAR(velocities[1], velocities[0], 1e-14);
}

TEST(ContactPointVelocityConstraint, MaintainsTheSignedLawThroughAWallVertex)
{
    Real direction = 1.0;
    Real crossing = 1.0;
    // The wildcard and the explicit marker both select the same wall edges.
    ConstrainedStrip strip([&](const auto& x) {
        return direction * (x[0] - crossing) - 0.2 * x[1];
    }, {2, 1, 0}, {-1, kWallMarker});
    auto& system = strip.system;
    for (const Real outward : {1.0, -1.0}) {
        direction = outward;
        for (const Real x : {0.9, 1.0, 1.1}) {
            crossing = x;
            SCOPED_TRACE(::testing::Message() << "direction=" << direction << " crossing=" << x);
            ASSERT_NO_THROW(system.rebuildConstraintState());
            const auto& samples = strip.constraint->samples();
            ASSERT_EQ(samples.size(), 1u);
            EXPECT_NEAR(samples[0].point[0], x, 1.0e-14);
            EXPECT_EQ(samples[0].point[1], 0.0);
            const Real cos_theta = -0.2 / std::sqrt(1.04);
            EXPECT_NEAR(samples[0].cos_theta, cos_theta, 1.0e-14);
            EXPECT_NEAR(samples[0].target, law(cos_theta), 1.0e-14);
            const auto left = x <= 1.0 ? 0 : 1;
            const Real t = x - left;
            EXPECT_NEAR(samples[0].fraction, direction > 0.0 ? t : 1.0 - t, 1.0e-14);
            std::vector<double> state(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.37);
            system.constraints().distribute(state);
            const auto u_left = componentDof(system, strip.u, left, 0);
            const auto u_right = componentDof(system, strip.u, left + 1, 0);
            const Real velocity = (1.0 - t) * state[u_left] + t * state[u_right];
            EXPECT_NEAR(direction * velocity, law(cos_theta), 1.0e-13);
            for (GlobalIndex v : {0, 1, 2}) {
                const auto normal = system.constraints().getConstraint(componentDof(system, strip.u, v, 1));
                ASSERT_TRUE(normal.has_value());
                EXPECT_TRUE(normal->isDirichlet());
                EXPECT_EQ(normal->inhomogeneity, 0.0);
            }
        }
    }
}

TEST(ContactPointVelocityConstraint, ExpandsAnAggregatedExactVertex)
{
    ConstrainedStrip strip([](const auto& x) { return (x[0] - 1.0) - 0.2 * x[1]; },
                           {1, 0, 2});  // u_1 = 2 u_0 - u_2.
    ASSERT_NO_THROW(strip.system.rebuildConstraintState());
    const auto& samples = strip.constraint->samples();
    ASSERT_EQ(samples.size(), 1u);
    const auto u0 = componentDof(strip.system, strip.u, 0, 0);
    const auto u2 = componentDof(strip.system, strip.u, 2, 0);
    EXPECT_EQ(samples[0].slave_dof, u0);
    const auto line = strip.system.constraints().getConstraint(u0);
    ASSERT_TRUE(line.has_value());
    ASSERT_EQ(line->entries.size(), 1u);
    EXPECT_EQ(line->entries[0].master_dof, u2);
    EXPECT_DOUBLE_EQ(line->entries[0].weight, 0.5);
    EXPECT_NEAR(line->inhomogeneity, 0.5 * law(-0.2 / std::sqrt(1.04)), 1.0e-14);
}

TEST(ContactPointVelocityConstraint, SkipsWallTangenciesEndpointsAndZeroIntervals)
{
    std::array<Real, 3> phi{};
    ConstrainedStrip strip([&](const auto& x) {
        return phi[static_cast<std::size_t>(x[0])] - 0.2 * x[1];
    });
    for (const std::array<Real, 3>& values : std::vector<std::array<Real, 3>>{
             {-1, 0, -1}, {1, 0, 1}, {0, 1, 2}, {-2, -1, 0},
             {-1, 0, 0}, {0, 0, 1}, {0, 0, 0}}) {
        phi = values;
        SCOPED_TRACE(::testing::PrintToString(values));
        ASSERT_NO_THROW(strip.system.rebuildConstraintState());
        EXPECT_TRUE(strip.constraint->samples().empty());
        EXPECT_FALSE(strip.system.constraints().isConstrained(componentDof(strip.system, strip.u, 1, 0)));
    }
}

TEST(ContactPointVelocityConstraint, RequiresOneFiniteSignedPlanarAxis)
{
    ContactPointVelocityLaw contact;
    contact.wall_boundary_markers = {kWallMarker};
    contact.level_set_field_name = "phi";
    contact.contact_speed = law;
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    for (const std::array<Real, 3>& invalid : std::vector<std::array<Real, 3>>{
             {0.6, -0.8, 0}, {1, 1, 0}, {-1, 1, 0}, {0, 0, 1}, {0, 0, 0},
             {1, nan, 0}, {nan, -1, 0}, {0, -1, nan}, {0, -1, inf}}) {
        contact.wall_normal = invalid;
        EXPECT_THROW(ContactPointVelocityConstraint(0, contact), std::invalid_argument)
            << ::testing::PrintToString(invalid);
    }
    for (const std::array<Real, 3>& valid : std::vector<std::array<Real, 3>>{
             {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}}) {
        contact.wall_normal = valid;
        EXPECT_NO_THROW(ContactPointVelocityConstraint(0, contact));
    }
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

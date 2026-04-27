#include "Constraints/MovingConstraintComposition.h"

#include <gtest/gtest.h>

using namespace svmp::FE::constraints;

TEST(MovingConstraintComposition, OrdersDeterministicallyAndSkipsInactiveEntries)
{
    MovingConstraintEntry sliding;
    sliding.id = "sliding";
    sliding.kind = MovingConstraintKind::SlidingInterface;
    sliding.priority = 40;
    sliding.constrained_dofs = {2};

    MovingConstraintEntry cyclic;
    cyclic.id = "cyclic";
    cyclic.kind = MovingConstraintKind::CyclicPeriodic;
    cyclic.priority = 10;
    cyclic.constrained_dofs = {1};

    MovingConstraintEntry inactive;
    inactive.id = "inactive";
    inactive.active = false;
    inactive.constrained_dofs = {3};

    const auto result = composeMovingConstraints({sliding, inactive, cyclic});
    ASSERT_TRUE(result.ok);
    ASSERT_EQ(result.ordered_entries.size(), 2u);
    EXPECT_EQ(result.ordered_entries[0].id, "cyclic");
    EXPECT_EQ(result.ordered_entries[1].id, "sliding");
}

TEST(MovingConstraintComposition, ReportsConflictingGeometryDependentConstraints)
{
    MovingConstraintEntry periodic;
    periodic.id = "periodic";
    periodic.kind = MovingConstraintKind::GeometryDependentPeriodic;
    periodic.priority = 10;
    periodic.constrained_dofs = {5};

    MovingConstraintEntry rigid;
    rigid.id = "rigid";
    rigid.kind = MovingConstraintKind::RigidRegion;
    rigid.priority = 20;
    rigid.constrained_dofs = {5};

    const auto result = composeMovingConstraints({rigid, periodic});
    EXPECT_FALSE(result.ok);
    ASSERT_EQ(result.conflicts.size(), 1u);
    EXPECT_EQ(result.conflicts[0].constrained_dof, 5);
    EXPECT_EQ(result.conflicts[0].first_id, "periodic");
    EXPECT_EQ(result.conflicts[0].second_id, "rigid");
}

TEST(MovingConstraintComposition, RebuildsOnMeshOrDofLayoutRevisionChange)
{
    MovingConstraintRevisionSnapshot cached;
    cached.geometry_revision = 1;
    cached.fe_dof_layout_revision = 8;

    auto current = cached;
    EXPECT_FALSE(movingConstraintRequiresRebuild(cached, current));

    current.fe_dof_layout_revision = 9;
    EXPECT_TRUE(movingConstraintRequiresRebuild(cached, current));
}

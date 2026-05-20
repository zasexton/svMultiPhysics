#include <gtest/gtest.h>

#include "Constraints/ConstraintDependency.h"
#include "Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "Constraints/MultiPointConstraint.h"
#include "Constraints/PeriodicBC.h"
#include "Constraints/TiedInterfaceConstraint.h"

#include <utility>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(ConstraintDependencyTest, RevisionMasksDetectOnlyDeclaredDomains)
{
    ConstraintRevisionSnapshot cached;
    cached.valid = true;
    cached.geometry = 1;
    cached.topology = 3;

    ConstraintRevisionSnapshot current = cached;
    current.geometry = 2;

    auto deps = ConstraintDependencyMask::meshGeometry();
    EXPECT_TRUE(dependency_changed(deps, cached, current));

    deps = ConstraintDependencyMask::meshBoundaryTopology();
    EXPECT_FALSE(dependency_changed(deps, cached, current));

    current.topology = 4;
    EXPECT_TRUE(dependency_changed(deps, cached, current));
}

TEST(ConstraintDependencyTest, CoordinateMatchedPeriodicDeclaresStructuralMotionDependencies)
{
    std::vector<GlobalIndex> slave_dofs = {0, 1};
    std::vector<std::array<double, 3>> slave_coords = {
        {{0.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
    };
    std::vector<GlobalIndex> master_dofs = {10, 11};
    std::vector<std::array<double, 3>> master_coords = {
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
    };

    PeriodicBC direct({0, 1}, {10, 11});
    EXPECT_FALSE(direct.dependencyDeclaration().any());

    PeriodicBC matched(std::move(slave_dofs),
                       std::move(slave_coords),
                       std::move(master_dofs),
                       std::move(master_coords),
                       std::array<double, 3>{{1.0, 0.0, 0.0}});
    const auto deps = matched.dependencyDeclaration();
    EXPECT_TRUE(deps.structural.geometry);
    EXPECT_TRUE(deps.structural.topology);
    EXPECT_TRUE(deps.structural.numbering);
    EXPECT_TRUE(deps.structural.fe_dof_layout);
}

TEST(ConstraintDependencyTest, MultiPointConstraintsCanDeclareMotionAndTangentHooks)
{
    MPCOptions opts;
    opts.dependency_declaration.value.geometry = true;
    opts.dependency_declaration.structural.fe_dof_layout = true;
    opts.dependency_declaration.tangent_policy = ConstraintTangentPolicy::Analytic;
    opts.dependency_declaration.tangent_hook_name = "rigid-link-linearization";

    MultiPointConstraint mpc(std::vector<MPCEquation>{}, opts);
    mpc.addConstraint(0, 1, 1.0);

    const auto deps = mpc.dependencyDeclaration();
    EXPECT_TRUE(deps.value.geometry);
    EXPECT_TRUE(deps.structural.fe_dof_layout);
    EXPECT_EQ(deps.tangent_policy, ConstraintTangentPolicy::Analytic);
    EXPECT_EQ(deps.tangent_hook_name, "rigid-link-linearization");
}

TEST(ConstraintDependencyTest,
     LevelSetActiveSideConstraintTracksFieldAndCutContextRevisions)
{
    LevelSetActiveSideVertexDirichletConstraint constraint(
        /*field=*/0,
        "phi",
        LevelSetConstraintSide::Negative,
        /*isovalue=*/0.0,
        /*inactive_value=*/0.0,
        /*interface_marker=*/7);

    const auto deps = constraint.dependencyDeclaration();
    EXPECT_TRUE(deps.structural.topology);
    EXPECT_TRUE(deps.structural.ownership);
    EXPECT_TRUE(deps.structural.numbering);
    EXPECT_TRUE(deps.structural.labels);
    EXPECT_TRUE(deps.structural.fe_dof_layout);
    EXPECT_TRUE(deps.structural.fe_constraint_layout);
    EXPECT_TRUE(deps.structural.mesh_field_layout);
    EXPECT_TRUE(deps.structural.mesh_field_values);

    ConstraintRevisionSnapshot cached;
    cached.valid = true;
    cached.mesh_field_layout = 3;
    cached.mesh_field_values = 11;
    cached.fe_constraint_layout = 5;

    auto current = cached;
    current.mesh_field_layout += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));

    current = cached;
    current.mesh_field_values += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));

    current = cached;
    current.fe_constraint_layout += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));
}

TEST(ConstraintDependencyTest, TiedInterfaceConstraintTracksRelationMapRevisions)
{
    ConstraintRevisionSnapshot built_at;
    built_at.valid = true;
    built_at.geometry = 7;
    built_at.topology = 3;
    built_at.ownership = 2;
    built_at.numbering = 5;
    built_at.labels = 11;
    built_at.fe_dof_layout = 13;
    built_at.fe_space = 17;

    TiedInterfaceRelationMap map;
    map.name = "wall-tie";
    map.revision = built_at;
    map.built = true;
    TiedInterfaceRelation relation;
    relation.slave_dof = 4;
    relation.masters = {{10, 0.25}, {11, 0.75}};
    relation.inhomogeneity = 0.5;
    relation.name = "pair-0";
    map.relations.push_back(std::move(relation));

    TiedInterfaceConstraint tied(map);
    const auto deps = tied.dependencyDeclaration();
    EXPECT_TRUE(deps.structural.geometry);
    EXPECT_TRUE(deps.structural.topology);
    EXPECT_TRUE(deps.structural.ownership);
    EXPECT_TRUE(deps.structural.numbering);
    EXPECT_TRUE(deps.structural.labels);
    EXPECT_TRUE(deps.structural.fe_dof_layout);

    EXPECT_FALSE(tied.relationMapStaleFor(built_at));
    auto moved = built_at;
    moved.geometry += 1;
    EXPECT_TRUE(tied.relationMapStaleFor(moved));

    AffineConstraints affine;
    tied.apply(affine);
    affine.close();

    ASSERT_TRUE(affine.isConstrained(4));
    const auto line = affine.getConstraint(4);
    ASSERT_TRUE(line.has_value());
    ASSERT_EQ(line->entries.size(), 2u);
    EXPECT_DOUBLE_EQ(line->entries[0].weight, 0.25);
    EXPECT_DOUBLE_EQ(line->entries[1].weight, 0.75);
    EXPECT_DOUBLE_EQ(line->inhomogeneity, 0.5);
}

TEST(ConstraintDependencyTest, TiedInterfaceDependenciesCoverMotionRemeshRebaseAndLayout)
{
    const auto deps = tiedInterfaceDependencyDeclaration();

    ConstraintRevisionSnapshot cached;
    cached.valid = true;
    cached.geometry = 1;
    cached.reference_rebase = 2;
    cached.topology = 3;
    cached.ownership = 4;
    cached.numbering = 5;
    cached.labels = 6;
    cached.fe_space = 7;
    cached.fe_dof_layout = 8;

    auto current = cached;
    current.geometry += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));

    current = cached;
    current.reference_rebase += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));

    current = cached;
    current.topology += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));

    current = cached;
    current.ownership += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));

    current = cached;
    current.numbering += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));

    current = cached;
    current.labels += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));

    current = cached;
    current.fe_space += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));

    current = cached;
    current.fe_dof_layout += 1;
    EXPECT_TRUE(structural_dependency_changed(deps, cached, current));
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

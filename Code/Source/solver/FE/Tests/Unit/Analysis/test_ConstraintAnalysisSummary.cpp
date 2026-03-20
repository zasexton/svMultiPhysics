/**
 * @file test_ConstraintAnalysisSummary.cpp
 * @brief Unit tests for ConstraintAnalysisSummary
 */

#include <gtest/gtest.h>

#include "Analysis/ConstraintAnalysisSummary.h"
#include "Constraints/AffineConstraints.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::constraints;

namespace {

// Helper: create and close an AffineConstraints with Dirichlet on given DOFs
AffineConstraints makeConstraints(std::span<const GlobalIndex> dirichlet_dofs,
                                   GlobalIndex /*n_total*/) {
    AffineConstraints ac;
    for (auto d : dirichlet_dofs) {
        ac.addDirichlet(d, 0.0);
    }
    ac.close();
    return ac;
}

} // namespace

// ============================================================================
// No constraints
// ============================================================================

TEST(ConstraintAnalysisSummary, NoConstraints_AllUnconstrained) {
    AffineConstraints ac;
    ac.close();

    ConstraintAnalysisSummary::FieldDofRange field0;
    field0.field_id = 0;
    field0.dof_offset = 0;
    field0.num_dofs = 10;
    field0.num_components = 1;

    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0});

    // Should have one aggregate set for field 0
    ASSERT_GE(summary.constrained_sets.size(), 1u);
    EXPECT_EQ(summary.constrained_sets[0].field, FieldId{0});
    EXPECT_EQ(summary.constrained_sets[0].num_constrained_dofs, 0);
    EXPECT_EQ(summary.constrained_sets[0].num_total_dofs, 10);
    EXPECT_DOUBLE_EQ(summary.constrained_sets[0].constrained_fraction, 0.0);

    auto unconstrained = summary.unconstrainedFields();
    ASSERT_EQ(unconstrained.size(), 1u);
    EXPECT_EQ(unconstrained[0], FieldId{0});

    EXPECT_TRUE(summary.fullyConstrainedFields().empty());
    EXPECT_FALSE(summary.hasConflicts());
}

// ============================================================================
// Partial Dirichlet
// ============================================================================

TEST(ConstraintAnalysisSummary, PartialDirichlet_CorrectFraction) {
    // Field 0: DOFs 0-9 (10 total), constrain DOFs 0,1,2 (3/10 = 30%)
    std::array<GlobalIndex, 3> constrained = {0, 1, 2};
    auto ac = makeConstraints(constrained, 10);

    ConstraintAnalysisSummary::FieldDofRange field0;
    field0.field_id = 0;
    field0.dof_offset = 0;
    field0.num_dofs = 10;
    field0.num_components = 1;

    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0});

    ASSERT_GE(summary.constrained_sets.size(), 1u);
    EXPECT_EQ(summary.constrained_sets[0].num_constrained_dofs, 3);
    EXPECT_NEAR(summary.constrained_sets[0].constrained_fraction, 0.3, 1e-10);
    EXPECT_EQ(summary.constrained_sets[0].constraint_source, "StrongDirichlet");

    EXPECT_TRUE(summary.unconstrainedFields().empty());
    EXPECT_TRUE(summary.fullyConstrainedFields().empty());

    EXPECT_NEAR(summary.constrainedFraction(0, -1, -1), 0.3, 1e-10);
}

// ============================================================================
// Full Dirichlet
// ============================================================================

TEST(ConstraintAnalysisSummary, FullDirichlet_FullyConstrained) {
    // Field 0: DOFs 0-3 (4 total), all constrained
    std::array<GlobalIndex, 4> constrained = {0, 1, 2, 3};
    auto ac = makeConstraints(constrained, 4);

    ConstraintAnalysisSummary::FieldDofRange field0;
    field0.field_id = 0;
    field0.dof_offset = 0;
    field0.num_dofs = 4;
    field0.num_components = 1;

    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0});

    EXPECT_EQ(summary.constrained_sets[0].num_constrained_dofs, 4);
    EXPECT_DOUBLE_EQ(summary.constrained_sets[0].constrained_fraction, 1.0);

    auto fully = summary.fullyConstrainedFields();
    ASSERT_EQ(fully.size(), 1u);
    EXPECT_EQ(fully[0], FieldId{0});
}

// ============================================================================
// Multi-field
// ============================================================================

TEST(ConstraintAnalysisSummary, TwoFields_IndependentCounts) {
    // Field 0: DOFs 0-3 (4 total), 2 constrained
    // Field 1: DOFs 4-7 (4 total), 0 constrained
    std::array<GlobalIndex, 2> constrained = {0, 1};
    auto ac = makeConstraints(constrained, 8);

    ConstraintAnalysisSummary::FieldDofRange field0{0, 0, 4, 1};
    ConstraintAnalysisSummary::FieldDofRange field1{1, 4, 4, 1};

    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0, field1});

    // Find aggregate sets
    const ConstrainedDofSet* cs0 = nullptr;
    const ConstrainedDofSet* cs1 = nullptr;
    for (const auto& cs : summary.constrained_sets) {
        if (cs.field == 0 && cs.component == -1) cs0 = &cs;
        if (cs.field == 1 && cs.component == -1) cs1 = &cs;
    }

    ASSERT_NE(cs0, nullptr);
    ASSERT_NE(cs1, nullptr);
    EXPECT_EQ(cs0->num_constrained_dofs, 2);
    EXPECT_EQ(cs1->num_constrained_dofs, 0);

    auto unconstrained = summary.unconstrainedFields();
    ASSERT_EQ(unconstrained.size(), 1u);
    EXPECT_EQ(unconstrained[0], FieldId{1});
}

// ============================================================================
// Multi-component field — per-component breakdown
// ============================================================================

TEST(ConstraintAnalysisSummary, MultiComponent_PerComponentFraction) {
    // Vector field: 3 components, 4 DOFs per component (12 total)
    // Layout: comp0=[0..3], comp1=[4..7], comp2=[8..11]
    // Constrain comp0 fully (DOFs 0-3), comp1 partially (DOFs 4,5)
    AffineConstraints ac;
    for (GlobalIndex d = 0; d < 4; ++d) ac.addDirichlet(d, 0.0);
    ac.addDirichlet(4, 0.0);
    ac.addDirichlet(5, 0.0);
    ac.close();

    ConstraintAnalysisSummary::FieldDofRange field0{0, 0, 12, 3};

    // Provide a component DOF provider for component-blocked layout
    auto comp_dofs = [](FieldId /*field_id*/, int component) -> std::vector<GlobalIndex> {
        // comp0=[0..3], comp1=[4..7], comp2=[8..11]
        std::vector<GlobalIndex> dofs;
        GlobalIndex start = static_cast<GlobalIndex>(component) * 4;
        for (GlobalIndex d = start; d < start + 4; ++d) dofs.push_back(d);
        return dofs;
    };

    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0},
                                                     nullptr, nullptr, comp_dofs);

    // Aggregate: 6/12 = 50%
    EXPECT_NEAR(summary.constrainedFraction(0, -1, -1), 0.5, 1e-10);

    // Per-component: comp0=100%, comp1=50%, comp2=0%
    EXPECT_NEAR(summary.constrainedFraction(0, 0, -1), 1.0, 1e-10);
    EXPECT_NEAR(summary.constrainedFraction(0, 1, -1), 0.5, 1e-10);
    EXPECT_NEAR(summary.constrainedFraction(0, 2, -1), 0.0, 1e-10);
}

// ============================================================================
// Affine relations (master-slave, not Dirichlet)
// ============================================================================

TEST(ConstraintAnalysisSummary, AffineRelation_DetectedAsSource) {
    AffineConstraints ac;
    ac.addLine(0);
    ac.addEntry(0, 1, 1.0);  // DOF 0 = DOF 1 (periodic-like)
    ac.close();

    ConstraintAnalysisSummary::FieldDofRange field0{0, 0, 4, 1};
    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0});

    EXPECT_EQ(summary.constrained_sets[0].num_constrained_dofs, 1);
    EXPECT_EQ(summary.constrained_sets[0].constraint_source, "AffineRelation");
}

// ============================================================================
// Mixed: Dirichlet + affine on same field
// ============================================================================

TEST(ConstraintAnalysisSummary, MixedSources_DetectedAsMixed) {
    AffineConstraints ac;
    ac.addDirichlet(0, 1.0);
    ac.addLine(1);
    ac.addEntry(1, 2, 1.0);
    ac.close();

    ConstraintAnalysisSummary::FieldDofRange field0{0, 0, 4, 1};
    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0});

    EXPECT_EQ(summary.constrained_sets[0].num_constrained_dofs, 2);
    EXPECT_EQ(summary.constrained_sets[0].constraint_source, "Mixed");
}

// ============================================================================
// Total counts
// ============================================================================

TEST(ConstraintAnalysisSummary, TotalCounts) {
    std::array<GlobalIndex, 2> constrained = {0, 1};
    auto ac = makeConstraints(constrained, 8);

    ConstraintAnalysisSummary::FieldDofRange field0{0, 0, 4, 1};
    ConstraintAnalysisSummary::FieldDofRange field1{1, 4, 4, 1};
    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0, field1});

    EXPECT_EQ(summary.totalConstrainedDofs(), 2);
    EXPECT_EQ(summary.totalDofs(), 8);
}

// ============================================================================
// Default construction
// ============================================================================

TEST(ConstraintAnalysisSummary, DefaultConstruction) {
    ConstraintAnalysisSummary summary;
    EXPECT_TRUE(summary.constrained_sets.empty());
    EXPECT_TRUE(summary.conflicts.empty());
    EXPECT_FALSE(summary.hasConflicts());
    EXPECT_EQ(summary.totalConstrainedDofs(), 0);
    EXPECT_EQ(summary.totalDofs(), 0);
    EXPECT_TRUE(summary.unconstrainedFields().empty());
    EXPECT_TRUE(summary.fullyConstrainedFields().empty());
    EXPECT_DOUBLE_EQ(summary.constrainedFraction(0), -1.0);
}

// ============================================================================
// Per-region grouping
// ============================================================================

TEST(ConstraintAnalysisSummary, PerRegion_WithDofRegionProvider) {
    // Field 0: DOFs 0-7 (8 total)
    // Region 0: DOFs 0-3, Region 1: DOFs 4-7
    // Constrain DOFs 0,1 (region 0 only)
    AffineConstraints ac;
    ac.addDirichlet(0, 0.0);
    ac.addDirichlet(1, 0.0);
    ac.close();

    ConstraintAnalysisSummary::FieldDofRange field0{0, 0, 8, 1};

    // DOF→region provider: DOFs 0-3 in region 0, DOFs 4-7 in region 1
    auto dof_region = [](GlobalIndex dof) -> int {
        if (dof < 4) return 0;
        if (dof < 8) return 1;
        return -1;
    };

    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0}, nullptr, dof_region);

    // Aggregate: 2/8 = 25%
    EXPECT_NEAR(summary.constrainedFraction(0, -1, -1), 0.25, 1e-10);

    // Per-region: region 0 has 2/4 = 50%, region 1 has 0/4 = 0%
    EXPECT_NEAR(summary.constrainedFraction(0, -1, 0), 0.5, 1e-10);
    EXPECT_NEAR(summary.constrainedFraction(0, -1, 1), 0.0, 1e-10);
}

TEST(ConstraintAnalysisSummary, PerRegion_NoDofRegionProvider) {
    // Without a provider, no per-region sets should be generated
    AffineConstraints ac;
    ac.addDirichlet(0, 0.0);
    ac.close();

    ConstraintAnalysisSummary::FieldDofRange field0{0, 0, 4, 1};
    auto summary = ConstraintAnalysisSummary::build(ac, std::array{field0});

    // Only aggregate set, no per-region
    for (const auto& cs : summary.constrained_sets) {
        EXPECT_EQ(cs.region, -1);
    }
}

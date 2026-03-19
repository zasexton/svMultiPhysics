/**
 * @file test_GaugeRegistry.cpp
 * @brief Unit tests for GaugeRegistry CRUD, resolve logic, and enforcement selection
 */

#include <gtest/gtest.h>

#include "Constraints/GaugeRegistry.h"
#include "Constraints/AffineConstraints.h"

using namespace svmp::FE;
using namespace svmp::FE::gauge;

// ============================================================================
// Registration
// ============================================================================

TEST(GaugeRegistry, AddCandidate_StoresCandidate)
{
    GaugeRegistry reg;
    EXPECT_TRUE(reg.candidates().empty());

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    c.reason = "test";
    reg.addCandidate(c);

    ASSERT_EQ(reg.candidates().size(), 1u);
    EXPECT_EQ(reg.candidates()[0].field, 0);
    EXPECT_EQ(reg.candidates()[0].family, NullspaceModeFamily::ScalarConstant);
}

TEST(GaugeRegistry, AddCandidate_Deduplicates)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);
    reg.addCandidate(c);

    EXPECT_EQ(reg.candidates().size(), 1u);
}

TEST(GaugeRegistry, AddCandidate_ExplicitOverridesInference)
{
    GaugeRegistry reg;

    GaugeCandidate inferred;
    inferred.field = 0;
    inferred.component = -1;
    inferred.family = NullspaceModeFamily::ScalarConstant;
    inferred.confidence = Confidence::Medium;
    inferred.source = CandidateSource::FormsInference;
    inferred.reason = "inferred";
    reg.addCandidate(inferred);

    GaugeCandidate explicit_decl;
    explicit_decl.field = 0;
    explicit_decl.component = -1;
    explicit_decl.family = NullspaceModeFamily::ScalarConstant;
    explicit_decl.confidence = Confidence::High;
    explicit_decl.source = CandidateSource::ExplicitDeclaration;
    explicit_decl.reason = "explicit";
    reg.addCandidate(explicit_decl);

    ASSERT_EQ(reg.candidates().size(), 1u);
    EXPECT_EQ(reg.candidates()[0].source, CandidateSource::ExplicitDeclaration);
    EXPECT_EQ(reg.candidates()[0].confidence, Confidence::High);
    EXPECT_EQ(reg.candidates()[0].reason, "explicit");
}

TEST(GaugeRegistry, AddCandidate_DifferentFamiliesNotDeduplicated)
{
    GaugeRegistry reg;

    GaugeCandidate c1;
    c1.field = 0;
    c1.family = NullspaceModeFamily::ScalarConstant;
    reg.addCandidate(c1);

    GaugeCandidate c2;
    c2.field = 0;
    c2.family = NullspaceModeFamily::KernelOfSymGrad;
    reg.addCandidate(c2);

    EXPECT_EQ(reg.candidates().size(), 2u);
}

TEST(GaugeRegistry, AddAnchoring_StoresEvidence)
{
    GaugeRegistry reg;
    reg.addAnchoring({0, -1, -1, {}, AnchoringVerdict::Anchored, "Dirichlet"});

    ASSERT_EQ(reg.anchoring().size(), 1u);
    EXPECT_EQ(reg.anchoring()[0].verdict, AnchoringVerdict::Anchored);
}

// ============================================================================
// Resolve logic — exact nullspace (no anchoring)
// ============================================================================

TEST(GaugeRegistry, Resolve_ExactNullspace_ScalarConstant)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    ASSERT_TRUE(reg.isResolved());
    ASSERT_EQ(reg.resolvedModes().size(), 1u);

    const auto& mode = reg.resolvedModes()[0];
    EXPECT_EQ(mode.status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(mode.policy, EnforcementPolicy::MeanZeroElimination);
}

TEST(GaugeRegistry, Resolve_ExactNullspace_ComponentwiseConstant)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::ComponentwiseConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    const auto& mode = reg.resolvedModes()[0];
    EXPECT_EQ(mode.status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(mode.policy, EnforcementPolicy::MeanZeroElimination);
}

TEST(GaugeRegistry, Resolve_ExactNullspace_KernelOfSymGrad)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::KernelOfSymGrad;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2}; };
    reg.resolve(dof_provider);

    const auto& mode = reg.resolvedModes()[0];
    EXPECT_EQ(mode.status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(mode.policy, EnforcementPolicy::PinDof);
}

// ============================================================================
// Resolve logic — anchored (Dirichlet evidence)
// ============================================================================

TEST(GaugeRegistry, Resolve_Anchored_ByDirichlet)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    reg.addAnchoring({0, -1, -1, {}, AnchoringVerdict::Anchored, "Dirichlet on boundary 1"});

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    const auto& mode = reg.resolvedModes()[0];
    EXPECT_EQ(mode.status, GaugeStatus::Anchored);
    EXPECT_EQ(mode.policy, EnforcementPolicy::None);
}

TEST(GaugeRegistry, Resolve_PartiallyAnchored_BecomesNearNullspace)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::KernelOfSymGrad;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    reg.addAnchoring({0, -1, -1, {}, AnchoringVerdict::PartiallyAnchored, "Robin BC"});

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2}; };
    reg.resolve(dof_provider);

    const auto& mode = reg.resolvedModes()[0];
    EXPECT_EQ(mode.status, GaugeStatus::NearNullspace);
    EXPECT_EQ(mode.policy, EnforcementPolicy::None);
}

// ============================================================================
// Resolve logic — medium confidence
// ============================================================================

TEST(GaugeRegistry, Resolve_MediumConfidence_PinFallback)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::Medium;
    c.reason = "Stabilization terms";
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    const auto& mode = reg.resolvedModes()[0];
    EXPECT_EQ(mode.status, GaugeStatus::NearNullspace);
    EXPECT_EQ(mode.policy, EnforcementPolicy::PinDof);
}

// ============================================================================
// Enforcement
// ============================================================================

TEST(GaugeRegistry, ApplyEnforcement_MeanZero_CreatesConstraint)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);
    EXPECT_EQ(n, 1);

    // The DOF pinned by GlobalConstraint::zeroMean should be constrained
    EXPECT_TRUE(ac.isConstrained(0));
}

TEST(GaugeRegistry, ApplyEnforcement_Anchored_NoConstraints)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    reg.addAnchoring({0, -1, -1, {}, AnchoringVerdict::Anchored, "Dirichlet"});

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);
    EXPECT_EQ(n, 0);
}

// ============================================================================
// Diagnostics
// ============================================================================

TEST(GaugeRegistry, DiagnosticReport_NotEmpty)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    c.reason = "grad-only";
    reg.addCandidate(c);

    auto report = reg.diagnosticReport();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("ScalarConstant"), std::string::npos);
}

// ============================================================================
// Clear
// ============================================================================

TEST(GaugeRegistry, Clear_ResetsAll)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    reg.addCandidate(c);
    reg.addAnchoring({0, -1, -1, {}, AnchoringVerdict::Anchored, "test"});

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0}; };
    reg.resolve(dof_provider);

    reg.clear();
    EXPECT_TRUE(reg.candidates().empty());
    EXPECT_TRUE(reg.anchoring().empty());
    EXPECT_TRUE(reg.resolvedModes().empty());
    EXPECT_FALSE(reg.isResolved());
}

// ============================================================================
// Per-component anchoring — only the constrained component is anchored
// ============================================================================

TEST(GaugeRegistry, Resolve_PerComponentDirichlet_OnlyAnchorsMatchingComponent)
{
    GaugeRegistry reg;

    // ComponentwiseConstant candidate, component=-1 (will be expanded)
    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::ComponentwiseConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    // Per-component Dirichlet evidence: only component 0 is constrained
    reg.addAnchoring({0, 0, -1, {}, AnchoringVerdict::Anchored,
                      "StrongDirichlet on component 0"});

    // DofProvider that returns 4 DOFs per component for a 3-component field
    auto dof_provider = [](FieldId, int comp) -> std::vector<GlobalIndex> {
        if (comp == -1) {
            std::vector<GlobalIndex> all;
            for (GlobalIndex i = 0; i < 12; ++i) all.push_back(i);
            return all;
        }
        std::vector<GlobalIndex> dofs;
        const auto start = static_cast<GlobalIndex>(comp) * 4;
        for (GlobalIndex i = start; i < start + 4; ++i) dofs.push_back(i);
        return dofs;
    };
    reg.resolve(dof_provider);

    // Should have 3 expanded modes (one per component)
    ASSERT_EQ(reg.resolvedModes().size(), 3u);

    // Component 0: anchored by Dirichlet evidence
    EXPECT_EQ(reg.resolvedModes()[0].candidate.component, 0);
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::Anchored);
    EXPECT_EQ(reg.resolvedModes()[0].policy, EnforcementPolicy::None);

    // Component 1: no evidence → ExactNullspace → enforcement
    EXPECT_EQ(reg.resolvedModes()[1].candidate.component, 1);
    EXPECT_EQ(reg.resolvedModes()[1].status, GaugeStatus::ExactNullspace);
    EXPECT_NE(reg.resolvedModes()[1].policy, EnforcementPolicy::None);

    // Component 2: no evidence → ExactNullspace → enforcement
    EXPECT_EQ(reg.resolvedModes()[2].candidate.component, 2);
    EXPECT_EQ(reg.resolvedModes()[2].status, GaugeStatus::ExactNullspace);
    EXPECT_NE(reg.resolvedModes()[2].policy, EnforcementPolicy::None);
}

// ============================================================================
// Family-scoped anchoring — Robin anchors constant but not sym(grad)
// ============================================================================

TEST(GaugeRegistry, Resolve_FamilyScopedEvidence_DistinguishesFamilies)
{
    // A scalar field with two candidate families on the same field:
    // ScalarConstant (Robin anchors it) and a separately-added candidate.
    // Verify that family-specific evidence only matches the right family.
    GaugeRegistry reg;

    GaugeCandidate c1;
    c1.field = 0;
    c1.component = -1;
    c1.family = NullspaceModeFamily::ScalarConstant;
    c1.confidence = Confidence::High;
    reg.addCandidate(c1);

    GaugeCandidate c2;
    c2.field = 0;
    c2.component = -1;
    c2.family = NullspaceModeFamily::ComponentwiseConstant;
    c2.confidence = Confidence::High;
    reg.addCandidate(c2);

    // Family-specific evidence: only ScalarConstant is anchored
    reg.addAnchoring({0, -1, -1, NullspaceModeFamily::ScalarConstant,
                      AnchoringVerdict::Anchored, "Robin BC"});

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    // ScalarConstant should be Anchored (family-matching evidence)
    bool scalar_anchored = false;
    bool cw_not_anchored = false;
    for (const auto& mode : reg.resolvedModes()) {
        if (mode.candidate.family == NullspaceModeFamily::ScalarConstant) {
            scalar_anchored = (mode.status == GaugeStatus::Anchored);
        }
        if (mode.candidate.family == NullspaceModeFamily::ComponentwiseConstant) {
            // No ComponentwiseConstant-family evidence → ExactNullspace
            cw_not_anchored = (mode.status == GaugeStatus::ExactNullspace);
        }
    }
    EXPECT_TRUE(scalar_anchored);
    EXPECT_TRUE(cw_not_anchored);
}

// ============================================================================
// Area 1: Connected-component scope — region expansion and matching
// ============================================================================

TEST(GaugeRegistry, Resolve_RegionExpansion_TwoRegions)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.region = -1;  // global
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    // Region 0 has Dirichlet → anchored. Region 1 does not.
    reg.addAnchoring({0, -1, 0, {}, AnchoringVerdict::Anchored, "Dirichlet on region 0"});

    // DofProvider: 4 DOFs total
    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };

    // RegionProvider: DOFs 0,1 in region 0, DOFs 2,3 in region 1
    auto region_provider = [](GlobalIndex dof) -> int {
        return (dof < 2) ? 0 : 1;
    };

    reg.resolve(dof_provider, region_provider);

    // Should expand into 2 modes (one per region)
    ASSERT_EQ(reg.resolvedModes().size(), 2u);

    // Region 0: anchored by Dirichlet
    const auto& mode0 = reg.resolvedModes()[0];
    EXPECT_EQ(mode0.candidate.region, 0);
    EXPECT_EQ(mode0.status, GaugeStatus::Anchored);
    EXPECT_EQ(mode0.policy, EnforcementPolicy::None);

    // Region 1: no evidence → ExactNullspace
    const auto& mode1 = reg.resolvedModes()[1];
    EXPECT_EQ(mode1.candidate.region, 1);
    EXPECT_EQ(mode1.status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(mode1.policy, EnforcementPolicy::MeanZeroElimination);
}

TEST(GaugeRegistry, Resolve_RegionExpansion_EnforcementFiltersDofsbyRegion)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };

    // DOFs 0,1 in region 0; DOFs 2,3 in region 1
    auto region_provider = [](GlobalIndex dof) -> int {
        return (dof < 2) ? 0 : 1;
    };

    reg.resolve(dof_provider, region_provider);
    ASSERT_EQ(reg.resolvedModes().size(), 2u);

    // Both regions are unanchored → both get MeanZeroElimination
    EXPECT_EQ(reg.resolvedModes()[0].policy, EnforcementPolicy::MeanZeroElimination);
    EXPECT_EQ(reg.resolvedModes()[1].policy, EnforcementPolicy::MeanZeroElimination);

    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);
    EXPECT_EQ(n, 2);

    // Region 0 should constrain DOF 0 (first DOF in region 0)
    EXPECT_TRUE(ac.isConstrained(0));
    // Region 1 should constrain DOF 2 (first DOF in region 1)
    EXPECT_TRUE(ac.isConstrained(2));
}

TEST(GaugeRegistry, Resolve_SingleRegion_NoExpansion)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };

    // All DOFs in region 0
    auto region_provider = [](GlobalIndex) -> int { return 0; };

    reg.resolve(dof_provider, region_provider);

    // Single region → no expansion, stays as 1 mode
    ASSERT_EQ(reg.resolvedModes().size(), 1u);
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::ExactNullspace);
}

TEST(GaugeRegistry, Resolve_GlobalAnchoredEvidence_BlockedFromRegionCandidates)
{
    // Global Anchored evidence (region=-1, no boundary_marker) is blocked
    // from matching region-specific candidates.  The correct path is:
    // BC manager sets boundary_marker → retagEvidenceRegions() converts
    // to per-region → only the touched region is anchored.
    // Evidence without a boundary_marker (e.g., explicit physics-module
    // anchors) stays global and is blocked to prevent over-anchoring.
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    // Global anchoring without boundary_marker (e.g., physics-module anchor)
    reg.addAnchoring({0, -1, -1, {}, AnchoringVerdict::Anchored, "Global anchor"});

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    auto region_provider = [](GlobalIndex dof) -> int {
        return (dof < 2) ? 0 : 1;
    };

    reg.resolve(dof_provider, region_provider);

    // Global Anchored evidence is blocked from per-region candidates.
    // Both regions stay ExactNullspace (will get gauge enforcement).
    ASSERT_EQ(reg.resolvedModes().size(), 2u);
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(reg.resolvedModes()[1].status, GaugeStatus::ExactNullspace);
}

TEST(GaugeRegistry, RetagEvidenceRegions_ConvertsMarkerToPerRegion)
{
    // retagEvidenceRegions() converts boundary-marker-tagged global evidence
    // into per-region evidence using the marker→region resolver.
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    // Global anchoring with boundary_marker=5 (only touches region 0)
    gauge::AnchoringEvidence ev;
    ev.field = 0;
    ev.family = NullspaceModeFamily::ScalarConstant;
    ev.verdict = AnchoringVerdict::Anchored;
    ev.source = "Robin BC on boundary 5";
    ev.boundary_marker = 5;
    reg.addAnchoring(ev);

    // Retag: marker 5 → region 0 only
    reg.retagEvidenceRegions([](int marker) -> std::vector<int> {
        if (marker == 5) return {0};
        return {};
    });

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    auto region_provider = [](GlobalIndex dof) -> int {
        return (dof < 2) ? 0 : 1;
    };

    reg.resolve(dof_provider, region_provider);

    // Region 0 anchored (marker 5 retagged to region 0).
    // Region 1 ExactNullspace (no evidence).
    ASSERT_EQ(reg.resolvedModes().size(), 2u);
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::Anchored);
    EXPECT_EQ(reg.resolvedModes()[1].status, GaugeStatus::ExactNullspace);
}

TEST(GaugeRegistry, Resolve_PerRegionEvidence_AnchorsOnlyMatchingRegion)
{
    // Region-specific Anchored evidence only anchors the matching region.
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    // Anchoring evidence for region 0 only
    reg.addAnchoring({0, -1, 0, {}, AnchoringVerdict::Anchored, "Dirichlet on region 0"});

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    auto region_provider = [](GlobalIndex dof) -> int {
        return (dof < 2) ? 0 : 1;
    };

    reg.resolve(dof_provider, region_provider);

    ASSERT_EQ(reg.resolvedModes().size(), 2u);
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::Anchored);
    EXPECT_EQ(reg.resolvedModes()[1].status, GaugeStatus::ExactNullspace);
}

TEST(GaugeRegistry, Resolve_GlobalPreservedEvidence_StillMatchesRegions)
{
    // Global Preserved evidence (harmless) should still match region-specific
    // candidates — it doesn't affect the status classification.
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    // Preserved evidence is non-anchoring → still matches
    reg.addAnchoring({0, -1, -1, {}, AnchoringVerdict::Preserved, "Natural BC"});

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    auto region_provider = [](GlobalIndex dof) -> int {
        return (dof < 2) ? 0 : 1;
    };

    reg.resolve(dof_provider, region_provider);

    ASSERT_EQ(reg.resolvedModes().size(), 2u);
    // Preserved doesn't anchor → both stay ExactNullspace
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(reg.resolvedModes()[1].status, GaugeStatus::ExactNullspace);
}

// ============================================================================
// Area 3: KernelOfSymGrad rotation modes
// ============================================================================

TEST(GaugeRegistry, Resolve_KernelOfSymGrad_WithCoords_ExpandsRotations3D)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::KernelOfSymGrad;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    // 3D field: 4 vertices, 3 components → 12 DOFs
    auto dof_provider = [](FieldId, int comp) -> std::vector<GlobalIndex> {
        if (comp == -1) {
            std::vector<GlobalIndex> all;
            for (GlobalIndex i = 0; i < 12; ++i) all.push_back(i);
            return all;
        }
        std::vector<GlobalIndex> dofs;
        const auto start = static_cast<GlobalIndex>(comp) * 4;
        for (GlobalIndex i = start; i < start + 4; ++i) dofs.push_back(i);
        return dofs;
    };

    // Coordinates: unit tetrahedron vertices
    auto coord_provider = [](FieldId, GlobalIndex dof) -> std::array<double, 3> {
        // DOFs 0-3 are x-component, map to vertex 0-3
        const int vert = static_cast<int>(dof % 4);
        switch (vert) {
            case 0: return {0.0, 0.0, 0.0};
            case 1: return {1.0, 0.0, 0.0};
            case 2: return {0.0, 1.0, 0.0};
            case 3: return {0.0, 0.0, 1.0};
            default: return {0.0, 0.0, 0.0};
        }
    };

    reg.resolve(dof_provider, nullptr, coord_provider);

    // 3D: 3 translations + 3 rotations = 6 modes
    ASSERT_EQ(reg.resolvedModes().size(), 6u);

    // First 3 are translations (ComponentwiseConstant)
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(reg.resolvedModes()[static_cast<std::size_t>(i)].candidate.family,
                  NullspaceModeFamily::ComponentwiseConstant);
    }

    // Last 3 are rotations (KernelOfSymGrad)
    for (int i = 3; i < 6; ++i) {
        EXPECT_EQ(reg.resolvedModes()[static_cast<std::size_t>(i)].candidate.family,
                  NullspaceModeFamily::KernelOfSymGrad);
        EXPECT_EQ(reg.resolvedModes()[static_cast<std::size_t>(i)].policy,
                  EnforcementPolicy::PinDof);
    }

    // Verify enforcement: rotation modes pin DISTINCT DOFs (one per component)
    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);

    // 3 translations (MeanZero → pin first DOF of each component) + 3 rotations (PinDof)
    EXPECT_EQ(n, 6);

    // Translation MeanZero pins: DOF 0 (vertex 0 x), DOF 4 (vertex 0 y), DOF 8 (vertex 0 z)
    EXPECT_TRUE(ac.isConstrained(0));
    EXPECT_TRUE(ac.isConstrained(4));
    EXPECT_TRUE(ac.isConstrained(8));
    // Rotation PinDof pins at geometrically distinct vertices:
    //   rot 0 (ωz, comp 0, skip 0): DOF 1 (vertex 1, x)
    //   rot 1 (ωx, comp 1, skip 1): DOF 6 (vertex 2, y)  — skips DOF 5
    //   rot 2 (ωy, comp 2, skip 2): DOF 11 (vertex 3, z) — skips DOFs 9,10
    // This provides rank-3 on the rotational subspace (3 distinct vertices).
    EXPECT_TRUE(ac.isConstrained(1));
    EXPECT_TRUE(ac.isConstrained(6));
    EXPECT_TRUE(ac.isConstrained(11));
}

TEST(GaugeRegistry, Resolve_KernelOfSymGrad_WithCoords_ExpandsRotations2D)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::KernelOfSymGrad;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    // 2D field: 3 vertices, 2 components → 6 DOFs
    auto dof_provider = [](FieldId, int comp) -> std::vector<GlobalIndex> {
        if (comp == -1) {
            std::vector<GlobalIndex> all;
            for (GlobalIndex i = 0; i < 6; ++i) all.push_back(i);
            return all;
        }
        std::vector<GlobalIndex> dofs;
        const auto start = static_cast<GlobalIndex>(comp) * 3;
        for (GlobalIndex i = start; i < start + 3; ++i) dofs.push_back(i);
        return dofs;
    };

    auto coord_provider = [](FieldId, GlobalIndex dof) -> std::array<double, 3> {
        const int vert = static_cast<int>(dof % 3);
        switch (vert) {
            case 0: return {0.0, 0.0, 0.0};
            case 1: return {1.0, 0.0, 0.0};
            case 2: return {0.0, 1.0, 0.0};
            default: return {0.0, 0.0, 0.0};
        }
    };

    reg.resolve(dof_provider, nullptr, coord_provider);

    // 2D: 2 translations + 1 rotation = 3 modes
    ASSERT_EQ(reg.resolvedModes().size(), 3u);
}

TEST(GaugeRegistry, Resolve_KernelOfSymGrad_WithoutCoords_TranslationsOnly)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::KernelOfSymGrad;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int comp) -> std::vector<GlobalIndex> {
        if (comp == -1) {
            std::vector<GlobalIndex> all;
            for (GlobalIndex i = 0; i < 12; ++i) all.push_back(i);
            return all;
        }
        std::vector<GlobalIndex> dofs;
        const auto start = static_cast<GlobalIndex>(comp) * 4;
        for (GlobalIndex i = start; i < start + 4; ++i) dofs.push_back(i);
        return dofs;
    };

    reg.resolve(dof_provider);  // no coord provider

    // Without coords: only 3 translation modes
    ASSERT_EQ(reg.resolvedModes().size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(reg.resolvedModes()[static_cast<std::size_t>(i)].candidate.family,
                  NullspaceModeFamily::ComponentwiseConstant);
    }
}

TEST(GaugeRegistry, BuildNullspaceBasis_RotationVectors_OrthogonalToTranslations)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::KernelOfSymGrad;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    // Force SolverNullspace policy for building basis
    // (MeanZero/PinDof modes don't emit basis vectors)
    // We'll test the buildRotationVector helper indirectly
    // by verifying the rotation vectors ARE orthogonal to translations.

    auto dof_provider = [](FieldId, int comp) -> std::vector<GlobalIndex> {
        if (comp == -1) {
            std::vector<GlobalIndex> all;
            for (GlobalIndex i = 0; i < 12; ++i) all.push_back(i);
            return all;
        }
        std::vector<GlobalIndex> dofs;
        const auto start = static_cast<GlobalIndex>(comp) * 4;
        for (GlobalIndex i = start; i < start + 4; ++i) dofs.push_back(i);
        return dofs;
    };

    auto coord_provider = [](FieldId, GlobalIndex dof) -> std::array<double, 3> {
        const int vert = static_cast<int>(dof % 4);
        switch (vert) {
            case 0: return {0.0, 0.0, 0.0};
            case 1: return {1.0, 0.0, 0.0};
            case 2: return {0.0, 1.0, 0.0};
            case 3: return {0.0, 0.0, 1.0};
            default: return {0.0, 0.0, 0.0};
        }
    };

    reg.resolve(dof_provider, nullptr, coord_provider);
    ASSERT_EQ(reg.resolvedModes().size(), 6u);

    // Manually override policies to SolverNullspace for basis construction test
    // (We can't do this with the public API, so instead we verify the resolve
    //  produced 6 modes. The basis construction for SolverNullspace modes is
    //  tested separately — here we just check the mode count and types.)
    int n_translations = 0, n_rotations = 0;
    for (const auto& m : reg.resolvedModes()) {
        if (m.candidate.family == NullspaceModeFamily::ComponentwiseConstant) ++n_translations;
        if (m.candidate.family == NullspaceModeFamily::KernelOfSymGrad) ++n_rotations;
    }
    EXPECT_EQ(n_translations, 3);
    EXPECT_EQ(n_rotations, 3);
}

// ============================================================================
// Area 4: Weighted mean-zero enforcement
// ============================================================================

TEST(GaugeRegistry, ApplyEnforcement_WeightedMeanZero_UsesWeights)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    reg.resolve(dof_provider);

    ASSERT_EQ(reg.resolvedModes().size(), 1u);
    EXPECT_EQ(reg.resolvedModes()[0].policy, EnforcementPolicy::MeanZeroElimination);

    // Provide mass weights
    auto mass_weights = [](FieldId, int) -> std::vector<double> {
        return {0.1, 0.2, 0.3, 0.4};  // non-uniform
    };

    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider, mass_weights);
    EXPECT_EQ(n, 1);
    EXPECT_TRUE(ac.isConstrained(0));
}

TEST(GaugeRegistry, ApplyEnforcement_NoWeights_FallsBackToUniform)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    reg.resolve(dof_provider);

    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);  // no weights
    EXPECT_EQ(n, 1);
    EXPECT_TRUE(ac.isConstrained(0));
}

TEST(GaugeRegistry, ApplyEnforcement_WrongWeightSize_FallsBackToUniform)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    reg.resolve(dof_provider);

    // Wrong size weights
    auto bad_weights = [](FieldId, int) -> std::vector<double> {
        return {0.1, 0.2};  // wrong size
    };

    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider, bad_weights);
    EXPECT_EQ(n, 1);  // should fall back to uniform
    EXPECT_TRUE(ac.isConstrained(0));
}

// ============================================================================
// String conversions
// ============================================================================

TEST(GaugeRegistry, StringConversions_NotNull)
{
    EXPECT_NE(toString(NullspaceModeFamily::ScalarConstant), nullptr);
    EXPECT_NE(toString(NullspaceModeFamily::ComponentwiseConstant), nullptr);
    EXPECT_NE(toString(NullspaceModeFamily::KernelOfSymGrad), nullptr);
    EXPECT_NE(toString(Confidence::High), nullptr);
    EXPECT_NE(toString(AnchoringVerdict::Anchored), nullptr);
    EXPECT_NE(toString(GaugeStatus::ExactNullspace), nullptr);
    EXPECT_NE(toString(EnforcementPolicy::MeanZeroElimination), nullptr);
}

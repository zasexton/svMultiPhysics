/**
 * @file test_GaugeAdapter.cpp
 * @brief Unit tests for GaugeAdapter — ProblemAnalysisReport ↔ GaugeRegistry bridge
 */

#include <gtest/gtest.h>

#include "Analysis/GaugeAdapter.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/FormStructureAnalyzer.h"
#include "Analysis/BoundaryConditionDescriptor.h"

#include "Constraints/GaugeRegistry.h"
#include "Forms/NullspaceAnalyzer.h"
#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;
using namespace svmp::FE::gauge;

namespace {

std::shared_ptr<spaces::FunctionSpace> scalarH1() {
    return std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
}

std::shared_ptr<spaces::FunctionSpace> vectorH1(int dim = 3) {
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    return std::make_shared<spaces::ProductSpace>(base, dim);
}

} // namespace

// ============================================================================
// claimsToCandidates
// ============================================================================

TEST(GaugeAdapter, ClaimsToCandidates_ScalarNullspace) {
    ProblemAnalysisReport report;
    PropertyClaim c;
    c.kind = PropertyKind::Nullspace;
    c.status = PropertyStatus::Exact;
    c.confidence = AnalysisConfidence::High;
    c.field = 0;
    c.component = -1;
    c.region = -1;
    c.description = "constant shift nullspace";
    report.claims.push_back(c);

    auto candidates = claimsToCandidates(report);
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].field, FieldId{0});
    EXPECT_EQ(candidates[0].family, NullspaceModeFamily::ScalarConstant);
    EXPECT_EQ(candidates[0].confidence, Confidence::High);
}

TEST(GaugeAdapter, ClaimsToCandidates_RigidBodyNullspace) {
    ProblemAnalysisReport report;
    PropertyClaim c;
    c.kind = PropertyKind::Nullspace;
    c.status = PropertyStatus::Exact;
    c.confidence = AnalysisConfidence::High;
    c.field = 0;
    c.description = "rigid-body modes via sym(grad)";
    report.claims.push_back(c);

    auto candidates = claimsToCandidates(report);
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].family, NullspaceModeFamily::KernelOfSymGrad);
}

TEST(GaugeAdapter, ClaimsToCandidates_VectorNullspace) {
    ProblemAnalysisReport report;
    PropertyClaim c;
    c.kind = PropertyKind::Nullspace;
    c.status = PropertyStatus::Exact;
    c.field = 0;
    c.description = "vector field per-component constant shifts";
    report.claims.push_back(c);

    auto candidates = claimsToCandidates(report);
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].family, NullspaceModeFamily::ComponentwiseConstant);
}

TEST(GaugeAdapter, ClaimsToCandidates_NonNullspaceIgnored) {
    ProblemAnalysisReport report;
    PropertyClaim c;
    c.kind = PropertyKind::MixedSaddlePoint;
    c.field = 0;
    report.claims.push_back(c);

    auto candidates = claimsToCandidates(report);
    EXPECT_TRUE(candidates.empty());
}

TEST(GaugeAdapter, ClaimsToCandidates_MediumConfidence) {
    ProblemAnalysisReport report;
    PropertyClaim c;
    c.kind = PropertyKind::Nullspace;
    c.status = PropertyStatus::Likely;
    c.confidence = AnalysisConfidence::Medium;
    c.field = 0;
    c.description = "stabilized constant nullspace";
    report.claims.push_back(c);

    auto candidates = claimsToCandidates(report);
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].confidence, Confidence::Medium);
}

// ============================================================================
// descriptorsToEvidence
// ============================================================================

TEST(GaugeAdapter, DescriptorsToEvidence_DirichletAnchors) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.boundary_marker = 3;
    desc.trace_kind = TraceKind::Value;
    desc.enforcement_kind = EnforcementKind::Strong;
    desc.anchors_constant_mode = true;
    desc.anchors_rigid_body_translation = true;
    desc.source = "EssentialBC on marker 3";

    auto evidence = descriptorsToEvidence({desc});
    // Should produce evidence for ScalarConstant, ComponentwiseConstant, and KernelOfSymGrad
    EXPECT_GE(evidence.size(), 2u);

    bool found_anchored = false;
    bool found_partial = false;
    for (const auto& ev : evidence) {
        EXPECT_EQ(ev.field, FieldId{0});
        EXPECT_EQ(ev.boundary_marker, 3);
        if (ev.verdict == AnchoringVerdict::Anchored) found_anchored = true;
        if (ev.verdict == AnchoringVerdict::PartiallyAnchored) found_partial = true;
    }
    EXPECT_TRUE(found_anchored);
    // KernelOfSymGrad: translation but not rotation → PartiallyAnchored
    EXPECT_TRUE(found_partial);
}

TEST(GaugeAdapter, DescriptorsToEvidence_NeumannPreserves) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.trace_kind = TraceKind::Flux;
    desc.enforcement_kind = EnforcementKind::WeakConsistent;

    auto evidence = descriptorsToEvidence({desc});
    for (const auto& ev : evidence) {
        EXPECT_EQ(ev.verdict, AnchoringVerdict::Preserved);
    }
}

TEST(GaugeAdapter, DescriptorsToEvidence_NonFieldIgnored) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::named(VariableKind::AuxiliaryState, "x");
    desc.anchors_constant_mode = true;

    auto evidence = descriptorsToEvidence({desc});
    EXPECT_TRUE(evidence.empty());
}

// ============================================================================
// populateRegistryFromReport
// ============================================================================

TEST(GaugeAdapter, PopulateRegistry_AddsToRegistry) {
    GaugeRegistry registry;

    ProblemAnalysisReport report;
    PropertyClaim c;
    c.kind = PropertyKind::Nullspace;
    c.status = PropertyStatus::Exact;
    c.confidence = AnalysisConfidence::High;
    c.field = 0;
    c.description = "constant shift nullspace";
    report.claims.push_back(c);

    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.trace_kind = TraceKind::Flux;
    desc.enforcement_kind = EnforcementKind::WeakConsistent;

    populateRegistryFromReport(registry, report, {desc});

    EXPECT_EQ(registry.candidates().size(), 1u);
    EXPECT_GE(registry.anchoring().size(), 1u);
}

// ============================================================================
// Roundtrip: NullspaceAnalyzer vs KernelAnalyzer → GaugeAdapter
// ============================================================================

TEST(GaugeAdapter, Roundtrip_ScalarPoisson) {
    auto space = scalarH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    // Path 1: Direct NullspaceAnalyzer
    NullspaceAnalyzer na;
    auto direct_candidates = na.analyze(residual, std::array{fid});

    // Path 2: KernelAnalyzer → GaugeAdapter
    ProblemAnalysisContext ctx;
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {fid};
    rec.residual_expr = residual.nodeShared();
    ctx.addFormulationRecord(rec);

    auto analyzer = ProblemAnalyzer::createDefault();
    auto report = analyzer.analyze(ctx);
    auto adapter_candidates = claimsToCandidates(report);

    // Both should produce one ScalarConstant candidate
    ASSERT_EQ(direct_candidates.size(), 1u);
    ASSERT_EQ(adapter_candidates.size(), 1u);
    EXPECT_EQ(direct_candidates[0].field, adapter_candidates[0].field);
    EXPECT_EQ(direct_candidates[0].family, adapter_candidates[0].family);
    EXPECT_EQ(direct_candidates[0].confidence, adapter_candidates[0].confidence);
}

TEST(GaugeAdapter, Roundtrip_LinearElasticity) {
    auto space = vectorH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx();

    // Direct
    NullspaceAnalyzer na;
    auto direct = na.analyze(residual, std::array{fid});

    // Via adapter
    ProblemAnalysisContext ctx;
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {fid};
    rec.residual_expr = residual.nodeShared();
    ctx.addFormulationRecord(rec);

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);
    auto adapted = claimsToCandidates(report);

    ASSERT_EQ(direct.size(), 1u);
    ASSERT_EQ(adapted.size(), 1u);
    EXPECT_EQ(direct[0].family, adapted[0].family);
    EXPECT_EQ(direct[0].family, NullspaceModeFamily::KernelOfSymGrad);
}

TEST(GaugeAdapter, Roundtrip_VectorGradient) {
    auto space = vectorH1();
    const FieldId fid = 0;

    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    NullspaceAnalyzer na;
    auto direct = na.analyze(residual, std::array{fid});

    ProblemAnalysisContext ctx;
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {fid};
    rec.residual_expr = residual.nodeShared();
    ctx.addFormulationRecord(rec);

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);
    auto adapted = claimsToCandidates(report);

    ASSERT_EQ(direct.size(), 1u);
    EXPECT_EQ(direct[0].family, NullspaceModeFamily::ComponentwiseConstant);

    // The analysis pipeline emits per-component claims (one per vector component),
    // which the GaugeAdapter maps to per-component candidates. The direct
    // NullspaceAnalyzer emits one field-wide ComponentwiseConstant candidate.
    // Both are correct representations — the per-component version is more precise.
    ASSERT_EQ(adapted.size(), 3u);  // 3 components for 3D vector
    for (const auto& c : adapted) {
        EXPECT_EQ(c.field, fid);
        EXPECT_EQ(c.family, NullspaceModeFamily::ScalarConstant);  // per-component = scalar constant per component
    }
}

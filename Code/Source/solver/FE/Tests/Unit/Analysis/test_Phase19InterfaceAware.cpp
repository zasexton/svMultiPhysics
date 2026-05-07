/**
 * @file test_Phase19InterfaceAware.cpp
 * @brief Phase 19 — Interface-aware and multiphysics integration tests
 *
 * Tests exercising:
 *   - Interface-face contributions with InterfaceTopologyContext
 *   - Handwritten kernel analysisContributions()
 *   - Global-kernel-only mixed systems
 *   - CoupledBoundaryManager lifecycle across repeated setup
 *   - Mixed FormExpr + handwritten kernel operators
 *   - Interface validation diagnostics
 */

#include <gtest/gtest.h>

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/InterfaceTopologyContext.h"
#include "Analysis/BoundaryConditionDescriptor.h"

#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;

namespace {

auto scalarH1() { return std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1); }
auto vectorH1(int dim = 3) {
    return std::make_shared<spaces::ProductSpace>(
        std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1), dim);
}

} // namespace

// ============================================================================
// Interface Nitsche — shared-node cells with InterfaceMesh
// ============================================================================

TEST(Phase19, InterfaceNitsche_SharedNodeCells) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Poisson formulation
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.active_variables.push_back(VariableKey::field(0));
    ctx.addFormulationRecord(rec);

    // Interface Nitsche contribution (from handwritten kernel)
    ContributionDescriptor nitsche_cd;
    nitsche_cd.operator_tag = "interface_nitsche";
    nitsche_cd.origin = "InterfaceNitscheKernel";
    nitsche_cd.domain = DomainKind::InterfaceFace;
    nitsche_cd.interface_scope = InterfaceScope::SpecificMarker;
    nitsche_cd.interface_marker = 1;
    nitsche_cd.role = ContributionRole::BoundaryConstraint;
    nitsche_cd.traits = OperatorTraitFlags::HasSecondOrder | OperatorTraitFlags::NullspaceLifting;
    nitsche_cd.test_variables = {VariableKey::field(0)};
    nitsche_cd.trial_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(nitsche_cd));

    // Penalty stabilization
    ContributionDescriptor penalty_cd;
    penalty_cd.operator_tag = "interface_nitsche";
    penalty_cd.origin = "InterfaceNitscheKernel";
    penalty_cd.domain = DomainKind::InterfaceFace;
    penalty_cd.role = ContributionRole::StabilizationBlock;
    penalty_cd.traits = OperatorTraitFlags::HasMass;
    penalty_cd.test_variables = {VariableKey::field(0)};
    penalty_cd.trial_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(penalty_cd));

    // InterfaceTopologyContext with marker 1
    InterfaceTopologyContext itopo;
    InterfaceFaceRecord irec;
    irec.interface_marker = 1;
    irec.minus_cell = 0;
    irec.plus_cell = 1;
    irec.minus_local_face = 0;
    irec.plus_local_face = 0;
    irec.is_two_sided = true;
    irec.has_orientation = true;
    itopo.faces.push_back(irec);
    itopo.marker_to_faces[1].push_back(0);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    auto report = analyzer.analyze(ctx);

    // No interface validation errors (marker 1 exists)
    for (const auto& issue : report.issues) {
        EXPECT_NE(issue.severity, IssueSeverity::Error)
            << "Unexpected error: " << issue.message;
    }

    // Stabilization detected
    EXPECT_GE(report.countByKind(PropertyKind::Stabilization), 1u);
}

// ============================================================================
// Interface handwritten kernel with analysisContributions() only
// ============================================================================

TEST(Phase19, InterfaceHandwrittenKernel) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Handwritten interface kernel provides only ContributionDescriptor
    auto cd = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "interface_op", "HandwrittenInterfaceKernel");
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::SpecificMarker;
    cd.interface_marker = 3;
    ctx.addContribution(std::move(cd));

    // Matching InterfaceMesh
    InterfaceTopologyContext itopo;
    InterfaceFaceRecord irec;
    irec.interface_marker = 3;
    irec.minus_cell = 0;
    irec.plus_cell = 1;
    irec.minus_local_face = 0;
    irec.plus_local_face = 0;
    irec.is_two_sided = true;
    irec.has_orientation = true;
    itopo.faces.push_back(irec);
    itopo.marker_to_faces[3].push_back(0);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    auto report = analyzer.analyze(ctx);

    // Symmetry detected from the DiagonalBlock contribution
    EXPECT_GE(report.countByKind(PropertyKind::OperatorSymmetry), 1u);

    // No validation errors
    for (const auto& issue : report.issues) {
        EXPECT_NE(issue.severity, IssueSeverity::Error);
    }
}

// ============================================================================
// Global-kernel-only mixed system
// ============================================================================

TEST(Phase19, GlobalKernelMixedSystem) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Two variables coupled only through a global kernel
    auto field_key = VariableKey::field(0);
    auto aux_key = VariableKey::named(VariableKind::AuxiliaryState, "global_scalar");

    // Diagonal block for field 0
    auto diag = ContributionDescriptor::diagonalSymmetric(
        field_key, "global_system", "GlobalMixedKernel");
    ctx.addContribution(std::move(diag));

    // Global coupling between field and aux state
    auto coupling = ContributionDescriptor::globalCoupling(
        {field_key}, {aux_key}, "global_system", "GlobalMixedKernel");
    ctx.addContribution(std::move(coupling));

    auto report = analyzer.analyze(ctx);

    // CoupledSystemStructure detected
    EXPECT_GE(report.countByKind(PropertyKind::CoupledSystemStructure), 1u);
}

// ============================================================================
// CoupledBoundaryManager records persist across setup cycles
// ============================================================================

TEST(Phase19, CoupledBoundaryRepeatedSetup) {
    ProblemAnalysisContext ctx;

    // Simulate definition-time contributions (from CoupledBoundaryManager)
    auto def_contrib = ContributionDescriptor::globalCoupling(
        {VariableKey::field(0)},
        {VariableKey::named(VariableKind::BoundaryFunctional, "Q")},
        "coupled_boundary", "CoupledBoundaryManager");
    ctx.addContribution(def_contrib);

    auto v1 = ctx.inputsVersion();
    EXPECT_EQ(ctx.contributions().size(), 1u);

    // Simulate first setup: add setup-time contributions
    auto setup_contrib = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "equations", "KernelSetup");
    ctx.addContribution(std::move(setup_contrib));

    EXPECT_EQ(ctx.contributions().size(), 2u);

    // Verify definition-time contribution is at index 0
    EXPECT_EQ(ctx.contributions()[0].origin, "CoupledBoundaryManager");
    EXPECT_EQ(ctx.contributions()[1].origin, "KernelSetup");
}

// ============================================================================
// Mixed FormExpr + handwritten kernel in same operator
// ============================================================================

TEST(Phase19, MixedFormExprPlusHandwritten) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // FormExpr-based formulation (Poisson)
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.block_residual_exprs.push_back({{0, 0}, residual.nodeShared()});
    rec.active_variables.push_back(VariableKey::field(0));
    ctx.addFormulationRecord(rec);

    // FormExpr-lowered contribution (simulating what FormsInstaller does)
    auto form_cd = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "equations", "FormsInstaller");
    NullspaceHint nh;
    nh.family = NullspaceFamily::ScalarConstant;
    nh.field = 0;
    nh.confidence = AnalysisConfidence::High;
    nh.reason = "constant shift nullspace";
    form_cd.nullspace_hints.push_back(nh);
    ctx.addContribution(std::move(form_cd));

    // Handwritten kernel adds a stabilization term
    auto hw_cd = ContributionDescriptor::stabilization(
        VariableKey::field(0), "equations", "HandwrittenStabilization");
    ctx.addContribution(std::move(hw_cd));

    auto report = analyzer.analyze(ctx);

    // Nullspace detected (from FormExpr contribution hint)
    EXPECT_GE(report.countByKind(PropertyKind::Nullspace), 1u);

    // Stabilization detected (from handwritten contribution)
    EXPECT_GE(report.countByKind(PropertyKind::Stabilization), 1u);

    // Symmetry detected (from FormExpr DiagonalBlock)
    EXPECT_GE(report.countByKind(PropertyKind::OperatorSymmetry), 1u);
}

// ============================================================================
// Missing InterfaceMesh diagnostic (post-setup)
// ============================================================================

TEST(Phase19, MissingInterfaceMesh_Diagnostic) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Contribution targeting interface marker 7
    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::SpecificMarker;
    cd.interface_marker = 7;
    cd.test_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(cd));

    // Post-setup: InterfaceTopologyContext exists but marker 7 is NOT registered
    InterfaceTopologyContext itopo;
    InterfaceFaceRecord irec;
    irec.interface_marker = 1;  // only marker 1 registered
    itopo.faces.push_back(irec);
    itopo.marker_to_faces[1].push_back(0);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    auto report = analyzer.analyze(ctx);

    // Should have Error for missing marker 7
    bool has_missing_error = false;
    for (const auto& issue : report.issues) {
        if (issue.severity == IssueSeverity::Error &&
            issue.message.find("7") != std::string::npos) {
            has_missing_error = true;
        }
    }
    EXPECT_TRUE(has_missing_error);

    // Should also have Info for unused marker 1
    bool has_unused_info = false;
    for (const auto& issue : report.issues) {
        if (issue.severity == IssueSeverity::Info &&
            issue.message.find("1") != std::string::npos) {
            has_unused_info = true;
        }
    }
    EXPECT_TRUE(has_unused_info);
}

// ============================================================================
// One-sided interface face set
// ============================================================================

TEST(Phase19, OneSidedInterface) {
    ProblemAnalysisContext ctx;

    // One-sided interface (boundary-like): plus_cell is invalid
    InterfaceTopologyContext itopo;
    InterfaceFaceRecord irec;
    irec.interface_marker = 2;
    irec.minus_cell = 0;
    irec.plus_cell = INVALID_GLOBAL_INDEX;
    irec.is_two_sided = false;
    irec.has_orientation = false;
    itopo.faces.push_back(irec);
    itopo.marker_to_faces[2].push_back(0);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    const auto* ictx = ctx.interfaceTopologyContext();
    ASSERT_NE(ictx, nullptr);
    ASSERT_EQ(ictx->numFaces(), 1u);
    EXPECT_FALSE(ictx->faces[0].is_two_sided);
    EXPECT_EQ(ictx->faces[0].plus_cell, INVALID_GLOBAL_INDEX);
}

// ============================================================================
// Interface marker mismatch — contribution references wrong marker
// ============================================================================

TEST(Phase19, InterfaceMarkerMismatch) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Contribution targeting marker 5
    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::SpecificMarker;
    cd.interface_marker = 5;
    cd.role = ContributionRole::DiagonalBlock;
    cd.test_variables = {VariableKey::field(0)};
    cd.trial_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(cd));

    // Only marker 10 registered — mismatch
    InterfaceTopologyContext itopo;
    InterfaceFaceRecord irec;
    irec.interface_marker = 10;
    irec.minus_cell = 0;
    irec.plus_cell = 1;
    irec.is_two_sided = true;
    itopo.faces.push_back(irec);
    itopo.marker_to_faces[10].push_back(0);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    auto report = analyzer.analyze(ctx);

    // Error: marker 5 missing
    bool has_error_5 = false;
    for (const auto& issue : report.issues) {
        if (issue.severity == IssueSeverity::Error &&
            issue.message.find("5") != std::string::npos) {
            has_error_5 = true;
        }
    }
    EXPECT_TRUE(has_error_5);

    // Info: marker 10 unused
    bool has_info_10 = false;
    for (const auto& issue : report.issues) {
        if (issue.severity == IssueSeverity::Info &&
            issue.message.find("10") != std::string::npos) {
            has_info_10 = true;
        }
    }
    EXPECT_TRUE(has_info_10);
}

/**
 * @file test_InterfaceValidation.cpp
 * @brief Tests for InterfaceValidationAnalyzer — interface contribution validation
 */

#include <gtest/gtest.h>

#include "Analysis/InterfaceValidationAnalyzer.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/InterfaceTopologyContext.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;

// ============================================================================
// No interface contributions → no issues
// ============================================================================

TEST(InterfaceValidation, NoInterfaceContributions_NoIssues) {
    InterfaceValidationAnalyzer pass;
    ProblemAnalysisContext ctx;
    ProblemAnalysisReport report;

    pass.run(ctx, report);
    EXPECT_TRUE(report.issues.empty());
}

// ============================================================================
// Post-setup: SpecificMarker with matching InterfaceMesh → no issues
// ============================================================================

TEST(InterfaceValidation, PostSetup_MatchingMarker_NoIssues) {
    InterfaceValidationAnalyzer pass;
    ProblemAnalysisContext ctx;

    // Register an interface topology with marker 5
    InterfaceTopologyContext itopo;
    InterfaceFaceRecord rec;
    rec.interface_marker = 5;
    rec.minus_cell = 0;
    rec.plus_cell = 1;
    rec.minus_local_face = 0;
    rec.plus_local_face = 0;
    rec.is_two_sided = true;
    rec.has_orientation = true;
    itopo.faces.push_back(rec);
    itopo.marker_to_faces[5].push_back(0);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    // Contribution targeting marker 5
    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::SpecificMarker;
    cd.interface_marker = 5;
    cd.test_variables = {VariableKey::field(0)};
    cd.trial_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(cd));

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    // No errors — marker 5 exists
    for (const auto& issue : report.issues) {
        EXPECT_NE(issue.severity, IssueSeverity::Error)
            << "Unexpected error: " << issue.message;
    }
}

// ============================================================================
// Post-setup: SpecificMarker with MISSING InterfaceMesh → Error
// ============================================================================

TEST(InterfaceValidation, PostSetup_MissingMarker_Error) {
    InterfaceValidationAnalyzer pass;
    ProblemAnalysisContext ctx;

    // Register interface topology with marker 5 only
    InterfaceTopologyContext itopo;
    InterfaceFaceRecord rec;
    rec.interface_marker = 5;
    itopo.faces.push_back(rec);
    itopo.marker_to_faces[5].push_back(0);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    // Contribution targeting marker 99 — not registered
    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::SpecificMarker;
    cd.interface_marker = 99;
    cd.test_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(cd));

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    // Should have Error for missing marker 99
    bool has_error = false;
    for (const auto& issue : report.issues) {
        if (issue.severity == IssueSeverity::Error &&
            issue.message.find("99") != std::string::npos) {
            has_error = true;
        }
    }
    EXPECT_TRUE(has_error);
}

// ============================================================================
// Post-setup: generated embedded marker covered by cut-interface rules → no error
// ============================================================================

TEST(InterfaceValidation, PostSetup_GeneratedEmbeddedMarker_NoInterfaceMeshRequired) {
    InterfaceValidationAnalyzer pass;
    ProblemAnalysisContext ctx;

    InterfaceTopologyContext itopo;
    itopo.addGeneratedEmbeddedMarker(1030234);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::SpecificMarker;
    cd.interface_marker = 1030234;
    cd.test_variables = {VariableKey::field(0)};
    cd.trial_variables = {VariableKey::field(1)};
    ctx.addContribution(std::move(cd));

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    for (const auto& issue : report.issues) {
        EXPECT_NE(issue.severity, IssueSeverity::Error)
            << "Unexpected error: " << issue.message;
    }
}

// ============================================================================
// Post-setup: AllRegisteredInterfaces with at least one mesh → OK
// ============================================================================

TEST(InterfaceValidation, PostSetup_WildcardWithMesh_OK) {
    InterfaceValidationAnalyzer pass;
    ProblemAnalysisContext ctx;

    InterfaceTopologyContext itopo;
    InterfaceFaceRecord rec;
    rec.interface_marker = 1;
    rec.minus_cell = 0;
    rec.plus_cell = 1;
    rec.minus_local_face = 0;
    rec.plus_local_face = 0;
    rec.is_two_sided = true;
    rec.has_orientation = true;
    itopo.faces.push_back(rec);
    itopo.marker_to_faces[1].push_back(0);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    // Wildcard contribution
    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::AllRegisteredInterfaces;
    cd.test_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(cd));

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    // No errors — wildcard is valid when any mesh exists
    for (const auto& issue : report.issues) {
        EXPECT_NE(issue.severity, IssueSeverity::Error)
            << "Unexpected error: " << issue.message;
    }
}

// ============================================================================
// Post-setup: AllRegisteredInterfaces with NO meshes → Error
// ============================================================================

TEST(InterfaceValidation, PostSetup_WildcardNoMesh_Error) {
    InterfaceValidationAnalyzer pass;
    ProblemAnalysisContext ctx;

    // Empty interface topology
    ctx.setInterfaceTopologyContext(InterfaceTopologyContext{});

    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::AllRegisteredInterfaces;
    cd.test_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(cd));

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    bool has_error = false;
    for (const auto& issue : report.issues) {
        if (issue.severity == IssueSeverity::Error &&
            issue.message.find("AllRegisteredInterfaces") != std::string::npos) {
            has_error = true;
        }
    }
    EXPECT_TRUE(has_error);
}

// ============================================================================
// Post-setup: Unused InterfaceMesh → Info
// ============================================================================

TEST(InterfaceValidation, PostSetup_UnusedMesh_Info) {
    InterfaceValidationAnalyzer pass;
    ProblemAnalysisContext ctx;

    // Register meshes for markers 1 and 2
    InterfaceTopologyContext itopo;
    InterfaceFaceRecord r1; r1.interface_marker = 1;
    InterfaceFaceRecord r2; r2.interface_marker = 2;
    itopo.faces.push_back(r1);
    itopo.faces.push_back(r2);
    itopo.marker_to_faces[1].push_back(0);
    itopo.marker_to_faces[2].push_back(1);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    // Contribution only targets marker 1 — marker 2 is unused
    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::SpecificMarker;
    cd.interface_marker = 1;
    cd.test_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(cd));

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    bool has_info_for_2 = false;
    for (const auto& issue : report.issues) {
        if (issue.severity == IssueSeverity::Info &&
            issue.message.find("2") != std::string::npos) {
            has_info_for_2 = true;
        }
    }
    EXPECT_TRUE(has_info_for_2);
}

// ============================================================================
// Pre-setup: specific marker → Warning (not Error)
// ============================================================================

TEST(InterfaceValidation, PreSetup_SpecificMarker_Warning) {
    InterfaceValidationAnalyzer pass;
    ProblemAnalysisContext ctx;

    // No interface topology (pre-setup)
    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::SpecificMarker;
    cd.interface_marker = 5;
    cd.test_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(cd));

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    // Should have Warning, not Error
    bool has_warning = false;
    bool has_error = false;
    for (const auto& issue : report.issues) {
        if (issue.severity == IssueSeverity::Warning) has_warning = true;
        if (issue.severity == IssueSeverity::Error) has_error = true;
    }
    EXPECT_TRUE(has_warning);
    EXPECT_FALSE(has_error);
}

// ============================================================================
// Wildcard targets all registered meshes → no "unused" info
// ============================================================================

TEST(InterfaceValidation, PostSetup_WildcardTargetsAll_NoUnused) {
    InterfaceValidationAnalyzer pass;
    ProblemAnalysisContext ctx;

    InterfaceTopologyContext itopo;
    InterfaceFaceRecord r1; r1.interface_marker = 1;
    InterfaceFaceRecord r2; r2.interface_marker = 2;
    itopo.faces.push_back(r1);
    itopo.faces.push_back(r2);
    itopo.marker_to_faces[1].push_back(0);
    itopo.marker_to_faces[2].push_back(1);
    ctx.setInterfaceTopologyContext(std::move(itopo));

    // Wildcard contribution targets ALL registered meshes
    ContributionDescriptor cd;
    cd.domain = DomainKind::InterfaceFace;
    cd.interface_scope = InterfaceScope::AllRegisteredInterfaces;
    cd.test_variables = {VariableKey::field(0)};
    ctx.addContribution(std::move(cd));

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    // No "unused" info — wildcard targets all
    for (const auto& issue : report.issues) {
        EXPECT_NE(issue.severity, IssueSeverity::Info)
            << "Unexpected info: " << issue.message;
    }
}

/**
 * @file test_ProblemAnalysisTypes.cpp
 * @brief Unit tests for core analysis types, report queries, and output formatting
 */

#include <gtest/gtest.h>

#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"
#include <sstream>
#include <type_traits>
#include <unordered_set>

using namespace svmp::FE;
using namespace svmp::FE::analysis;

// ============================================================================
// toString — all enums
// ============================================================================

TEST(ProblemAnalysisTypes, ToString_PropertyKind) {
    EXPECT_STREQ(toString(PropertyKind::Nullspace), "Nullspace");
    EXPECT_STREQ(toString(PropertyKind::OverConstraint), "OverConstraint");
    EXPECT_STREQ(toString(PropertyKind::UnderConstraint), "UnderConstraint");
    EXPECT_STREQ(toString(PropertyKind::MixedSaddlePoint), "MixedSaddlePoint");
    EXPECT_STREQ(toString(PropertyKind::CompatibilityCondition), "CompatibilityCondition");
    EXPECT_STREQ(toString(PropertyKind::OperatorSymmetry), "OperatorSymmetry");
    EXPECT_STREQ(toString(PropertyKind::OperatorDefiniteness), "OperatorDefiniteness");
    EXPECT_STREQ(toString(PropertyKind::Stabilization), "Stabilization");
    EXPECT_STREQ(toString(PropertyKind::TopologyScopedKernel), "TopologyScopedKernel");
    EXPECT_STREQ(toString(PropertyKind::ConstraintRedundancy), "ConstraintRedundancy");
    EXPECT_STREQ(toString(PropertyKind::CoupledSystemStructure), "CoupledSystemStructure");
    EXPECT_STREQ(toString(PropertyKind::InterfaceCondition), "InterfaceCondition");
}

TEST(ProblemAnalysisTypes, ToString_PropertyStatus) {
    EXPECT_STREQ(toString(PropertyStatus::Exact), "Exact");
    EXPECT_STREQ(toString(PropertyStatus::Likely), "Likely");
    EXPECT_STREQ(toString(PropertyStatus::Violated), "Violated");
    EXPECT_STREQ(toString(PropertyStatus::Preserved), "Preserved");
    EXPECT_STREQ(toString(PropertyStatus::Unknown), "Unknown");
}

TEST(ProblemAnalysisTypes, ToString_AnalysisConfidence) {
    EXPECT_STREQ(toString(AnalysisConfidence::High), "High");
    EXPECT_STREQ(toString(AnalysisConfidence::Medium), "Medium");
    EXPECT_STREQ(toString(AnalysisConfidence::Low), "Low");
}

TEST(ProblemAnalysisTypes, ToString_IssueSeverity) {
    EXPECT_STREQ(toString(IssueSeverity::Error), "ERROR");
    EXPECT_STREQ(toString(IssueSeverity::Warning), "WARNING");
    EXPECT_STREQ(toString(IssueSeverity::Info), "INFO");
}

TEST(ProblemAnalysisTypes, ToString_VariableKind) {
    EXPECT_STREQ(toString(VariableKind::FieldComponent), "FieldComponent");
    EXPECT_STREQ(toString(VariableKind::AuxiliaryState), "AuxiliaryState");
    EXPECT_STREQ(toString(VariableKind::AuxiliaryInput), "AuxiliaryInput");
    EXPECT_STREQ(toString(VariableKind::AuxiliaryOutput), "AuxiliaryOutput");
    EXPECT_STREQ(toString(VariableKind::BoundaryFunctional), "BoundaryFunctional");
    EXPECT_STREQ(toString(VariableKind::GlobalScalar), "GlobalScalar");
}

TEST(ProblemAnalysisTypes, ToString_DomainKind) {
    EXPECT_STREQ(toString(DomainKind::Cell), "Cell");
    EXPECT_STREQ(toString(DomainKind::Boundary), "Boundary");
    EXPECT_STREQ(toString(DomainKind::InteriorFace), "InteriorFace");
    EXPECT_STREQ(toString(DomainKind::InterfaceFace), "InterfaceFace");
    EXPECT_STREQ(toString(DomainKind::Global), "Global");
    EXPECT_STREQ(toString(DomainKind::CoupledBoundary), "CoupledBoundary");
    EXPECT_STREQ(toString(DomainKind::AuxiliaryCoupling), "AuxiliaryCoupling");
}

// ============================================================================
// VariableKey
// ============================================================================

TEST(VariableKey, FieldComponent_Construction) {
    auto k = VariableKey::field(0, 2);
    EXPECT_EQ(k.kind, VariableKind::FieldComponent);
    EXPECT_EQ(k.field_id, 0);
    EXPECT_EQ(k.component, 2);
    EXPECT_TRUE(k.name.empty());
}

TEST(VariableKey, FieldComponent_DefaultComponent) {
    auto k = VariableKey::field(1);
    EXPECT_EQ(k.component, -1);
}

TEST(VariableKey, Named_AuxiliaryState) {
    auto k = VariableKey::named(VariableKind::AuxiliaryState, "rcr_pressure");
    EXPECT_EQ(k.kind, VariableKind::AuxiliaryState);
    EXPECT_EQ(k.name, "rcr_pressure");
    EXPECT_EQ(k.field_id, INVALID_FIELD_ID);
}

TEST(VariableKey, Named_BoundaryFunctional) {
    auto k = VariableKey::named(VariableKind::BoundaryFunctional, "outlet_flux");
    EXPECT_EQ(k.kind, VariableKind::BoundaryFunctional);
    EXPECT_EQ(k.name, "outlet_flux");
}

TEST(VariableKey, Named_GlobalScalar) {
    auto k = VariableKey::named(VariableKind::GlobalScalar, "total_volume");
    EXPECT_EQ(k.kind, VariableKind::GlobalScalar);
    EXPECT_EQ(k.name, "total_volume");
}

TEST(VariableKey, Equality_FieldComponent) {
    auto a = VariableKey::field(0, 1);
    auto b = VariableKey::field(0, 1);
    auto c = VariableKey::field(0, 2);
    auto d = VariableKey::field(1, 1);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
}

TEST(VariableKey, Equality_Named) {
    auto a = VariableKey::named(VariableKind::AuxiliaryState, "foo");
    auto b = VariableKey::named(VariableKind::AuxiliaryState, "foo");
    auto c = VariableKey::named(VariableKind::AuxiliaryState, "bar");
    auto d = VariableKey::named(VariableKind::BoundaryFunctional, "foo");

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);  // Different kind
}

TEST(VariableKey, Equality_FieldVsNamed) {
    auto field_key = VariableKey::field(0);
    auto named_key = VariableKey::named(VariableKind::AuxiliaryState, "x");
    EXPECT_NE(field_key, named_key);
}

TEST(VariableKey, LessThan_Ordering) {
    auto f0 = VariableKey::field(0);
    auto f1 = VariableKey::field(1);
    auto aux = VariableKey::named(VariableKind::AuxiliaryState, "x");

    // FieldComponent < AuxiliaryState (by enum value)
    EXPECT_LT(f0, aux);
    EXPECT_LT(f0, f1);
}

TEST(VariableKey, Hash_WorksInUnorderedSet) {
    std::unordered_set<VariableKey, VariableKeyHash> s;
    s.insert(VariableKey::field(0, 1));
    s.insert(VariableKey::field(0, 2));
    s.insert(VariableKey::named(VariableKind::AuxiliaryState, "Q"));

    EXPECT_EQ(s.size(), 3u);
    EXPECT_TRUE(s.count(VariableKey::field(0, 1)));
    EXPECT_TRUE(s.count(VariableKey::named(VariableKind::AuxiliaryState, "Q")));
    EXPECT_FALSE(s.count(VariableKey::field(0, 3)));
}

// ============================================================================
// VariableDescriptor
// ============================================================================

TEST(VariableDescriptor, Construction) {
    VariableDescriptor vd;
    vd.key = VariableKey::field(0);
    vd.label = "velocity";
    vd.field_type = FieldType::Vector;
    vd.value_dimension = 3;

    EXPECT_EQ(vd.key.kind, VariableKind::FieldComponent);
    EXPECT_EQ(vd.label, "velocity");
    EXPECT_EQ(vd.value_dimension, 3);
    EXPECT_EQ(vd.region, -1);
}

TEST(VariableDescriptor, NonFieldVariable) {
    VariableDescriptor vd;
    vd.key = VariableKey::named(VariableKind::AuxiliaryState, "rcr_P");
    vd.label = "RCR distal pressure";
    vd.value_dimension = 1;

    EXPECT_EQ(vd.key.kind, VariableKind::AuxiliaryState);
    EXPECT_EQ(vd.key.name, "rcr_P");
}

// ============================================================================
// PropertyClaim
// ============================================================================

TEST(PropertyClaim, DefaultConstruction) {
    PropertyClaim claim;
    EXPECT_EQ(claim.kind, PropertyKind::Nullspace);
    EXPECT_EQ(claim.status, PropertyStatus::Unknown);
    EXPECT_EQ(claim.confidence, AnalysisConfidence::High);
    EXPECT_EQ(claim.field, INVALID_FIELD_ID);
    EXPECT_EQ(claim.component, -1);
    EXPECT_EQ(claim.region, -1);
    EXPECT_EQ(claim.domain, DomainKind::Cell);
    EXPECT_TRUE(claim.variables.empty());
    EXPECT_TRUE(claim.description.empty());
    EXPECT_TRUE(claim.evidence.empty());
    EXPECT_FALSE(claim.applicability_class.has_value());
    EXPECT_FALSE(claim.certification_class.has_value());
    EXPECT_FALSE(claim.matrix_sign_structure_class.has_value());
    EXPECT_FALSE(claim.operator_symmetry_class.has_value());
    EXPECT_FALSE(claim.temporal_stability_class.has_value());
}

TEST(PropertyClaim, AddEvidence) {
    PropertyClaim claim;
    claim.kind = PropertyKind::Nullspace;
    claim.field = 0;
    claim.addEvidence("FormStructureAnalyzer", "field only through gradient",
                      AnalysisConfidence::High);
    claim.addEvidence("BoundaryConditions", "no Dirichlet on marker 3",
                      AnalysisConfidence::Medium, 3);

    ASSERT_EQ(claim.evidence.size(), 2u);
    EXPECT_EQ(claim.evidence[0].source, "FormStructureAnalyzer");
    EXPECT_EQ(claim.evidence[0].confidence, AnalysisConfidence::High);
    EXPECT_EQ(claim.evidence[0].boundary_marker, -1);
    EXPECT_EQ(claim.evidence[1].source, "BoundaryConditions");
    EXPECT_EQ(claim.evidence[1].boundary_marker, 3);
}

TEST(PropertyClaim, WithVariables) {
    PropertyClaim claim;
    claim.kind = PropertyKind::CoupledSystemStructure;
    claim.domain = DomainKind::CoupledBoundary;
    claim.variables.push_back(VariableKey::field(0));
    claim.variables.push_back(VariableKey::named(VariableKind::BoundaryFunctional, "Q_out"));
    claim.variables.push_back(VariableKey::named(VariableKind::AuxiliaryState, "P_d"));

    ASSERT_EQ(claim.variables.size(), 3u);
    EXPECT_EQ(claim.variables[0].kind, VariableKind::FieldComponent);
    EXPECT_EQ(claim.variables[1].kind, VariableKind::BoundaryFunctional);
    EXPECT_EQ(claim.variables[2].kind, VariableKind::AuxiliaryState);
    EXPECT_EQ(claim.domain, DomainKind::CoupledBoundary);
}

// ============================================================================
// ProblemAnalysisReport — queries
// ============================================================================

class ProblemAnalysisReportTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Claim 0: Nullspace, Exact, field 0
        PropertyClaim c0;
        c0.kind = PropertyKind::Nullspace;
        c0.status = PropertyStatus::Exact;
        c0.confidence = AnalysisConfidence::High;
        c0.field = 0;
        c0.description = "Scalar constant nullspace";
        c0.addEvidence("KernelAnalyzer", "field only through gradient");
        report.claims.push_back(c0);

        // Claim 1: Nullspace, Likely, field 1
        PropertyClaim c1;
        c1.kind = PropertyKind::Nullspace;
        c1.status = PropertyStatus::Likely;
        c1.confidence = AnalysisConfidence::Medium;
        c1.field = 1;
        c1.description = "Pressure constant nullspace (stabilized)";
        report.claims.push_back(c1);

        // Claim 2: MixedSaddlePoint, Exact, system-wide
        PropertyClaim c2;
        c2.kind = PropertyKind::MixedSaddlePoint;
        c2.status = PropertyStatus::Exact;
        c2.confidence = AnalysisConfidence::High;
        c2.description = "Velocity-pressure saddle point";
        report.claims.push_back(c2);

        // Claim 3: UnderConstraint, Unknown, field 0
        PropertyClaim c3;
        c3.kind = PropertyKind::UnderConstraint;
        c3.status = PropertyStatus::Unknown;
        c3.field = 0;
        report.claims.push_back(c3);

        // Claim 4: CoupledSystemStructure with variables
        PropertyClaim c4;
        c4.kind = PropertyKind::CoupledSystemStructure;
        c4.status = PropertyStatus::Exact;
        c4.variables.push_back(VariableKey::field(0));
        c4.variables.push_back(VariableKey::named(VariableKind::BoundaryFunctional, "Q_out"));
        c4.description = "FE-boundary coupling";
        report.claims.push_back(c4);

        // Issue 0: Warning
        AnalysisIssue w;
        w.severity = IssueSeverity::Warning;
        w.message = "Compatibility condition may not be satisfied";
        w.related_claim_indices = {0};
        report.issues.push_back(w);
    }

    ProblemAnalysisReport report;
};

TEST_F(ProblemAnalysisReportTest, CountByKind) {
    EXPECT_EQ(report.countByKind(PropertyKind::Nullspace), 2u);
    EXPECT_EQ(report.countByKind(PropertyKind::MixedSaddlePoint), 1u);
    EXPECT_EQ(report.countByKind(PropertyKind::UnderConstraint), 1u);
    EXPECT_EQ(report.countByKind(PropertyKind::CoupledSystemStructure), 1u);
    EXPECT_EQ(report.countByKind(PropertyKind::OverConstraint), 0u);
    EXPECT_EQ(report.countByKind(PropertyKind::InterfaceCondition), 0u);
}

TEST_F(ProblemAnalysisReportTest, CountByStatus) {
    EXPECT_EQ(report.countByStatus(PropertyStatus::Exact), 3u);
    EXPECT_EQ(report.countByStatus(PropertyStatus::Likely), 1u);
    EXPECT_EQ(report.countByStatus(PropertyStatus::Unknown), 1u);
    EXPECT_EQ(report.countByStatus(PropertyStatus::Violated), 0u);
}

TEST_F(ProblemAnalysisReportTest, CountBySeverity) {
    EXPECT_EQ(report.countBySeverity(IssueSeverity::Warning), 1u);
    EXPECT_EQ(report.countBySeverity(IssueSeverity::Error), 0u);
    EXPECT_EQ(report.countBySeverity(IssueSeverity::Info), 0u);
}

TEST_F(ProblemAnalysisReportTest, ClaimsForField) {
    auto field0 = report.claimsForField(0);
    ASSERT_EQ(field0.size(), 2u);
    EXPECT_EQ(field0[0]->kind, PropertyKind::Nullspace);
    EXPECT_EQ(field0[1]->kind, PropertyKind::UnderConstraint);

    auto field1 = report.claimsForField(1);
    ASSERT_EQ(field1.size(), 1u);
    EXPECT_EQ(field1[0]->kind, PropertyKind::Nullspace);

    auto field99 = report.claimsForField(99);
    EXPECT_TRUE(field99.empty());
}

TEST_F(ProblemAnalysisReportTest, ClaimsForVariable_FieldKey) {
    auto var_field0 = report.claimsForVariable(VariableKey::field(0));
    // Should find claims with field=0 (claims 0,3) plus claim 4 (variables contains field(0))
    EXPECT_GE(var_field0.size(), 3u);
}

TEST_F(ProblemAnalysisReportTest, ClaimsForVariable_NamedKey) {
    auto var_q = report.claimsForVariable(
        VariableKey::named(VariableKind::BoundaryFunctional, "Q_out"));
    ASSERT_EQ(var_q.size(), 1u);
    EXPECT_EQ(var_q[0]->kind, PropertyKind::CoupledSystemStructure);
}

TEST_F(ProblemAnalysisReportTest, ClaimsOfKind) {
    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_EQ(nullspace.size(), 2u);

    auto saddle = report.claimsOfKind(PropertyKind::MixedSaddlePoint);
    ASSERT_EQ(saddle.size(), 1u);
    EXPECT_EQ(saddle[0]->status, PropertyStatus::Exact);

    auto coupled = report.claimsOfKind(PropertyKind::CoupledSystemStructure);
    ASSERT_EQ(coupled.size(), 1u);
}

TEST_F(ProblemAnalysisReportTest, HasErrors_HasWarnings) {
    EXPECT_FALSE(report.hasErrors());
    EXPECT_TRUE(report.hasWarnings());

    AnalysisIssue err;
    err.severity = IssueSeverity::Error;
    err.message = "test error";
    report.issues.push_back(err);

    EXPECT_TRUE(report.hasErrors());
    EXPECT_TRUE(report.hasWarnings());
}

// ============================================================================
// ProblemAnalysisReport — output
// ============================================================================

TEST_F(ProblemAnalysisReportTest, Summary) {
    auto s = report.summary();
    EXPECT_NE(s.find("5 claims"), std::string::npos);
    EXPECT_NE(s.find("3 exact"), std::string::npos);
    EXPECT_NE(s.find("1 likely"), std::string::npos);
    EXPECT_NE(s.find("1 unknown"), std::string::npos);
    EXPECT_NE(s.find("1 issues"), std::string::npos);
    EXPECT_NE(s.find("1 warnings"), std::string::npos);
    EXPECT_NE(s.find("0 errors"), std::string::npos);
}

TEST_F(ProblemAnalysisReportTest, Print_ProducesNonEmptyOutput) {
    std::ostringstream oss;
    report.print(oss);
    auto output = oss.str();

    EXPECT_FALSE(output.empty());
    // Header
    EXPECT_NE(output.find("Problem Analysis Report"), std::string::npos);
    // Grouped sections
    EXPECT_NE(output.find("--- Nullspace ---"), std::string::npos);
    EXPECT_NE(output.find("--- MixedSaddlePoint ---"), std::string::npos);
    EXPECT_NE(output.find("--- UnderConstraint ---"), std::string::npos);
    EXPECT_NE(output.find("--- CoupledSystemStructure ---"), std::string::npos);
    // Issue section
    EXPECT_NE(output.find("--- Issues ---"), std::string::npos);
    EXPECT_NE(output.find("[WARNING]"), std::string::npos);
    // Claim details
    EXPECT_NE(output.find("[Exact/High]"), std::string::npos);
    EXPECT_NE(output.find("[Likely/Medium]"), std::string::npos);
    EXPECT_NE(output.find("field=0"), std::string::npos);
    EXPECT_NE(output.find("field=1"), std::string::npos);
}

TEST(ProblemAnalysisReport, EmptyReport_PrintsCleanly) {
    ProblemAnalysisReport report;
    std::ostringstream oss;
    report.print(oss);
    auto output = oss.str();

    EXPECT_NE(output.find("Problem Analysis Report"), std::string::npos);
    EXPECT_NE(output.find("No property claims"), std::string::npos);

    auto s = report.summary();
    EXPECT_NE(s.find("0 claims"), std::string::npos);
}

TEST(ProblemAnalysisReport, ApplicationLogSummarizesGenericDecisions) {
    ProblemAnalysisReport report;

    PropertyClaim dmp;
    dmp.kind = PropertyKind::DiscreteMaximumPrinciple;
    dmp.status = PropertyStatus::Unknown;
    dmp.claim_origin = "DiscreteMonotonicityAnalyzer";
    dmp.field = 7;
    dmp.applicability_class = ApplicabilityClass::Applicable;
    dmp.certification_class = CertificationClass::Unknown;
    dmp.description = "scalar operator requires reduced sign evidence";
    report.claims.push_back(dmp);

    PropertyClaim z_matrix;
    z_matrix.kind = PropertyKind::ZMatrixStructure;
    z_matrix.status = PropertyStatus::Violated;
    z_matrix.claim_origin = "DiscreteMonotonicityAnalyzer";
    z_matrix.field = 7;
    z_matrix.matrix_sign_structure_class = MatrixSignStructureClass::NotZMatrix;
    z_matrix.tested_block_id = "scalar:scalar:cell";
    z_matrix.addEvidence("DiscreteMonotonicityAnalyzer",
                         "positive off-diagonal count is nonzero");
    report.claims.push_back(z_matrix);

    AnalysisSummaryRequest request;
    request.summary_kind = AnalysisSummaryKind::ReducedMatrix;
    request.domain = DomainKind::Cell;
    request.variables = {VariableKey::field(7)};
    request.request_id = "ReducedMatrix:Cell";
    request.source_analyzers = {"DiscreteMonotonicityAnalyzer"};
    request.reasons = {"reduced free-free sign scan is required"};
    report.request_plan.summary_requests.push_back(request);

    AnalysisIssue issue;
    issue.severity = IssueSeverity::Warning;
    issue.message = "Z-matrix violation may break monotonicity";
    issue.related_claim_indices = {1};
    report.issues.push_back(issue);

    std::ostringstream oss;
    report.printApplicationLog(oss);
    const auto output = oss.str();

    EXPECT_NE(output.find("[FE/Analysis] Applicable analyzers: "
                          "DiscreteMonotonicityAnalyzer"),
              std::string::npos);
    EXPECT_NE(output.find("Requested summaries: ReducedMatrix"), std::string::npos);
    EXPECT_NE(output.find("kind=DiscreteMaximumPrinciple"), std::string::npos);
    EXPECT_NE(output.find("status=unknown"), std::string::npos);
    EXPECT_NE(output.find("applicability=Applicable"), std::string::npos);
    EXPECT_NE(output.find("kind=ZMatrixStructure"), std::string::npos);
    EXPECT_NE(output.find("status=violated"), std::string::npos);
    EXPECT_NE(output.find("matrix_sign=NotZMatrix"), std::string::npos);
    EXPECT_NE(output.find("block=scalar:scalar:cell"), std::string::npos);
    EXPECT_NE(output.find("Issue severity=warning"), std::string::npos);
}

TEST(ProblemAnalysisReport, TraceLogIncludesBoundedSummaryEvidence) {
    ProblemAnalysisReport report;
    PropertyClaim claim;
    claim.kind = PropertyKind::ZMatrixStructure;
    claim.status = PropertyStatus::Violated;
    claim.claim_origin = "DiscreteMonotonicityAnalyzer";
    claim.addEvidence("DiscreteMonotonicityAnalyzer",
                      "worst positive off-diagonal retained");
    report.claims.push_back(claim);

    AnalysisSummarySet summaries;

    DiscreteMatrixSummary matrix;
    matrix.block.operator_tag = "generic_cell_block";
    matrix.rows = 4;
    matrix.cols = 4;
    matrix.positive_offdiag_count = 1;
    matrix.row_sum_violation_count = 1;
    matrix.min_row_sum = -0.25;
    matrix.max_row_sum = 1.0;
    matrix.sign_tolerance = 1.0e-12;
    matrix.addWorstEntry({3, 1, 0.5, 0, 0, "positive offdiag"});
    summaries.discrete_matrices.push_back(matrix);

    ReducedMatrixSummary reduced;
    reduced.free_free_matrix = matrix;
    reduced.reduction_kind = ConstraintReductionKind::StrongDirichletElimination;
    reduced.free_dof_count = 3;
    reduced.constrained_dof_count = 1;
    reduced.reduction_exact_for_analysis = true;
    summaries.reduced_matrices.push_back(reduced);

    LocalStencilSummary stencil;
    stencil.block.operator_tag = "generic_element_block";
    stencil.element = 42;
    stencil.positive_offdiag_count = 1;
    stencil.sign_tolerance = 1.0e-12;
    stencil.addWorstLocalEntry({2, 0, 0.125, 0, 0, "local sign"});
    summaries.local_stencils.push_back(stencil);

    MeshGeometryQualitySummary geometry;
    geometry.mesh_revision = 11;
    geometry.min_jacobian = 0.25;
    geometry.max_jacobian = 2.0;
    geometry.poor_quality_element_count = 1;
    geometry.worst_elements = {42, 43};
    summaries.mesh_geometry_quality.push_back(geometry);

    InitialCompatibilitySummary initial;
    initial.initial_constraint_residual = 1.0e-3;
    initial.initial_boundary_residual = 2.0e-3;
    initial.invariant_domain_initial_violation_count = 1;
    initial.residual_tolerance = 1.0e-8;
    summaries.initial_compatibility.push_back(initial);

    std::ostringstream oss;
    report.printTraceLog(oss, &summaries);
    const auto output = oss.str();

    EXPECT_NE(output.find("[FE/Analysis][trace] claim_count=1"), std::string::npos);
    EXPECT_NE(output.find("evidence claim=0"), std::string::npos);
    EXPECT_NE(output.find("row_summary block=generic_cell_block"), std::string::npos);
    EXPECT_NE(output.find("worst_entry row=3 col=1"), std::string::npos);
    EXPECT_NE(output.find("ReducedMatrix constraint_summary"), std::string::npos);
    EXPECT_NE(output.find("free_dofs=3 constrained_dofs=1"), std::string::npos);
    EXPECT_NE(output.find("LocalStencil row_summary"), std::string::npos);
    EXPECT_NE(output.find("MeshGeometry worst_element id=42"), std::string::npos);
    EXPECT_NE(output.find("InitialCompatibility constraint_summary"),
              std::string::npos);
}

// ============================================================================
// ProblemAnalysisContext
// ============================================================================

TEST(ProblemAnalysisContext, DefaultIsEmpty) {
    ProblemAnalysisContext ctx;
    EXPECT_TRUE(ctx.empty());
    EXPECT_TRUE(ctx.fieldDescriptors().empty());
    EXPECT_TRUE(ctx.variableDescriptors().empty());
    EXPECT_TRUE(ctx.formulationRecords().empty());
    EXPECT_TRUE(ctx.contributions().empty());
    EXPECT_TRUE(ctx.bcDescriptors().empty());
    EXPECT_EQ(ctx.topologyContext(), nullptr);
    EXPECT_EQ(ctx.constraintSummary(), nullptr);
    EXPECT_EQ(ctx.analysisSummaries(), nullptr);
    EXPECT_FALSE(ctx.hasSummaryKind(AnalysisSummaryKind::DiscreteMatrix));
    EXPECT_EQ(ctx.summaryCertificationOrUnknown(AnalysisSummaryKind::DiscreteMatrix,
                                                CertificationClass::Certified),
              CertificationClass::Unknown);
    EXPECT_EQ(ctx.inputsVersion(), 0u);
}

TEST(ProblemAnalysisContext, AddFieldDescriptor) {
    ProblemAnalysisContext ctx;

    FieldDescriptor fd;
    fd.field_id = 0;
    fd.name = "velocity";
    fd.field_type = FieldType::Vector;
    fd.value_dimension = 3;
    ctx.addFieldDescriptor(fd);

    EXPECT_FALSE(ctx.empty());
    ASSERT_EQ(ctx.fieldDescriptors().size(), 1u);
    EXPECT_EQ(ctx.inputsVersion(), 1u);

    const auto* result = ctx.fieldDescriptor(0);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->name, "velocity");
    EXPECT_EQ(result->field_type, FieldType::Vector);
    EXPECT_EQ(result->value_dimension, 3);

    EXPECT_EQ(ctx.fieldDescriptor(99), nullptr);
}

TEST(ProblemAnalysisContext, AddFieldDescriptor_UpdateExisting) {
    ProblemAnalysisContext ctx;

    FieldDescriptor fd;
    fd.field_id = 0;
    fd.name = "v1";
    ctx.addFieldDescriptor(fd);

    fd.name = "v2";
    ctx.addFieldDescriptor(fd);

    EXPECT_EQ(ctx.fieldDescriptors().size(), 1u);
    EXPECT_EQ(ctx.fieldDescriptor(0)->name, "v2");
    EXPECT_EQ(ctx.inputsVersion(), 2u);
}

TEST(ProblemAnalysisContext, MultipleFields) {
    ProblemAnalysisContext ctx;

    FieldDescriptor velocity;
    velocity.field_id = 0;
    velocity.name = "velocity";
    velocity.field_type = FieldType::Vector;
    velocity.value_dimension = 3;
    ctx.addFieldDescriptor(velocity);

    FieldDescriptor pressure;
    pressure.field_id = 1;
    pressure.name = "pressure";
    pressure.field_type = FieldType::Scalar;
    pressure.value_dimension = 1;
    ctx.addFieldDescriptor(pressure);

    EXPECT_EQ(ctx.fieldDescriptors().size(), 2u);
    EXPECT_EQ(ctx.fieldDescriptor(0)->name, "velocity");
    EXPECT_EQ(ctx.fieldDescriptor(1)->name, "pressure");
}

TEST(ProblemAnalysisContext, VariableDescriptors) {
    ProblemAnalysisContext ctx;

    VariableDescriptor vd_field;
    vd_field.key = VariableKey::field(0);
    vd_field.label = "velocity";
    vd_field.field_type = FieldType::Vector;
    vd_field.value_dimension = 3;
    ctx.addVariableDescriptor(vd_field);

    VariableDescriptor vd_aux;
    vd_aux.key = VariableKey::named(VariableKind::AuxiliaryState, "rcr_P");
    vd_aux.label = "RCR distal pressure";
    vd_aux.value_dimension = 1;
    ctx.addVariableDescriptor(vd_aux);

    EXPECT_FALSE(ctx.empty());
    ASSERT_EQ(ctx.variableDescriptors().size(), 2u);

    const auto* vf = ctx.variableDescriptor(VariableKey::field(0));
    ASSERT_NE(vf, nullptr);
    EXPECT_EQ(vf->label, "velocity");

    const auto* va = ctx.variableDescriptor(
        VariableKey::named(VariableKind::AuxiliaryState, "rcr_P"));
    ASSERT_NE(va, nullptr);
    EXPECT_EQ(va->label, "RCR distal pressure");

    EXPECT_EQ(ctx.variableDescriptor(VariableKey::field(99)), nullptr);
}

TEST(ProblemAnalysisContext, VariableDescriptor_UpdateExisting) {
    ProblemAnalysisContext ctx;

    VariableDescriptor vd;
    vd.key = VariableKey::named(VariableKind::AuxiliaryState, "x");
    vd.label = "old";
    ctx.addVariableDescriptor(vd);

    vd.label = "new";
    ctx.addVariableDescriptor(vd);

    EXPECT_EQ(ctx.variableDescriptors().size(), 1u);
    EXPECT_EQ(ctx.variableDescriptor(vd.key)->label, "new");
}

TEST(ProblemAnalysisContext, FieldDescriptor_Phase21_SpaceMetadata) {
    ProblemAnalysisContext ctx;

    FieldDescriptor fd;
    fd.field_id = 0;
    fd.name = "sigma";
    fd.space_family = SpaceFamily::HDiv;
    fd.trace_capabilities = TraceCapabilityFlags::NormalComponent;
    fd.has_exact_sequence_structure = true;
    fd.supports_local_balance_closure = true;
    fd.component_extractable = false;
    ctx.addFieldDescriptor(fd);

    const auto* result = ctx.fieldDescriptor(0);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->space_family, SpaceFamily::HDiv);
    EXPECT_TRUE(hasTraceFlag(result->trace_capabilities, TraceCapabilityFlags::NormalComponent));
    EXPECT_FALSE(hasTraceFlag(result->trace_capabilities, TraceCapabilityFlags::Value));
    EXPECT_TRUE(result->has_exact_sequence_structure);
    EXPECT_TRUE(result->supports_local_balance_closure);
}

TEST(ProblemAnalysisContext, FieldDescriptor_MixedDimensionalMetadata) {
    ProblemAnalysisContext ctx;

    FieldDescriptor fd;
    fd.field_id = 7;
    fd.name = "lambda";
    fd.domain = DomainKind::InterfaceFace;
    fd.interface_marker = 17;
    ctx.addFieldDescriptor(fd);

    const auto* result = ctx.fieldDescriptor(7);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->domain, DomainKind::InterfaceFace);
    EXPECT_EQ(result->interface_marker, 17);
}

TEST(ProblemAnalysisContext, InputsVersionIncrementsOnMutation) {
    ProblemAnalysisContext ctx;
    EXPECT_EQ(ctx.inputsVersion(), 0u);

    FieldDescriptor fd;
    fd.field_id = 0;
    ctx.addFieldDescriptor(fd);
    EXPECT_EQ(ctx.inputsVersion(), 1u);

    VariableDescriptor vd;
    vd.key = VariableKey::field(0);
    ctx.addVariableDescriptor(vd);
    EXPECT_EQ(ctx.inputsVersion(), 2u);

    FormulationRecord fr;
    ctx.addFormulationRecord(fr);
    EXPECT_EQ(ctx.inputsVersion(), 3u);

    BoundaryConditionDescriptor bd;
    ctx.addBCDescriptor(bd);
    EXPECT_EQ(ctx.inputsVersion(), 4u);

    ctx.setTopologyContext(TopologyAnalysisContext{});
    EXPECT_EQ(ctx.inputsVersion(), 5u);

    ctx.setConstraintSummary(ConstraintAnalysisSummary{});
    EXPECT_EQ(ctx.inputsVersion(), 6u);

    AnalysisSummarySet summaries;
    summaries.discrete_matrices.push_back(DiscreteMatrixSummary{});
    ctx.setAnalysisSummaries(std::move(summaries));
    EXPECT_EQ(ctx.inputsVersion(), 7u);

    ctx.clearAnalysisSummaries();
    EXPECT_EQ(ctx.inputsVersion(), 8u);
}

TEST(ProblemAnalysisContext, EmptySummaryStorageKeepsContextEmptyButVersioned) {
    ProblemAnalysisContext ctx;
    EXPECT_TRUE(ctx.empty());

    ctx.setAnalysisSummaries(AnalysisSummarySet{});

    EXPECT_EQ(ctx.inputsVersion(), 1u);
    EXPECT_TRUE(ctx.empty());
    ASSERT_NE(ctx.analysisSummaries(), nullptr);
    EXPECT_TRUE(ctx.analysisSummaries()->empty());
    EXPECT_EQ(ctx.analysisSummaries()->totalSummaryCount(), 0u);
}

TEST(ProblemAnalysisContext, AnalysisSummaryStorageAndAvailability) {
    ProblemAnalysisContext ctx;

    AnalysisSummarySet summaries;
    MeshGeometryQualitySummary mesh;
    mesh.mesh_revision = 42;
    mesh.min_jacobian = 0.5;
    summaries.mesh_geometry_quality.push_back(mesh);

    ctx.setAnalysisSummaries(std::move(summaries));

    EXPECT_FALSE(ctx.empty());
    ASSERT_NE(ctx.analysisSummaries(), nullptr);
    EXPECT_EQ(ctx.analysisSummaries()->totalSummaryCount(), 1u);
    EXPECT_TRUE(ctx.hasSummaryKind(AnalysisSummaryKind::MeshGeometryQuality));
    EXPECT_FALSE(ctx.hasSummaryKind(AnalysisSummaryKind::DiscreteMatrix));
    EXPECT_EQ(ctx.summaryCertificationOrUnknown(AnalysisSummaryKind::MeshGeometryQuality,
                                                CertificationClass::Certified),
              CertificationClass::Certified);
    EXPECT_EQ(ctx.summaryCertificationOrUnknown(AnalysisSummaryKind::DiscreteMatrix,
                                                CertificationClass::Certified),
              CertificationClass::Unknown);

    ctx.clearAnalysisSummaries();

    EXPECT_TRUE(ctx.empty());
    EXPECT_EQ(ctx.analysisSummaries(), nullptr);
    EXPECT_FALSE(ctx.hasSummaryKind(AnalysisSummaryKind::MeshGeometryQuality));
}

// ============================================================================
// ProblemAnalyzer — orchestration
// ============================================================================

namespace {

/// Trivial pass that emits one Nullspace claim
class StubNullspacePass : public AnalyzerPass {
public:
    std::string name() const override { return "StubNullspace"; }
    void run(const ProblemAnalysisContext& /*ctx*/,
             ProblemAnalysisReport& report) const override {
        PropertyClaim c;
        c.kind = PropertyKind::Nullspace;
        c.status = PropertyStatus::Exact;
        c.field = 0;
        c.description = "stub nullspace";
        report.claims.push_back(std::move(c));
    }
};

/// Pass that reads prior claims and emits a compatibility issue if nullspace found
class StubCompatibilityPass : public AnalyzerPass {
public:
    std::string name() const override { return "StubCompatibility"; }
    void run(const ProblemAnalysisContext& /*ctx*/,
             ProblemAnalysisReport& report) const override {
        auto nullspace_claims = report.claimsOfKind(PropertyKind::Nullspace);
        if (!nullspace_claims.empty()) {
            PropertyClaim c;
            c.kind = PropertyKind::CompatibilityCondition;
            c.status = PropertyStatus::Likely;
            c.field = nullspace_claims[0]->field;
            c.description = "solvability requires integral(f)=0";
            report.claims.push_back(std::move(c));
        }
    }
};

} // anonymous namespace

TEST(ProblemAnalyzer, DefaultHasAllPasses) {
    auto analyzer = ProblemAnalyzer::createDefault();
    EXPECT_EQ(analyzer.numPasses(), 29u);
    auto names = analyzer.passNames();
    ASSERT_EQ(names.size(), 29u);
    EXPECT_EQ(names[0], "CouplingGraphAnalyzer");
    EXPECT_EQ(names[1], "KernelAnalyzer");
    EXPECT_EQ(names[2], "MixedOperatorAnalyzer");
    EXPECT_EQ(names[3], "OperatorClassAnalyzer");
    EXPECT_EQ(names[4], "StabilizationAnalyzer");
    EXPECT_EQ(names[5], "ConstraintRankAnalyzer");
    EXPECT_EQ(names[6], "CompatibilityAnalyzer");
    EXPECT_EQ(names[7], "TopologyScopeAnalyzer");
    EXPECT_EQ(names[8], "InterfaceValidationAnalyzer");
    EXPECT_EQ(names[9], "InfSupAnalyzer");
    EXPECT_EQ(names[10], "TransportCharacterAnalyzer");
    EXPECT_EQ(names[11], "ConservationAnalyzer");
    EXPECT_EQ(names[12], "DAEStructureAnalyzer");
    EXPECT_EQ(names[13], "SpaceCompatibilityAnalyzer");
    EXPECT_EQ(names[14], "DiscreteMonotonicityAnalyzer");
    EXPECT_EQ(names[15], "MeshGeometryAnalyzer");
    EXPECT_EQ(names[16], "TemporalStabilityAnalyzer");
    EXPECT_EQ(names[17], "EnergyEntropyLawAnalyzer");
    EXPECT_EQ(names[18], "CoefficientConstitutiveAnalyzer");
    EXPECT_EQ(names[19], "NonlinearTangentAnalyzer");
    EXPECT_EQ(names[20], "LockingRiskAnalyzer");
    EXPECT_EQ(names[21], "SpectralSpuriousModeAnalyzer");
    EXPECT_EQ(names[22], "ErrorEstimatorAnalyzer");
    EXPECT_EQ(names[23], "QuadratureAdequacyAnalyzer");
    EXPECT_EQ(names[24], "MinimumResidualStabilityAnalyzer");
    EXPECT_EQ(names[25], "PreservationStructureAnalyzer");
    EXPECT_EQ(names[26], "CoupledSystemStabilityAnalyzer");
    EXPECT_EQ(names[27], "SolverCompatibilityAnalyzer");
    EXPECT_EQ(names[28], "NumericSummaryPlanner");
}

// ============================================================================
// Phase 21 — Advanced claim types and structured outputs
// ============================================================================

TEST(ProblemAnalysisTypes, ToString_Phase21_PropertyKinds) {
    EXPECT_STREQ(toString(PropertyKind::InfSupCondition), "InfSupCondition");
    EXPECT_STREQ(toString(PropertyKind::ConservationStructure), "ConservationStructure");
    EXPECT_STREQ(toString(PropertyKind::DifferentialAlgebraicStructure), "DifferentialAlgebraicStructure");
    EXPECT_STREQ(toString(PropertyKind::SpaceCompatibility), "SpaceCompatibility");
    EXPECT_STREQ(toString(PropertyKind::OperatorTransportCharacter), "OperatorTransportCharacter");
}

TEST(ProblemAnalysisTypes, ToString_Phase21_ClassificationEnums) {
    EXPECT_STREQ(toString(InfSupClass::Required), "Required");
    EXPECT_STREQ(toString(InfSupClass::StructurallySupported), "StructurallySupported");
    EXPECT_STREQ(toString(InfSupClass::NumericallySupported), "NumericallySupported");
    EXPECT_STREQ(toString(InfSupClass::StabilizedSurrogate), "StabilizedSurrogate");
    EXPECT_STREQ(toString(InfSupClass::LikelyViolated), "LikelyViolated");

    EXPECT_STREQ(toString(ConservationClass::LocalClosureExpected), "LocalClosureExpected");
    EXPECT_STREQ(toString(ConservationClass::ExchangeBalanced), "ExchangeBalanced");
    EXPECT_STREQ(toString(ConservationClass::ClosureBroken), "ClosureBroken");

    EXPECT_STREQ(toString(DAEClass::PureODELike), "PureODELike");
    EXPECT_STREQ(toString(DAEClass::Index1DAELike), "Index1DAELike");
    EXPECT_STREQ(toString(DAEClass::HigherIndexRisk), "HigherIndexRisk");

    EXPECT_STREQ(toString(SpaceCompatibilityClass::Compatible), "Compatible");
    EXPECT_STREQ(toString(SpaceCompatibilityClass::Incompatible), "Incompatible");

    EXPECT_STREQ(toString(TransportCharacterClass::DiffusionLike), "DiffusionLike");
    EXPECT_STREQ(toString(TransportCharacterClass::TransportDominatedRisk), "TransportDominatedRisk");

    EXPECT_STREQ(toString(TemporalStateKind::Dynamic), "Dynamic");
    EXPECT_STREQ(toString(TemporalStateKind::Algebraic), "Algebraic");

    EXPECT_STREQ(toString(SpaceFamily::H1), "H1");
    EXPECT_STREQ(toString(SpaceFamily::HDiv), "HDiv");
    EXPECT_STREQ(toString(SpaceFamily::HCurl), "HCurl");
    EXPECT_STREQ(toString(SpaceFamily::L2), "L2");
}

// ============================================================================
// Phase 1 roadmap vocabulary and structured report plumbing
// ============================================================================

TEST(ProblemAnalysisTypes, ToString_Phase1_RoadmapPropertyKinds) {
    EXPECT_STREQ(toString(PropertyKind::DiscreteMaximumPrinciple), "DiscreteMaximumPrinciple");
    EXPECT_STREQ(toString(PropertyKind::ZMatrixStructure), "ZMatrixStructure");
    EXPECT_STREQ(toString(PropertyKind::MMatrixStructure), "MMatrixStructure");
    EXPECT_STREQ(toString(PropertyKind::MatrixMonotonicityRisk), "MatrixMonotonicityRisk");
    EXPECT_STREQ(toString(PropertyKind::CompatibleComplexStructure), "CompatibleComplexStructure");
    EXPECT_STREQ(toString(PropertyKind::EnergyStability), "EnergyStability");
    EXPECT_STREQ(toString(PropertyKind::EntropyStability), "EntropyStability");
    EXPECT_STREQ(toString(PropertyKind::TemporalStability), "TemporalStability");
    EXPECT_STREQ(toString(PropertyKind::WeakBoundaryCoercivity), "WeakBoundaryCoercivity");
    EXPECT_STREQ(toString(PropertyKind::MeshGeometryValidity), "MeshGeometryValidity");
    EXPECT_STREQ(toString(PropertyKind::CoefficientPositivity), "CoefficientPositivity");
    EXPECT_STREQ(toString(PropertyKind::NonlinearTangentStructure), "NonlinearTangentStructure");
    EXPECT_STREQ(toString(PropertyKind::LockingRisk), "LockingRisk");
    EXPECT_STREQ(toString(PropertyKind::SpectralCorrectness), "SpectralCorrectness");
    EXPECT_STREQ(toString(PropertyKind::ErrorEstimatorEligibility), "ErrorEstimatorEligibility");
    EXPECT_STREQ(toString(PropertyKind::SolverCompatibility), "SolverCompatibility");
    EXPECT_STREQ(toString(PropertyKind::QuadratureAdequacy), "QuadratureAdequacy");
    EXPECT_STREQ(toString(PropertyKind::BoundaryComplementingCondition), "BoundaryComplementingCondition");
    EXPECT_STREQ(toString(PropertyKind::IndefiniteOperatorResolution), "IndefiniteOperatorResolution");
    EXPECT_STREQ(toString(PropertyKind::MinimumResidualStability), "MinimumResidualStability");
    EXPECT_STREQ(toString(PropertyKind::InvariantDomainPreservation), "InvariantDomainPreservation");
    EXPECT_STREQ(toString(PropertyKind::EquilibriumPreservation), "EquilibriumPreservation");
    EXPECT_STREQ(toString(PropertyKind::GeometricConservation), "GeometricConservation");
    EXPECT_STREQ(toString(PropertyKind::TransferOperatorCompatibility), "TransferOperatorCompatibility");
    EXPECT_STREQ(toString(PropertyKind::AdjointConsistency), "AdjointConsistency");
    EXPECT_STREQ(toString(PropertyKind::ParameterRobustness), "ParameterRobustness");
    EXPECT_STREQ(toString(PropertyKind::InitialDataCompatibility), "InitialDataCompatibility");
}

TEST(ProblemAnalysisTypes, ToString_Phase1_ClassificationEnums) {
    EXPECT_STREQ(toString(ApplicabilityClass::Applicable), "Applicable");
    EXPECT_STREQ(toString(ApplicabilityClass::NotApplicable), "NotApplicable");
    EXPECT_STREQ(toString(ApplicabilityClass::Unknown), "Unknown");

    EXPECT_STREQ(toString(CertificationClass::Certified), "Certified");
    EXPECT_STREQ(toString(CertificationClass::Violated), "Violated");
    EXPECT_STREQ(toString(CertificationClass::NotCertified), "NotCertified");
    EXPECT_STREQ(toString(CertificationClass::Unknown), "Unknown");

    EXPECT_STREQ(toString(MatrixSignStructureClass::ZMatrix), "ZMatrix");
    EXPECT_STREQ(toString(MatrixSignStructureClass::NotZMatrix), "NotZMatrix");
    EXPECT_STREQ(toString(MatrixSignStructureClass::MMatrixCertified), "MMatrixCertified");
    EXPECT_STREQ(toString(MatrixSignStructureClass::MMatrixNotCertified), "MMatrixNotCertified");
    EXPECT_STREQ(toString(MatrixSignStructureClass::Unknown), "Unknown");

    EXPECT_STREQ(toString(OperatorSymmetryClass::Symmetric), "Symmetric");
    EXPECT_STREQ(toString(OperatorSymmetryClass::Skew), "Skew");
    EXPECT_STREQ(toString(OperatorSymmetryClass::Nonsymmetric), "Nonsymmetric");
    EXPECT_STREQ(toString(OperatorSymmetryClass::Unknown), "Unknown");

    EXPECT_STREQ(toString(TemporalStabilityClass::AStable), "AStable");
    EXPECT_STREQ(toString(TemporalStabilityClass::LStable), "LStable");
    EXPECT_STREQ(toString(TemporalStabilityClass::BStable), "BStable");
    EXPECT_STREQ(toString(TemporalStabilityClass::SSP), "SSP");
    EXPECT_STREQ(toString(TemporalStabilityClass::ConditionallyStable), "ConditionallyStable");
    EXPECT_STREQ(toString(TemporalStabilityClass::Unknown), "Unknown");

    EXPECT_STREQ(toString(CoercivityClass::Coercive), "Coercive");
    EXPECT_STREQ(toString(CoercivityClass::Semicoercive), "Semicoercive");
    EXPECT_STREQ(toString(CoercivityClass::Indefinite), "Indefinite");
    EXPECT_STREQ(toString(CoercivityClass::NotCoercive), "NotCoercive");
    EXPECT_STREQ(toString(CoercivityClass::Unknown), "Unknown");

    EXPECT_STREQ(toString(NullspaceHandlingClass::NotApplicable), "NotApplicable");
    EXPECT_STREQ(toString(NullspaceHandlingClass::AnchoredByConstraints), "AnchoredByConstraints");
    EXPECT_STREQ(toString(NullspaceHandlingClass::ProjectedOut), "ProjectedOut");
    EXPECT_STREQ(toString(NullspaceHandlingClass::Retained), "Retained");
    EXPECT_STREQ(toString(NullspaceHandlingClass::Uncontrolled), "Uncontrolled");
    EXPECT_STREQ(toString(NullspaceHandlingClass::Unknown), "Unknown");
}

TEST(PropertyClaim, Phase1_StructuredRoadmapFields) {
    PropertyClaim monotonicity;
    monotonicity.kind = PropertyKind::ZMatrixStructure;
    monotonicity.applicability_class = ApplicabilityClass::Applicable;
    monotonicity.certification_class = CertificationClass::Certified;
    monotonicity.matrix_sign_structure_class = MatrixSignStructureClass::ZMatrix;
    monotonicity.coercivity_class = CoercivityClass::Semicoercive;
    monotonicity.reduced_definiteness_class = CertificationClass::Certified;
    monotonicity.nullspace_handling_class = NullspaceHandlingClass::AnchoredByConstraints;
    monotonicity.tested_block_id = "u:u:cell";

    EXPECT_EQ(*monotonicity.applicability_class, ApplicabilityClass::Applicable);
    EXPECT_EQ(*monotonicity.certification_class, CertificationClass::Certified);
    EXPECT_EQ(*monotonicity.matrix_sign_structure_class, MatrixSignStructureClass::ZMatrix);
    EXPECT_EQ(*monotonicity.coercivity_class, CoercivityClass::Semicoercive);
    EXPECT_EQ(*monotonicity.reduced_definiteness_class, CertificationClass::Certified);
    EXPECT_EQ(*monotonicity.nullspace_handling_class, NullspaceHandlingClass::AnchoredByConstraints);
    EXPECT_EQ(*monotonicity.tested_block_id, "u:u:cell");

    PropertyClaim transport;
    transport.kind = PropertyKind::OperatorTransportCharacter;
    transport.peclet_number = 42.0;
    transport.cfl_number = 0.75;
    transport.nonnormality_indicator = 3.0;
    transport.invariant_domain_metadata_present = true;
    transport.well_balanced_metadata_present = false;

    EXPECT_DOUBLE_EQ(*transport.peclet_number, 42.0);
    EXPECT_DOUBLE_EQ(*transport.cfl_number, 0.75);
    EXPECT_TRUE(*transport.invariant_domain_metadata_present);
    EXPECT_FALSE(*transport.well_balanced_metadata_present);
}

TEST(ProblemAnalysisReport, Phase1_PrintIncludesNewKindsAndStructuredFields) {
    ProblemAnalysisReport report;

    PropertyClaim dmp;
    dmp.kind = PropertyKind::DiscreteMaximumPrinciple;
    dmp.status = PropertyStatus::Unknown;
    dmp.applicability_class = ApplicabilityClass::Applicable;
    dmp.certification_class = CertificationClass::Unknown;
    dmp.description = "DMP check awaits numeric summaries";
    report.claims.push_back(dmp);

    PropertyClaim z;
    z.kind = PropertyKind::ZMatrixStructure;
    z.status = PropertyStatus::Exact;
    z.matrix_sign_structure_class = MatrixSignStructureClass::ZMatrix;
    z.tested_block_id = "pressure:pressure:cell";
    report.claims.push_back(z);

    PropertyClaim time;
    time.kind = PropertyKind::TemporalStability;
    time.status = PropertyStatus::Likely;
    time.temporal_stability_class = TemporalStabilityClass::AStable;
    report.claims.push_back(time);

    std::ostringstream oss;
    report.print(oss);
    const auto output = oss.str();

    EXPECT_NE(output.find("--- DiscreteMaximumPrinciple ---"), std::string::npos);
    EXPECT_NE(output.find("--- ZMatrixStructure ---"), std::string::npos);
    EXPECT_NE(output.find("--- TemporalStability ---"), std::string::npos);
    EXPECT_NE(output.find("applicability=Applicable"), std::string::npos);
    EXPECT_NE(output.find("certification=Unknown"), std::string::npos);
    EXPECT_NE(output.find("matrix_sign=ZMatrix"), std::string::npos);
    EXPECT_NE(output.find("tested_block=pressure:pressure:cell"), std::string::npos);
    EXPECT_NE(output.find("temporal=AStable"), std::string::npos);
}

// ============================================================================
// Phase 2 - Common discrete summary metadata contracts
// ============================================================================

TEST(AnalysisSummaryTypes, CommonSummaryContractsStoreRequiredFields) {
    AnalysisSummarySet summaries;

    CoefficientPropertySummary coeff;
    coeff.coefficient = "kappa";
    coeff.tensor_rank = TensorRank::Rank2Tensor;
    coeff.symmetry = SymmetryClass::Symmetric;
    coeff.positivity = PositivityClass::Positive;
    coeff.anisotropy_ratio = 4.0;
    coeff.contrast_ratio = 10.0;
    coeff.coefficient_region_coverage_complete = true;
    coeff.quadrature_point_coverage_complete = true;
    coeff.lower_bound_valid_for_all_samples = true;
    coeff.tolerance_metadata_present = true;
    summaries.coefficient_properties.push_back(coeff);

    DiscreteMatrixSummary matrix;
    matrix.backend_kind = svmp::FE::backends::BackendKind::Eigen;
    matrix.block.test_variables = {VariableKey::field(0)};
    matrix.block.trial_variables = {VariableKey::field(0)};
    matrix.square = true;
    matrix.structurally_symmetric = true;
    matrix.sign_tolerance = 1.0e-14;
    matrix.row_sum_tolerance = 1.0e-12;
    matrix.symmetry_tolerance = 1.0e-13;
    matrix.diagonal_count = 3;
    matrix.offdiag_count = 6;
    matrix.positive_offdiag_count = 0;
    matrix.min_row_sum = -1.0e-15;
    matrix.max_row_sum = 1.0e-15;
    matrix.condition_estimate = 12.0;
    summaries.discrete_matrices.push_back(matrix);

    ReducedMatrixSummary reduced;
    reduced.reduction_kind = ConstraintReductionKind::StrongDirichletElimination;
    reduced.free_dof_count = 2;
    reduced.constrained_dof_count = 1;
    reduced.affine_terms_accounted_for = true;
    summaries.reduced_matrices.push_back(reduced);

    SchurComplementSummary schur;
    schur.schur_id = "schur";
    schur.schur_available = true;
    schur.reduction_exact_for_analysis = true;
    schur.primal_block_invertible_evidence_present = true;
    schur.inf_sup_evidence_present = true;
    schur.nullspace_handling_evidence_present = true;
    schur.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    schur.schur_definiteness_evidence_present = true;
    schur.schur_positivity = PositivityClass::Positive;
    schur.spectral_equivalence_bounds_present = true;
    schur.preconditioner_equivalence_bounds_present = true;
    summaries.schur_complements.push_back(schur);

    LocalStencilSummary stencil;
    stencil.element = 7;
    stencil.positive_offdiag_count = 0;
    stencil.negative_offdiag_count = 2;
    summaries.local_stencils.push_back(stencil);

    MeshGeometryQualitySummary mesh;
    mesh.mesh_revision = 3;
    mesh.min_jacobian = 0.25;
    mesh.max_jacobian = 2.0;
    mesh.min_angle = 45.0;
    mesh.max_aspect_ratio = 1.5;
    mesh.min_cut_cell_fraction = 0.1;
    mesh.worst_elements.push_back(4);
    summaries.mesh_geometry_quality.push_back(mesh);

    FluxBalanceSummary flux;
    flux.local_residual_norm = 1.0e-12;
    flux.global_residual_norm = 2.0e-12;
    flux.interface_pair_residual_norm = 3.0e-12;
    summaries.flux_balances.push_back(flux);

    TemporalStabilitySummary temporal;
    temporal.time_scheme = "generalized-alpha";
    temporal.stability_class = TemporalStabilityClass::AStable;
    temporal.cfl_estimate = 0.5;
    summaries.temporal_stability.push_back(temporal);

    BoundarySymbolSummary boundary;
    boundary.principal_operator_order = 2;
    boundary.boundary_operator_order = 1;
    boundary.trace_coverage = TraceCapabilityFlags::Value;
    boundary.complementing_condition_satisfied = true;
    boundary.boundary_condition_count = 1;
    boundary.required_boundary_condition_count = 1;
    boundary.principal_symbol_rank_evidence_present = true;
    boundary.boundary_symbol_rank_evidence_present = true;
    boundary.component_coverage_complete = true;
    boundary.dof_coverage_complete = true;
    summaries.boundary_symbols.push_back(boundary);

    InfSupEstimateSummary infsup;
    infsup.primal_variable = VariableKey::field(0);
    infsup.multiplier_variable = VariableKey::field(1);
    infsup.estimate_value = 0.2;
    infsup.estimate_scope = "unit-test";
    infsup.nullspace_handling = NullspaceHandlingClass::ProjectedOut;
    summaries.inf_sup_estimates.push_back(infsup);

    InfSupPairCertificationSummary infsup_pair;
    infsup_pair.primal_variable = VariableKey::field(0);
    infsup_pair.multiplier_variable = VariableKey::field(1);
    infsup_pair.pair_family = "Taylor-Hood";
    infsup_pair.known_stable_pair = true;
    infsup_pair.mesh_assumption_evidence_present = true;
    infsup_pair.domain_assumption_evidence_present = true;
    summaries.inf_sup_pair_certifications.push_back(infsup_pair);

    EnergyEntropySummary energy;
    energy.energy_entropy_id = "E";
    energy.expected_production_sign = BalanceSignClass::Nonpositive;
    energy.observed_discrete_balance = -1.0e-8;
    summaries.energy_entropy.push_back(energy);

    InvariantDomainSummary invariant;
    invariant.invariant_set_id = "positive";
    invariant.variables.push_back(VariableKey::field(0));
    invariant.lower_bound = 0.0;
    invariant.lower_bound_active = true;
    invariant.limiter_evidence_present = true;
    summaries.invariant_domains.push_back(invariant);

    EquilibriumPreservationSummary equilibrium;
    equilibrium.equilibrium_id = "rest";
    equilibrium.flux_source_residual = 0.0;
    equilibrium.source_quadrature_metadata_present = true;
    summaries.equilibrium_preservation.push_back(equilibrium);

    MovingDomainSummary moving;
    moving.mesh_revision = 4;
    moving.mesh_velocity_metadata_present = true;
    moving.geometric_conservation_residual = 1.0e-13;
    summaries.moving_domain.push_back(moving);

    TransferOperatorSummary transfer;
    transfer.interface_pair_id = "left-right";
    transfer.projection_space_id = "mortar-p1";
    transfer.conservation_residual = 1.0e-12;
    transfer.constant_preservation_residual = 0.0;
    summaries.transfer_operators.push_back(transfer);

    AdjointConsistencySummary adjoint;
    adjoint.contribution_id = "nitsche-face";
    adjoint.adjoint_consistency = AdjointConsistencyKind::Yes;
    adjoint.transpose_backend_support = true;
    adjoint.goal_functional_id = "drag";
    summaries.adjoint_consistency.push_back(adjoint);

    ParameterScaleSummary parameter;
    parameter.nondimensional_parameter_id = "Pe";
    parameter.min_scale_value = 0.1;
    parameter.max_scale_value = 100.0;
    parameter.layer_resolution_metric = 0.25;
    summaries.parameter_scales.push_back(parameter);

    StabilizationAdequacySummary stabilization;
    stabilization.stabilization_id = "supg";
    stabilization.method_family = "SUPG";
    stabilization.parameter_formula_metadata_present = true;
    stabilization.residual_consistency_evidence_present = true;
    stabilization.regime_metadata_present = true;
    stabilization.peclet_condition_satisfied = true;
    stabilization.cfl_condition_satisfied = true;
    summaries.stabilization_adequacy.push_back(stabilization);

    InitialCompatibilitySummary initial;
    initial.initial_constraint_residual = 1.0e-14;
    initial.initial_boundary_residual = 2.0e-14;
    initial.invariant_domain_initial_violation_count = 0;
    summaries.initial_compatibility.push_back(initial);

    DAEStructureEvidenceSummary dae;
    dae.system_id = "semi-explicit";
    dae.variables = {VariableKey::field(0), VariableKey::field(1)};
    dae.dae_form_class = DAEFormClass::SemiExplicit;
    dae.mass_matrix_rank_metadata_present = true;
    dae.algebraic_jacobian_rank_metadata_present = true;
    dae.algebraic_jacobian_full_rank = true;
    dae.hidden_constraint_metadata_present = true;
    dae.consistent_initial_condition_evidence_present = true;
    summaries.dae_structure_evidence.push_back(dae);

    CompatibleComplexSummary complex;
    complex.complex_id = "generic-sequence";
    complex.exact_sequence_compatible = true;
    complex.commuting_projection_available = true;
    complex.trace_sequence_compatible = true;
    summaries.compatible_complexes.push_back(complex);

    NonlinearTangentSummary tangent;
    tangent.residual_id = "nonlinear-residual";
    tangent.tangent_consistency = TangentConsistencyClass::Exact;
    tangent.jacobian_action_available = true;
    tangent.jacobian_nonsingularity_evidence_present = true;
    summaries.nonlinear_tangents.push_back(tangent);

    SpectralStructureSummary spectral;
    spectral.eigenproblem_declared = true;
    spectral.self_adjoint_evidence = true;
    spectral.compactness_evidence = true;
    summaries.spectral_structures.push_back(spectral);

    ErrorEstimatorSummary estimator;
    estimator.estimator_id = "residual-estimator";
    estimator.residual_metadata_present = true;
    estimator.jump_metadata_present = true;
    estimator.norm_metadata_present = true;
    estimator.pde_operator_class_metadata_present = true;
    estimator.boundary_residual_metadata_present = true;
    estimator.data_oscillation_metadata_present = true;
    estimator.coefficient_source_regularity_metadata_present = true;
    estimator.shape_regular_mesh_evidence_present = true;
    estimator.reliability_constant_metadata_present = true;
    estimator.efficiency_constant_metadata_present = true;
    estimator.effectivity_bounds_present = true;
    estimator.refinement_evidence_present = true;
    summaries.error_estimators.push_back(estimator);

    QuadratureAdequacySummary quadrature;
    quadrature.integrand_polynomial_degree = 2;
    quadrature.quadrature_exact_degree = 3;
    summaries.quadrature_adequacy.push_back(quadrature);

    CoupledSystemStabilitySummary coupled;
    coupled.coupling_group = "generic-coupling";
    coupled.monolithic_coupling = true;
    summaries.coupled_system_stability.push_back(coupled);

    MinimumResidualStabilitySummary minres;
    minres.method_id = "dpg";
    minres.method_class = MinimumResidualMethodClass::DPG;
    minres.trial_space_metadata_present = true;
    minres.test_space_metadata_present = true;
    minres.residual_norm_metadata_present = true;
    minres.test_norm_metadata_present = true;
    minres.riesz_map_metadata_present = true;
    minres.fortin_operator_evidence_present = true;
    minres.enrichment_sufficiency_evidence_present = true;
    minres.residual_control_constant_present = true;
    minres.local_trial_to_test_conditioning_present = true;
    minres.normal_equation_conditioning_present = true;
    summaries.minimum_residual_stability.push_back(minres);

    EXPECT_EQ(summaries.totalSummaryCount(), 28u);
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::CoefficientProperties));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::DiscreteMatrix));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::ReducedMatrix));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::SchurComplement));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::LocalStencil));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::MeshGeometryQuality));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::FluxBalance));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::TemporalStability));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::BoundarySymbol));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::InfSupEstimate));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::InfSupPairCertification));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::EnergyEntropyBalance));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::InvariantDomain));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::EquilibriumPreservation));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::MovingDomain));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::TransferOperator));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::AdjointConsistency));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::ParameterScale));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::StabilizationAdequacy));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::InitialCompatibility));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::DAEStructureEvidence));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::CompatibleComplex));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::NonlinearTangent));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::SpectralStructure));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::ErrorEstimator));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::QuadratureAdequacy));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::CoupledSystemStability));
    EXPECT_TRUE(summaries.has(AnalysisSummaryKind::MinimumResidualStability));

    EXPECT_EQ(summaries.coefficient_properties[0].tensor_rank, TensorRank::Rank2Tensor);
    EXPECT_EQ(summaries.coefficient_properties[0].symmetry, SymmetryClass::Symmetric);
    EXPECT_EQ(summaries.coefficient_properties[0].positivity, PositivityClass::Positive);
    EXPECT_DOUBLE_EQ(summaries.discrete_matrices[0].row_sum_tolerance, 1.0e-12);
    EXPECT_DOUBLE_EQ(*summaries.discrete_matrices[0].condition_estimate, 12.0);
    EXPECT_EQ(summaries.reduced_matrices[0].reduction_kind,
              ConstraintReductionKind::StrongDirichletElimination);
    EXPECT_TRUE(*summaries.boundary_symbols[0].complementing_condition_satisfied);
    EXPECT_EQ(summaries.inf_sup_estimates[0].nullspace_handling,
              NullspaceHandlingClass::ProjectedOut);
}

TEST(AnalysisSummaryTypes, WorstSamplesAreBoundedAndDeterministic) {
    DiscreteMatrixSummary matrix;
    matrix.worst_entry_sample_limit = 2;
    matrix.addWorstEntry(MatrixEntrySample{3, 4, 1.0, 1, 2, "mild"});
    matrix.addWorstEntry(MatrixEntrySample{2, 5, -3.0, 0, 1, "worst"});
    matrix.addWorstEntry(MatrixEntrySample{1, 6, 2.0, 0, 3, "middle"});

    ASSERT_EQ(matrix.worst_entries.size(), 2u);
    EXPECT_EQ(matrix.worst_entries[0].row, 2);
    EXPECT_EQ(matrix.worst_entries[0].col, 5);
    EXPECT_DOUBLE_EQ(matrix.worst_entries[0].value, -3.0);
    EXPECT_EQ(matrix.worst_entries[1].row, 1);
    EXPECT_EQ(matrix.worst_entries[1].col, 6);

    LocalStencilSummary stencil;
    stencil.worst_entry_sample_limit = 1;
    stencil.addWorstLocalEntry(MatrixEntrySample{3, 3, 4.0, 1, 2, "rank1"});
    stencil.addWorstLocalEntry(MatrixEntrySample{2, 3, 4.0, 0, 1, "rank0"});

    ASSERT_EQ(stencil.worst_local_entries.size(), 1u);
    EXPECT_EQ(stencil.worst_local_entries[0].owning_rank, 0);
    EXPECT_EQ(stencil.worst_local_entries[0].row, 2);
}

TEST(AnalysisSummaryTypes, SummaryObjectsAreValueTypes) {
    EXPECT_TRUE(std::is_copy_constructible_v<AnalysisSummarySet>);
    EXPECT_TRUE(std::is_move_constructible_v<AnalysisSummarySet>);
    EXPECT_TRUE(std::is_copy_constructible_v<DiscreteMatrixSummary>);
    EXPECT_TRUE(std::is_move_constructible_v<MeshGeometryQualitySummary>);
}

TEST(ProblemAnalysisTypes, TraceCapabilityFlags_Bitmask) {
    auto flags = TraceCapabilityFlags::Value | TraceCapabilityFlags::NormalFlux;
    EXPECT_TRUE(hasTraceFlag(flags, TraceCapabilityFlags::Value));
    EXPECT_TRUE(hasTraceFlag(flags, TraceCapabilityFlags::NormalFlux));
    EXPECT_FALSE(hasTraceFlag(flags, TraceCapabilityFlags::Jump));
    EXPECT_FALSE(hasTraceFlag(TraceCapabilityFlags::None, TraceCapabilityFlags::Value));
}

TEST(PropertyClaim, Phase21_StructuredOutputs) {
    PropertyClaim claim;
    claim.kind = PropertyKind::InfSupCondition;
    claim.inf_sup_class = InfSupClass::Required;
    claim.claim_origin = "InfSupAnalyzer";

    EXPECT_TRUE(claim.inf_sup_class.has_value());
    EXPECT_EQ(*claim.inf_sup_class, InfSupClass::Required);

    PropertyClaim dae;
    dae.kind = PropertyKind::DifferentialAlgebraicStructure;
    dae.dae_class = DAEClass::Index1DAELike;
    EXPECT_EQ(*dae.dae_class, DAEClass::Index1DAELike);

    PropertyClaim transport;
    transport.kind = PropertyKind::OperatorTransportCharacter;
    transport.transport_character_class = TransportCharacterClass::TransportDominatedRisk;
    EXPECT_EQ(*transport.transport_character_class, TransportCharacterClass::TransportDominatedRisk);
}

TEST(VariableDescriptor, Phase21_TemporalMetadata) {
    VariableDescriptor vd;
    vd.key = VariableKey::field(0);
    vd.temporal_state_kind = TemporalStateKind::Dynamic;
    vd.max_time_derivative_order = 1;
    vd.participates_in_constraint_blocks = false;
    vd.participates_in_mass_blocks = true;

    EXPECT_EQ(vd.temporal_state_kind, TemporalStateKind::Dynamic);
    EXPECT_EQ(vd.max_time_derivative_order, 1);
    EXPECT_TRUE(vd.participates_in_mass_blocks);
}

TEST(ProblemAnalyzer, EmptyAnalyzerReturnsEmptyReport) {
    ProblemAnalyzer analyzer;
    ProblemAnalysisContext ctx;
    auto report = analyzer.analyze(ctx);

    EXPECT_TRUE(report.claims.empty());
    EXPECT_TRUE(report.issues.empty());
}

TEST(ProblemAnalyzer, SinglePass) {
    ProblemAnalyzer analyzer;
    analyzer.addPass(std::make_unique<StubNullspacePass>());

    EXPECT_EQ(analyzer.numPasses(), 1u);
    EXPECT_EQ(analyzer.passNames()[0], "StubNullspace");

    ProblemAnalysisContext ctx;
    auto report = analyzer.analyze(ctx);

    ASSERT_EQ(report.claims.size(), 1u);
    EXPECT_EQ(report.claims[0].kind, PropertyKind::Nullspace);
    EXPECT_EQ(report.claims[0].status, PropertyStatus::Exact);
}

TEST(ProblemAnalyzer, PassOrdering_LaterPassReadsEarlierClaims) {
    ProblemAnalyzer analyzer;
    analyzer.addPass(std::make_unique<StubNullspacePass>());
    analyzer.addPass(std::make_unique<StubCompatibilityPass>());

    ProblemAnalysisContext ctx;
    auto report = analyzer.analyze(ctx);

    ASSERT_EQ(report.claims.size(), 2u);
    EXPECT_EQ(report.claims[0].kind, PropertyKind::Nullspace);
    EXPECT_EQ(report.claims[1].kind, PropertyKind::CompatibilityCondition);
    EXPECT_EQ(report.claims[1].field, 0);
    EXPECT_NE(report.claims[1].description.find("integral"), std::string::npos);
}

TEST(ProblemAnalyzer, NullPassIgnored) {
    ProblemAnalyzer analyzer;
    analyzer.addPass(nullptr);
    EXPECT_EQ(analyzer.numPasses(), 0u);
}

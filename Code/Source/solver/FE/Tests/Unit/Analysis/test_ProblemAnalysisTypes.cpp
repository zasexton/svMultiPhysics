/**
 * @file test_ProblemAnalysisTypes.cpp
 * @brief Unit tests for core analysis types, report queries, and output formatting
 */

#include <gtest/gtest.h>

#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"
#include <sstream>
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
    EXPECT_EQ(analyzer.numPasses(), 14u);
    auto names = analyzer.passNames();
    ASSERT_EQ(names.size(), 14u);
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

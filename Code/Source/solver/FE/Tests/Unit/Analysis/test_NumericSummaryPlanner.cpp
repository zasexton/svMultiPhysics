/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NumericSummaryPlanner.cpp
 * @brief Unit tests for Phase 3 symbolic-to-numeric summary request planning.
 */

#include <gtest/gtest.h>

#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/NumericSummaryPlanner.h"
#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include <sstream>
#include <utility>

using namespace svmp::FE;
using namespace svmp::FE::analysis;

namespace {

FieldDescriptor scalarField(FieldId id, std::string name = "scalar") {
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.field_type = FieldType::Scalar;
    fd.value_dimension = 1;
    fd.space_family = SpaceFamily::H1;
    fd.polynomial_order = 1;
    fd.trace_capabilities = TraceCapabilityFlags::Value | TraceCapabilityFlags::NormalFlux;
    return fd;
}

FieldDescriptor vectorField(FieldId id, std::string name = "vector") {
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.field_type = FieldType::Vector;
    fd.value_dimension = 3;
    fd.space_family = SpaceFamily::H1;
    fd.polynomial_order = 2;
    fd.trace_capabilities = TraceCapabilityFlags::Value | TraceCapabilityFlags::NormalFlux;
    return fd;
}

const AnalysisSummaryRequest* firstRequest(const AnalysisRequestPlan& plan,
                                           AnalysisSummaryKind kind) {
    auto requests = plan.requestsOfKind(kind);
    return requests.empty() ? nullptr : requests.front();
}

bool requestReasonsContain(const AnalysisRequestPlan& plan,
                           AnalysisSummaryKind kind,
                           const std::string& needle) {
    for (const auto* request : plan.requestsOfKind(kind)) {
        for (const auto& reason : request->reasons) {
            if (reason.find(needle) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

PropertyClaim claimFrom(const char* analyzer, PropertyKind kind) {
    PropertyClaim claim;
    claim.kind = kind;
    claim.status = PropertyStatus::Likely;
    claim.confidence = AnalysisConfidence::Medium;
    claim.claim_origin = analyzer;
    claim.variables.push_back(VariableKey::field(0));
    claim.addEvidence(analyzer, "synthetic symbolic analyzer evidence");
    return claim;
}

} // namespace

TEST(NumericSummaryPlanner, ReusesNamedSymbolicAnalyzerClaims) {
    ProblemAnalysisContext ctx;
    ProblemAnalysisReport report;

    report.claims.push_back(claimFrom("OperatorClassAnalyzer", PropertyKind::OperatorDefiniteness));
    report.claims.push_back(claimFrom("KernelAnalyzer", PropertyKind::Nullspace));
    report.claims.push_back(claimFrom("MixedOperatorAnalyzer", PropertyKind::MixedSaddlePoint));
    report.claims.push_back(claimFrom("InfSupAnalyzer", PropertyKind::InfSupCondition));
    report.claims.push_back(claimFrom("TransportCharacterAnalyzer", PropertyKind::OperatorTransportCharacter));
    report.claims.push_back(claimFrom("ConservationAnalyzer", PropertyKind::ConservationStructure));
    report.claims.push_back(claimFrom("DAEStructureAnalyzer", PropertyKind::DifferentialAlgebraicStructure));
    report.claims.push_back(claimFrom("SpaceCompatibilityAnalyzer", PropertyKind::SpaceCompatibility));
    report.claims.push_back(claimFrom("ConstraintRankAnalyzer", PropertyKind::UnderConstraint));
    report.claims.push_back(claimFrom("TopologyScopeAnalyzer", PropertyKind::TopologyScopedKernel));

    NumericSummaryPlanner planner;
    planner.run(ctx, report);

    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("OperatorClassAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("KernelAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("MixedOperatorAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("InfSupAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("TransportCharacterAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("ConservationAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("DAEStructureAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("SpaceCompatibilityAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("ConstraintRankAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("TopologyScopeAnalyzer"));

    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::DiscreteMatrix));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::ReducedMatrix));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::InfSupEstimate));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::FluxBalance));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::TemporalStability));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::InitialCompatibility));
}

TEST(NumericSummaryPlanner, ScalarDiffusionTriggersEllipticAndMonotonicityRequests) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(scalarField(0, "u"));
    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "diffusion", "unit-test"));

    auto report = analyzer.analyze(ctx);

    EXPECT_GE(report.countByKind(PropertyKind::OperatorDefiniteness), 1u);
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::DiscreteMatrix));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::ReducedMatrix));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::CoefficientProperties));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::MeshGeometryQuality));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::LocalStencil));
    EXPECT_TRUE(requestReasonsContain(report.request_plan,
                                      AnalysisSummaryKind::LocalStencil,
                                      "DMP/Z-matrix/M-matrix monotonicity"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("OperatorClassAnalyzer"));
}

TEST(NumericSummaryPlanner, MixedBlockSystemTriggersInfSupAndSaddlePointRequests) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(vectorField(0, "velocity"));
    ctx.addFieldDescriptor(scalarField(1, "pressure"));

    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "momentum", "unit-test"));
    ctx.addContribution(ContributionDescriptor::constraintPairDesc(
        VariableKey::field(0), VariableKey::field(1),
        "velocity-pressure", "continuity", "unit-test"));

    auto report = analyzer.analyze(ctx);

    EXPECT_GE(report.countByKind(PropertyKind::MixedSaddlePoint), 1u);
    EXPECT_GE(report.countByKind(PropertyKind::InfSupCondition), 1u);
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::InfSupEstimate));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::ReducedMatrix));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::DiscreteMatrix));
    EXPECT_TRUE(requestReasonsContain(report.request_plan,
                                      AnalysisSummaryKind::InfSupEstimate,
                                      "saddle-point"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("MixedOperatorAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("InfSupAnalyzer"));
}

TEST(NumericSummaryPlanner, StabilizedSurrogateInfSupDoesNotRequestStablePairCertification) {
    ProblemAnalysisContext ctx;
    ProblemAnalysisReport report;

    auto mixed = claimFrom("MixedOperatorAnalyzer", PropertyKind::MixedSaddlePoint);
    mixed.variables = {VariableKey::field(0), VariableKey::field(1)};
    report.claims.push_back(std::move(mixed));

    auto surrogate = claimFrom("InfSupAnalyzer", PropertyKind::InfSupCondition);
    surrogate.variables = {VariableKey::field(0), VariableKey::field(1)};
    surrogate.inf_sup_class = InfSupClass::StabilizedSurrogate;
    report.claims.push_back(std::move(surrogate));

    NumericSummaryPlanner planner;
    planner.run(ctx, report);

    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::InfSupEstimate));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::SchurComplement));
    EXPECT_FALSE(report.request_plan.has(AnalysisSummaryKind::InfSupPairCertification));
}

TEST(NumericSummaryPlanner, FirstOrderOperatorsTriggerTransportAndStabilizationRequests) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(scalarField(0, "c"));
    ctx.addContribution(ContributionDescriptor::transportLike(
        VariableKey::field(0), "advection", "unit-test"));
    ctx.addContribution(ContributionDescriptor::stabilization(
        VariableKey::field(0), "supg", "unit-test"));

    auto report = analyzer.analyze(ctx);

    EXPECT_GE(report.countByKind(PropertyKind::OperatorTransportCharacter), 1u);
    EXPECT_GE(report.countByKind(PropertyKind::Stabilization), 1u);
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::TemporalStability));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::ParameterScale));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::DiscreteMatrix));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::InvariantDomain));
    EXPECT_TRUE(requestReasonsContain(report.request_plan,
                                      AnalysisSummaryKind::TemporalStability,
                                      "CFL/eigenvalue-scale"));
    EXPECT_TRUE(requestReasonsContain(report.request_plan,
                                      AnalysisSummaryKind::ParameterScale,
                                      "stabilization parameter"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("TransportCharacterAnalyzer"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("StabilizationAnalyzer"));
}

TEST(NumericSummaryPlanner, CellFormulationRequestsCoefficientPropertiesUpFront) {
    ProblemAnalysisContext ctx;
    FormulationRecord record;
    record.operator_tag = "cell-form";
    record.active_fields = {0};
    record.active_variables = {VariableKey::field(0)};
    record.active_domains = {DomainKind::Cell};
    ctx.addFormulationRecord(record);

    ProblemAnalysisReport report;
    NumericSummaryPlanner planner;
    planner.run(ctx, report);

    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::CoefficientProperties));
    EXPECT_TRUE(requestReasonsContain(report.request_plan,
                                      AnalysisSummaryKind::CoefficientProperties,
                                      "before matrix-based operator classification"));
}

TEST(NumericSummaryPlanner, FormulationInvariantDomainsRequestBoundSummaries) {
    ProblemAnalysisContext ctx;
    FormulationRecord record;
    record.operator_tag = "equations";
    record.active_fields = {1};
    record.active_variables = {VariableKey::field(1)};
    record.active_domains = {DomainKind::Cell};

    InvariantDomainDescriptor descriptor;
    descriptor.invariant_set_id = "primitive:field:1:less-than:10";
    descriptor.variables.push_back(VariableKey::field(1));
    descriptor.sampled_field = 1;
    descriptor.upper_bound = Real{10};
    descriptor.source_admissibility_evidence_present = true;
    record.invariant_domain_descriptors.push_back(std::move(descriptor));

    ctx.addFormulationRecord(record);

    ProblemAnalysisReport report;
    NumericSummaryPlanner planner;
    planner.run(ctx, report);

    const auto requests =
        report.request_plan.requestsOfKind(AnalysisSummaryKind::InvariantDomain);
    ASSERT_EQ(requests.size(), 1u);
    ASSERT_NE(requests.front(), nullptr);
    EXPECT_EQ(requests.front()->block_id, "equations");
    EXPECT_EQ(requests.front()->scope_id, "primitive:field:1:less-than:10");
    EXPECT_EQ(requests.front()->variables, std::vector<VariableKey>{VariableKey::field(1)});
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("ProblemAnalysisContext"));
    EXPECT_TRUE(requestReasonsContain(report.request_plan,
                                      AnalysisSummaryKind::InvariantDomain,
                                      "primitive-DAG invariant-domain metadata"));
}

TEST(NumericSummaryPlanner, DGAndInterfaceFormsTriggerPenaltyAndFluxRequests) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(scalarField(0, "u"));

    FormulationRecord dg;
    dg.operator_tag = "dg-interface";
    dg.active_fields = {0};
    dg.active_variables = {VariableKey::field(0)};
    dg.has_interior_face_terms = true;
    dg.active_domains = {DomainKind::InteriorFace, DomainKind::InterfaceFace};
    ctx.addFormulationRecord(dg);

    BoundaryConditionDescriptor nitsche;
    nitsche.primary_variable = VariableKey::field(0);
    nitsche.domain = DomainKind::InterfaceFace;
    nitsche.enforcement_kind = EnforcementKind::WeakNitsche;
    nitsche.trace_kind = TraceKind::Value;
    nitsche.source = "unit-test Nitsche interface";
    nitsche.adjoint_consistency = AdjointConsistencyKind::Yes;
    ctx.addBCDescriptor(nitsche);

    auto report = analyzer.analyze(ctx);

    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::BoundarySymbol));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::FluxBalance));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::TransferOperator));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::ParameterScale));
    EXPECT_TRUE(report.request_plan.has(AnalysisSummaryKind::AdjointConsistency));
    EXPECT_TRUE(requestReasonsContain(report.request_plan,
                                      AnalysisSummaryKind::BoundarySymbol,
                                      "penalty"));
    EXPECT_TRUE(requestReasonsContain(report.request_plan,
                                      AnalysisSummaryKind::FluxBalance,
                                      "flux-balance"));
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("ProblemAnalysisContext"));
}

TEST(NumericSummaryPlanner, RequestPlanTracksReasonsSourcesAndAvailability) {
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    auto var = VariableKey::field(0);
    DiscreteMatrixSummary matrix;
    matrix.block.domain = DomainKind::Cell;
    matrix.block.operator_tag = "matching-block";
    matrix.block.test_variables = {var};
    matrix.block.trial_variables = {var};
    summaries.discrete_matrices.push_back(matrix);
    ctx.setAnalysisSummaries(std::move(summaries));

    ProblemAnalysisReport report;
    auto claim = claimFrom("OperatorClassAnalyzer", PropertyKind::OperatorDefiniteness);
    claim.claim_origin.clear();
    claim.variables = {var};
    claim.tested_block_id = "matching-block";
    report.claims.push_back(std::move(claim));

    NumericSummaryPlanner planner;
    planner.run(ctx, report);

    ASSERT_TRUE(report.request_plan.has(AnalysisSummaryKind::DiscreteMatrix));
    const auto* request = firstRequest(report.request_plan,
                                       AnalysisSummaryKind::DiscreteMatrix);
    ASSERT_NE(request, nullptr);
    EXPECT_TRUE(request->already_available);
    EXPECT_FALSE(request->reasons.empty());
    EXPECT_EQ(request->source_claim_indices.size(), 1u);
    EXPECT_TRUE(report.request_plan.hasSourceAnalyzer("OperatorClassAnalyzer"));

    std::ostringstream oss;
    report.print(oss);
    const auto output = oss.str();
    EXPECT_NE(output.find("--- Requested Numeric Summaries ---"), std::string::npos);
    EXPECT_NE(output.find("DiscreteMatrix"), std::string::npos);
}

TEST(NumericSummaryPlanner, RequestPlanDoesNotUseWrongScopedSummaryAsAvailable) {
    ProblemAnalysisContext ctx;
    AnalysisSummarySet summaries;
    DiscreteMatrixSummary matrix;
    matrix.block.domain = DomainKind::Cell;
    matrix.block.operator_tag = "other-block";
    matrix.block.test_variables = {VariableKey::field(2)};
    matrix.block.trial_variables = {VariableKey::field(2)};
    summaries.discrete_matrices.push_back(matrix);
    ctx.setAnalysisSummaries(std::move(summaries));

    ProblemAnalysisReport report;
    auto claim = claimFrom("OperatorClassAnalyzer", PropertyKind::OperatorDefiniteness);
    claim.variables = {VariableKey::field(0)};
    claim.tested_block_id = "target-block";
    report.claims.push_back(std::move(claim));

    NumericSummaryPlanner planner;
    planner.run(ctx, report);

    const auto* request = firstRequest(report.request_plan,
                                       AnalysisSummaryKind::DiscreteMatrix);
    ASSERT_NE(request, nullptr);
    EXPECT_FALSE(request->already_available);
}

TEST(AnalysisSummaryMatching, ContributionIdRequiresCompatibleScope) {
    OperatorBlockId evidence;
    evidence.contribution_id = "contrib-1";
    evidence.operator_tag = "cell-op";
    evidence.domain = DomainKind::Cell;
    evidence.marker = 4;
    evidence.test_variables = {VariableKey::field(0)};
    evidence.trial_variables = {VariableKey::field(0)};

    OperatorBlockId target = evidence;
    EXPECT_TRUE(blockEvidenceMatches(evidence, target));
    EXPECT_TRUE(blockEvidenceCovers(evidence, target));

    target.domain = DomainKind::Boundary;
    EXPECT_FALSE(blockEvidenceMatches(evidence, target));
    EXPECT_FALSE(blockEvidenceCovers(evidence, target));

    target = evidence;
    target.test_variables = {VariableKey::field(1)};
    target.trial_variables = {VariableKey::field(1)};
    EXPECT_FALSE(blockEvidenceMatches(evidence, target));
    EXPECT_FALSE(blockEvidenceCovers(evidence, target));
}

TEST(NumericSummaryPlanner, RequestPlanKeepsInfSupScopesDistinct) {
    ProblemAnalysisContext ctx;
    ProblemAnalysisReport report;

    auto left = claimFrom("InfSupAnalyzer", PropertyKind::InfSupCondition);
    left.variables = {VariableKey::field(0), VariableKey::field(1)};
    left.tested_block_id = "left-pair";
    report.claims.push_back(std::move(left));

    auto right = claimFrom("InfSupAnalyzer", PropertyKind::InfSupCondition);
    right.variables = {VariableKey::field(0), VariableKey::field(1)};
    right.tested_block_id = "right-pair";
    report.claims.push_back(std::move(right));

    NumericSummaryPlanner planner;
    planner.run(ctx, report);

    const auto requests =
        report.request_plan.requestsOfKind(AnalysisSummaryKind::InfSupEstimate);
    ASSERT_EQ(requests.size(), 2u);

    bool saw_left = false;
    bool saw_right = false;
    for (const auto* request : requests) {
        ASSERT_NE(request, nullptr);
        EXPECT_EQ(request->variables.size(), 2u);
        if (request->block_id == "left-pair") {
            saw_left = true;
            EXPECT_EQ(request->scope_id, "left-pair");
        }
        if (request->block_id == "right-pair") {
            saw_right = true;
            EXPECT_EQ(request->scope_id, "right-pair");
        }
    }
    EXPECT_TRUE(saw_left);
    EXPECT_TRUE(saw_right);
}

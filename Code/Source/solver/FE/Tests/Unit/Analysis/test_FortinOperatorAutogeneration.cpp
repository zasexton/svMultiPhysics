/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include <sstream>

#include "Analysis/FortinOperatorAutogeneration.h"
#include "Analysis/ProblemAnalyzer.h"
#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;

namespace {

FieldDescriptor taylorHoodVelocity(FieldId id = 0)
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = "u";
    fd.field_type = FieldType::Vector;
    fd.value_dimension = 2;
    fd.topological_dimension = 2;
    fd.polynomial_order = 2;
    fd.space_family = SpaceFamily::H1;
    fd.element_family = ElementFamily::Lagrange;
    fd.continuity_class = SpaceContinuityClass::Continuous;
    fd.mapping_transform = MappingTransform::Identity;
    fd.reference_cell_family = ReferenceCellFamily::Simplex;
    fd.shape_regular_mesh_assumed = true;
    fd.domain_assumptions_present = true;
    fd.lipschitz_domain_assumed = true;
    fd.boundary_condition_scope_metadata_present = true;
    fd.strong_dirichlet_boundary_present = true;
    return fd;
}

FieldDescriptor h1Pressure(FieldId id = 1, int order = 1)
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = "p";
    fd.field_type = FieldType::Scalar;
    fd.value_dimension = 1;
    fd.topological_dimension = 2;
    fd.polynomial_order = order;
    fd.space_family = SpaceFamily::H1;
    fd.element_family = ElementFamily::Lagrange;
    fd.continuity_class = SpaceContinuityClass::Continuous;
    fd.mapping_transform = MappingTransform::Identity;
    fd.reference_cell_family = ReferenceCellFamily::Simplex;
    fd.shape_regular_mesh_assumed = true;
    fd.domain_assumptions_present = true;
    fd.lipschitz_domain_assumed = true;
    fd.mean_zero_constraint_present = true;
    return fd;
}

FieldDescriptor miniVelocity(FieldId id = 0, int bubble_degree = 1)
{
    auto fd = taylorHoodVelocity(id);
    fd.name = "u_mini";
    fd.polynomial_order = 1;
    fd.element_family = ElementFamily::BubbleEnrichedLagrange;
    fd.enrichment.visible_to_analysis = true;
    fd.enrichment.bubble_degree = bubble_degree;
    return fd;
}

FieldDescriptor rtFlux(FieldId id = 0, int order = 1)
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = "sigma";
    fd.field_type = FieldType::Vector;
    fd.value_dimension = 2;
    fd.topological_dimension = 2;
    fd.polynomial_order = order;
    fd.space_family = SpaceFamily::HDiv;
    fd.element_family = ElementFamily::RaviartThomas;
    fd.continuity_class = SpaceContinuityClass::NormalContinuous;
    fd.mapping_transform = MappingTransform::ContravariantPiola;
    fd.reference_cell_family = ReferenceCellFamily::Simplex;
    fd.shape_regular_mesh_assumed = true;
    fd.domain_assumptions_present = true;
    fd.lipschitz_domain_assumed = true;
    fd.boundary_condition_scope_metadata_present = true;
    fd.normal_trace_boundary_scope_present = true;
    fd.conformity.commuting_projection_metadata_present = true;
    return fd;
}

FieldDescriptor dgPressure(FieldId id = 1, int order = 1)
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = "p";
    fd.field_type = FieldType::Scalar;
    fd.value_dimension = 1;
    fd.topological_dimension = 2;
    fd.polynomial_order = order;
    fd.space_family = SpaceFamily::L2;
    fd.element_family = ElementFamily::DG;
    fd.continuity_class = SpaceContinuityClass::Discontinuous;
    fd.mapping_transform = MappingTransform::Identity;
    fd.reference_cell_family = ReferenceCellFamily::Simplex;
    fd.shape_regular_mesh_assumed = true;
    fd.domain_assumptions_present = true;
    fd.lipschitz_domain_assumed = true;
    return fd;
}

ContributionDescriptor constraintPair(bool stabilized = false)
{
    auto c = ContributionDescriptor::constraintPairDesc(
        VariableKey::field(0),
        VariableKey::field(1),
        "mixed-pair",
        "constraint",
        "unit-test");
    if (stabilized) {
        c.pairings.front().has_stabilizing_surrogate = true;
    }
    return c;
}

} // namespace

TEST(FortinOperatorAutogeneration, ClassifierRecognizesVectorDivergenceScalarMultiplierDag)
{
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(taylorHoodVelocity());
    ctx.addFieldDescriptor(h1Pressure());

    auto vel_space = std::make_shared<spaces::ProductSpace>(
        std::make_shared<spaces::H1Space>(ElementType::Triangle3, 2), 2);
    auto pres_space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    auto u = FormExpr::stateField(0, *vel_space, "u");
    auto q = FormExpr::testFunction(1, *pres_space, "q");
    auto residual = (q * div(u)).dx();

    auto c = constraintPair();
    c.source_expression = residual.nodeShared();
    ctx.addContribution(c);

    const auto descriptors = MixedCouplingClassifier{}.classify(ctx);
    ASSERT_EQ(descriptors.size(), 1u);
    EXPECT_EQ(descriptors[0].coupling_family,
              MixedCouplingFamily::VectorDivergenceScalarMultiplier);
    EXPECT_EQ(descriptors[0].evidence_strength,
              MixedCouplingEvidenceStrength::Strong);
    EXPECT_TRUE(descriptors[0].evidence_from_dag);
}

TEST(FortinOperatorAutogeneration, RegistryAcceptsAndRejectsKnownPairs)
{
    MixedCouplingDescriptor coupling;
    coupling.primal_variable = VariableKey::field(0);
    coupling.multiplier_variable = VariableKey::field(1);
    coupling.coupling_family = MixedCouplingFamily::VectorDivergenceScalarMultiplier;
    coupling.evidence_strength = MixedCouplingEvidenceStrength::Strong;

    FortinTheoremRegistry registry;
    auto match = registry.match(coupling, taylorHoodVelocity(), h1Pressure());
    ASSERT_TRUE(match.matched);
    ASSERT_NE(match.entry, nullptr);
    EXPECT_EQ(match.entry->theorem_id, "fortin:taylor-hood-p2-p1-simplex");

    auto wrong_pressure = h1Pressure(1, 2);
    auto miss = registry.match(coupling, taylorHoodVelocity(), wrong_pressure);
    EXPECT_FALSE(miss.matched);
    EXPECT_NE(std::find(miss.rejection_reasons.begin(),
                        miss.rejection_reasons.end(),
                        FortinRejectionReason::WrongOrderRelation),
              miss.rejection_reasons.end());

    auto p3 = taylorHoodVelocity();
    p3.polynomial_order = 3;
    auto p2 = h1Pressure(1, 2);
    auto generic = registry.match(coupling, p3, p2);
    EXPECT_FALSE(generic.matched);
}

TEST(FortinOperatorAutogeneration, RegistryRequiresPositiveMiniBubble)
{
    MixedCouplingDescriptor coupling;
    coupling.primal_variable = VariableKey::field(0);
    coupling.multiplier_variable = VariableKey::field(1);
    coupling.coupling_family = MixedCouplingFamily::VectorDivergenceScalarMultiplier;
    coupling.evidence_strength = MixedCouplingEvidenceStrength::Strong;

    FortinTheoremRegistry registry;
    auto ok = registry.match(coupling, miniVelocity(), h1Pressure());
    EXPECT_TRUE(ok.matched);

    auto missing_bubble = registry.match(coupling, miniVelocity(0, 0),
                                         h1Pressure());
    EXPECT_FALSE(missing_bubble.matched);
}

TEST(FortinOperatorAutogeneration, RegistryRejectsMiniWithDgPressure)
{
    MixedCouplingDescriptor coupling;
    coupling.primal_variable = VariableKey::field(0);
    coupling.multiplier_variable = VariableKey::field(1);
    coupling.coupling_family = MixedCouplingFamily::VectorDivergenceScalarMultiplier;
    coupling.evidence_strength = MixedCouplingEvidenceStrength::Strong;

    FortinTheoremRegistry registry;
    auto dg_mini = registry.match(coupling, miniVelocity(), dgPressure(1, 1));
    EXPECT_FALSE(dg_mini.matched);
}

TEST(FortinOperatorAutogeneration, CandidateBuilderCompletesTaylorHoodAndRTDG)
{
    ProblemAnalysisContext th;
    th.addFieldDescriptor(taylorHoodVelocity());
    th.addFieldDescriptor(h1Pressure());
    th.addContribution(constraintPair());

    auto th_result = FortinCandidateBuilder{}.build(th);
    ASSERT_EQ(th_result.candidates.size(), 1u);
    EXPECT_EQ(th_result.candidates[0].status, FortinCandidateStatus::Complete);
    EXPECT_TRUE(th_result.candidates[0].constructive_fortin);
    ASSERT_TRUE(th_result.candidates[0].projection_plan.has_value());

    ProblemAnalysisContext rt;
    rt.addFieldDescriptor(rtFlux());
    rt.addFieldDescriptor(dgPressure());
    rt.addContribution(constraintPair());

    auto rt_result = FortinCandidateBuilder{}.build(rt);
    ASSERT_EQ(rt_result.candidates.size(), 1u);
    EXPECT_EQ(rt_result.candidates[0].status, FortinCandidateStatus::Complete);
    EXPECT_TRUE(rt_result.candidates[0].commuting_projection);
}

TEST(FortinOperatorAutogeneration, MissingGaugeAndUnknownSpaceStayNoncertifying)
{
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(taylorHoodVelocity());
    auto pressure = h1Pressure();
    pressure.mean_zero_constraint_present = false;
    pressure.gauge_fixing_metadata_present = false;
    ctx.addFieldDescriptor(pressure);
    ctx.addContribution(constraintPair());

    auto result = FortinCandidateBuilder{}.build(ctx);
    ASSERT_EQ(result.candidates.size(), 1u);
    EXPECT_EQ(result.candidates[0].status, FortinCandidateStatus::Incomplete);
    EXPECT_NE(std::find(result.candidates[0].missing_or_rejected_reasons.begin(),
                        result.candidates[0].missing_or_rejected_reasons.end(),
                        FortinRejectionReason::MissingBoundaryNullspaceAssumption),
              result.candidates[0].missing_or_rejected_reasons.end());

    ProblemAnalysisContext custom;
    auto velocity = taylorHoodVelocity();
    velocity.element_family = ElementFamily::Unknown;
    custom.addFieldDescriptor(velocity);
    custom.addFieldDescriptor(h1Pressure());
    custom.addContribution(constraintPair());
    auto custom_result = FortinCandidateBuilder{}.build(custom);
    ASSERT_EQ(custom_result.candidates.size(), 1u);
    EXPECT_NE(custom_result.candidates[0].status, FortinCandidateStatus::Complete);
}

TEST(FortinOperatorAutogeneration, StabilizedSurrogateDoesNotRequestOrCertifyFortin)
{
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(taylorHoodVelocity());
    ctx.addFieldDescriptor(h1Pressure());
    ctx.addContribution(constraintPair(/*stabilized=*/true));

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);
    EXPECT_FALSE(report.request_plan.has(
        AnalysisSummaryKind::InfSupPairCertification));
    for (const auto& claim : report.claims) {
        EXPECT_FALSE(claim.kind == PropertyKind::InfSupCondition &&
                     claim.certification_class &&
                     *claim.certification_class == CertificationClass::Certified);
    }
}

TEST(FortinOperatorAutogeneration, AnalyzerProducesCertifiedSummaryClaim)
{
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(taylorHoodVelocity());
    ctx.addFieldDescriptor(h1Pressure());
    ctx.addContribution(constraintPair());

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);
    EXPECT_TRUE(report.request_plan.has(
        AnalysisSummaryKind::InfSupPairCertification));

    bool certified = false;
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition &&
            claim.certification_class &&
            *claim.certification_class == CertificationClass::Certified &&
            claim.claim_origin == "FortinCertificationAnalyzer") {
            certified = true;
        }
    }
    EXPECT_TRUE(certified);
}

TEST(FortinOperatorAutogeneration, LocalProjectionBuilderIsOptionGated)
{
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(taylorHoodVelocity());
    ctx.addFieldDescriptor(h1Pressure());
    ctx.addContribution(constraintPair());

    auto result = FortinCandidateBuilder{}.build(ctx);
    ASSERT_EQ(result.candidates.size(), 1u);
    const auto& candidate = result.candidates[0];
    ASSERT_EQ(candidate.status, FortinCandidateStatus::Complete);

    LocalFortinProjectionBuilder builder;
    auto metadata_only = builder.build(candidate, LocalFortinProjectionOptions{});
    EXPECT_EQ(metadata_only.status, LocalFortinProjectionStatus::MetadataOnly);

    LocalFortinProjectionOptions verify;
    verify.verify_preservation_identities = true;
    auto verified = builder.build(candidate, ctx, verify);
    EXPECT_EQ(verified.status,
              LocalFortinProjectionStatus::LocalDiagnosticPass);
    EXPECT_TRUE(verified.preservation_identity_verified);
    EXPECT_TRUE(verified.local_projection_matrix_present);
    EXPECT_GT(verified.target_dof_count, 0);
    EXPECT_GT(verified.source_dof_count, 0);
    EXPECT_GT(verified.multiplier_dof_count, 0);
    EXPECT_LE(verified.preservation_residual_max_abs,
              verified.preservation_residual_tolerance);

    LocalFortinProjectionOptions dense;
    dense.build_local_projection_matrices = true;
    dense.verify_preservation_identities = true;
    dense.estimate_norm_bound = true;
    auto constructed = builder.build(candidate, ctx, dense);
    EXPECT_EQ(constructed.status,
              LocalFortinProjectionStatus::LocalDiagnosticPass);
    ASSERT_EQ(constructed.local_matrices.size(), 1u);
    EXPECT_EQ(constructed.local_matrices[0].rows,
              constructed.target_dof_count);
    EXPECT_EQ(constructed.local_matrices[0].cols,
              constructed.source_dof_count);
    EXPECT_TRUE(constructed.norm_bound_estimate_present);
    EXPECT_GT(constructed.norm_bound_estimate, 0.0);
}

TEST(FortinOperatorAutogeneration, LocalProjectionBuilderVerifiesRTDG)
{
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(rtFlux());
    ctx.addFieldDescriptor(dgPressure());
    ctx.addContribution(constraintPair());

    auto result = FortinCandidateBuilder{}.build(ctx);
    ASSERT_EQ(result.candidates.size(), 1u);
    const auto& candidate = result.candidates[0];
    ASSERT_EQ(candidate.status, FortinCandidateStatus::Complete);

    LocalFortinProjectionOptions options;
    options.build_local_projection_matrices = true;
    options.verify_preservation_identities = true;

    const auto projection =
        LocalFortinProjectionBuilder{}.build(candidate, ctx, options);
    EXPECT_EQ(projection.status,
              LocalFortinProjectionStatus::LocalDiagnosticPass);
    EXPECT_TRUE(projection.commuting_projection_metadata_present);
    EXPECT_TRUE(projection.local_projection_matrix_present);
    EXPECT_LE(projection.preservation_residual_max_abs,
              projection.preservation_residual_tolerance);
}

TEST(FortinOperatorAutogeneration, AnalyzerEmitsRichRunLogSummary)
{
    ProblemAnalysisContext ctx;
    ctx.addFieldDescriptor(taylorHoodVelocity());
    ctx.addFieldDescriptor(h1Pressure());
    ctx.addContribution(constraintPair());

    auto report = ProblemAnalyzer::createDefault().analyze(ctx);
    ASSERT_FALSE(report.run_logs.empty());

    const auto it = std::find_if(
        report.run_logs.begin(),
        report.run_logs.end(),
        [](const AnalyzerRunLogSummary& log) {
            return log.analyzer == "FortinCertificationAnalyzer";
        });
    ASSERT_NE(it, report.run_logs.end());
    EXPECT_EQ(it->status, "certified");
    EXPECT_EQ(it->attempted_count, 1u);
    EXPECT_EQ(it->certified_count, 1u);
    EXPECT_FALSE(it->detail_lines.empty());

    std::ostringstream app;
    report.printApplicationLog(app);
    EXPECT_NE(app.str().find("RunLog analyzer=FortinCertificationAnalyzer"),
              std::string::npos);

    std::ostringstream trace;
    report.printTraceLog(trace);
    EXPECT_NE(trace.str().find("run_log_detail"), std::string::npos);
}

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/AnalysisSummaryProducer.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ProblemAnalyzer.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;

namespace {

FieldDescriptor h1Vector(FieldId id, std::string name, int order = 2)
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
    fd.field_type = FieldType::Vector;
    fd.value_dimension = 2;
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
    fd.boundary_condition_scope_metadata_present = true;
    fd.strong_dirichlet_boundary_present = true;
    return fd;
}

FieldDescriptor h1Scalar(FieldId id, std::string name, int order = 1)
{
    FieldDescriptor fd;
    fd.field_id = id;
    fd.name = std::move(name);
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

AnalysisSummaryRequest matrixRequest(std::string block,
                                     VariableKey variable)
{
    AnalysisSummaryRequest request;
    request.summary_kind = AnalysisSummaryKind::DiscreteMatrix;
    request.domain = DomainKind::Cell;
    request.variables = {variable};
    request.test_variables = {variable};
    request.trial_variables = {variable};
    request.block_id = std::move(block);
    request.request_id = "matrix-request";
    return request;
}

class MatrixAssemblyAccess final : public AssemblyAccess {
public:
    explicit MatrixAssemblyAccess(DiscreteMatrixSummary summary)
        : summary_(std::move(summary)) {}

    std::optional<DiscreteMatrixSummary>
    discreteMatrixSummary(const AnalysisSummaryRequest&) const override
    {
        return summary_;
    }

private:
    DiscreteMatrixSummary summary_;
};

class EmptyMeshAccess final : public MeshAccess {
public:
    std::string meshRevision() const override { return "mesh-rev-test"; }
};

class EmptySolverAccess final : public SolverAccess {};
class EmptyAssemblyAccess final : public AssemblyAccess {};

const AnalysisSummaryRequest* firstRequest(const AnalysisRequestPlan& plan,
                                           AnalysisSummaryKind kind)
{
    const auto requests = plan.requestsOfKind(kind);
    return requests.empty() ? nullptr : requests.front();
}

} // namespace

TEST(AnalysisSummaryProducerRegistry, FulfillsRequestPlanWithStrictScopedProducer)
{
    const auto u = VariableKey::field(0);
    auto request = matrixRequest("target-block", u);
    AnalysisRequestPlan plan;
    plan.summary_requests.push_back(request);

    DiscreteMatrixSummary matrix;
    matrix.block = requestTargetBlock(request);
    matrix.rows = 2;
    matrix.cols = 2;
    matrix.sign_evidence_complete = true;

    ProblemAnalysisContext context;
    MatrixAssemblyAccess assembly(matrix);
    EmptyMeshAccess mesh;
    EmptySolverAccess solver;

    auto registry = AnalysisSummaryProducerRegistry::createDefault();
    auto batch = registry.fulfillRequestPlan(plan,
                                             context,
                                             assembly,
                                             mesh,
                                             solver);

    ASSERT_EQ(batch.produced_summaries.discrete_matrices.size(), 1u);
    ASSERT_EQ(batch.fulfilled_plan.summary_requests.size(), 1u);
    EXPECT_TRUE(batch.fulfilled_plan.summary_requests.front().production_succeeded);
    EXPECT_TRUE(batch.fulfilled_plan.summary_requests.front().already_available);
    EXPECT_EQ(batch.produced_summaries.discrete_matrices.front().evidence.producer_id,
              "DiscreteMatrixSummaryProducer");
    EXPECT_TRUE(analysisSummarySetCoversRequest(batch.produced_summaries,
                                                request));
}

TEST(AnalysisSummaryProducerRegistry, RejectsProducedEvidenceWithWrongScope)
{
    const auto u = VariableKey::field(0);
    auto request = matrixRequest("target-block", u);
    AnalysisRequestPlan plan;
    plan.summary_requests.push_back(request);

    DiscreteMatrixSummary wrong;
    wrong.block = requestTargetBlock(request);
    wrong.block.operator_tag = "other-block";
    wrong.block.test_variables = {VariableKey::field(1)};
    wrong.block.trial_variables = {VariableKey::field(1)};

    ProblemAnalysisContext context;
    MatrixAssemblyAccess assembly(wrong);
    EmptyMeshAccess mesh;
    EmptySolverAccess solver;

    auto registry = AnalysisSummaryProducerRegistry::createDefault();
    auto batch = registry.fulfillRequestPlan(plan,
                                             context,
                                             assembly,
                                             mesh,
                                             solver);

    EXPECT_TRUE(batch.produced_summaries.empty());
    ASSERT_EQ(batch.fulfilled_plan.summary_requests.size(), 1u);
    EXPECT_FALSE(batch.fulfilled_plan.summary_requests.front().production_succeeded);
    EXPECT_TRUE(batch.fulfilled_plan.summary_requests.front().production_unavailable);
    EXPECT_EQ(batch.failed_count, 1u);
    EXPECT_EQ(batch.fulfilled_plan.summary_requests.front().production_status,
              "FailedStrictScopeCoverage");
}

TEST(AnalysisSummaryProducerRegistry, NormInferenceUsesSpaceMetadataNotFieldNames)
{
    ProblemAnalysisContext context;
    context.addFieldDescriptor(h1Vector(10, "banana"));
    auto multiplier = h1Scalar(11, "not_pressure");
    multiplier.mean_zero_constraint_present = true;
    context.addFieldDescriptor(multiplier);

    AnalysisSummaryRequest request;
    request.summary_kind = AnalysisSummaryKind::NormMetadata;
    request.domain = DomainKind::Cell;
    request.variables = {VariableKey::field(10), VariableKey::field(11)};
    request.test_variables = request.variables;
    request.trial_variables = request.variables;
    request.request_id = "norms-for-mixed-pair";
    request.scope_id = "mixed-pair-product";

    AnalysisRequestPlan plan;
    plan.summary_requests.push_back(request);

    EmptyAssemblyAccess assembly;
    EmptyMeshAccess mesh;
    EmptySolverAccess solver;
    auto registry = AnalysisSummaryProducerRegistry::createDefault();
    auto batch = registry.fulfillRequestPlan(plan,
                                             context,
                                             assembly,
                                             mesh,
                                             solver);

    ASSERT_EQ(batch.produced_summaries.norm_metadata.size(), 3u);
    EXPECT_EQ(batch.produced_summaries.norm_metadata[0].norm_family, "H1-vector");
    EXPECT_EQ(batch.produced_summaries.norm_metadata[1].norm_family, "H1");
    EXPECT_EQ(batch.produced_summaries.norm_metadata[1].nullspace_handling,
              NullspaceHandlingClass::ProjectedOut);
    EXPECT_EQ(batch.produced_summaries.norm_metadata[0].evidence.provenance.front(),
              EvidenceProvenance::InferredFromFormAndSpaces);
    const auto& product = batch.produced_summaries.norm_metadata[2];
    EXPECT_EQ(product.norm_family, "Product");
    EXPECT_EQ(product.scope_id, request.scope_id);
    EXPECT_EQ(product.variables.size(), 2u);
    EXPECT_TRUE(analysisSummarySetCoversRequest(batch.produced_summaries,
                                                request));
    ASSERT_EQ(batch.fulfilled_plan.summary_requests.size(), 1u);
    EXPECT_TRUE(batch.fulfilled_plan.summary_requests.front().production_succeeded);
}

TEST(AnalysisSummaryProducerRegistry, StandardMissingKindsHaveRegisteredProducers)
{
    AnalysisRequestPlan plan;
    const std::vector<AnalysisSummaryKind> kinds{
        AnalysisSummaryKind::NullspaceDegeneracy,
        AnalysisSummaryKind::ParameterScale,
        AnalysisSummaryKind::NumericalErrorBudget,
        AnalysisSummaryKind::StabilizationAdequacy,
        AnalysisSummaryKind::InitialCompatibility,
        AnalysisSummaryKind::LocalStencil,
    };
    int index = 0;
    for (const auto kind : kinds) {
        AnalysisSummaryRequest request;
        request.summary_kind = kind;
        request.domain = DomainKind::Cell;
        request.variables = {VariableKey::field(0)};
        request.test_variables = request.variables;
        request.trial_variables = request.variables;
        request.request_id = "registered-kind-" + std::to_string(index++);
        plan.summary_requests.push_back(std::move(request));
    }

    ProblemAnalysisContext context;
    context.addFieldDescriptor(h1Scalar(0, "field"));
    EmptyAssemblyAccess assembly;
    EmptyMeshAccess mesh;
    EmptySolverAccess solver;

    auto registry = AnalysisSummaryProducerRegistry::createDefault();
    auto batch = registry.fulfillRequestPlan(plan,
                                             context,
                                             assembly,
                                             mesh,
                                             solver);

    ASSERT_EQ(batch.fulfilled_plan.summary_requests.size(), kinds.size());
    for (const auto& request : batch.fulfilled_plan.summary_requests) {
        EXPECT_NE(request.unavailable_reason, "no registered producer")
            << toString(request.summary_kind);
    }
}

TEST(AnalysisSummaryProducerRegistry, FortinStablePairProducerIgnoresFieldNames)
{
    ProblemAnalysisContext context;
    context.addFieldDescriptor(h1Vector(0, "alpha", 2));
    context.addFieldDescriptor(h1Scalar(1, "beta", 1));
    auto contribution = ContributionDescriptor::constraintPairDesc(
        VariableKey::field(0),
        VariableKey::field(1),
        "mixed-pair",
        "constraint",
        "unit-test");
    const auto contribution_id = contribution.contribution_id;
    context.addContribution(std::move(contribution));

    AnalysisSummaryRequest request;
    request.summary_kind = AnalysisSummaryKind::InfSupPairCertification;
    request.domain = DomainKind::Cell;
    request.variables = {VariableKey::field(0), VariableKey::field(1)};
    request.test_variables = request.variables;
    request.trial_variables = request.variables;
    request.block_id = "constraint";
    request.contribution_id = contribution_id;
    request.request_id = "fortin-request";

    AnalysisRequestPlan plan;
    plan.summary_requests.push_back(request);

    EmptyAssemblyAccess assembly;
    EmptyMeshAccess mesh;
    EmptySolverAccess solver;
    auto registry = AnalysisSummaryProducerRegistry::createDefault();
    auto batch = registry.fulfillRequestPlan(plan,
                                             context,
                                             assembly,
                                             mesh,
                                             solver);

    ASSERT_EQ(batch.produced_summaries.inf_sup_pair_certifications.size(), 1u);
    const auto& summary =
        batch.produced_summaries.inf_sup_pair_certifications.front();
    EXPECT_EQ(summary.inf_sup_theorem_id,
              "fortin:taylor-hood-p2-p1-simplex");
    EXPECT_EQ(summary.evidence.producer_id, "FortinStablePairProducer");
    EXPECT_TRUE(summary.evidence.theorem_matched_evidence);
}

TEST(ProblemAnalyzer, EvidenceSynthesisFulfillsPlannerBeforeSummaryConsumers)
{
    ProblemAnalysisContext context;
    context.addFieldDescriptor(h1Vector(0, "alpha", 2));
    context.addFieldDescriptor(h1Scalar(1, "beta", 1));
    context.addContribution(ContributionDescriptor::constraintPairDesc(
        VariableKey::field(0),
        VariableKey::field(1),
        "mixed-pair",
        "constraint",
        "unit-test"));

    EmptyAssemblyAccess assembly;
    EmptyMeshAccess mesh;
    EmptySolverAccess solver;
    auto analyzer = ProblemAnalyzer::createDefault();
    auto registry = AnalysisSummaryProducerRegistry::createDefault();
    auto report = analyzer.analyzeWithEvidenceSynthesis(context,
                                                        registry,
                                                        assembly,
                                                        mesh,
                                                        solver);

    const auto* request =
        firstRequest(report.request_plan,
                     AnalysisSummaryKind::InfSupPairCertification);
    ASSERT_NE(request, nullptr);
    EXPECT_TRUE(request->already_available);
    EXPECT_FALSE(request->production_status.empty());

    bool saw_certified = false;
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::InfSupCondition &&
            claim.certification_class &&
            *claim.certification_class == CertificationClass::Certified) {
            saw_certified = true;
            break;
        }
    }
    EXPECT_TRUE(saw_certified);
}

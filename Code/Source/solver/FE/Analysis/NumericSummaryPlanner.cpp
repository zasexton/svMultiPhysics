/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/NumericSummaryPlanner.h"

#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/ContributionDescriptor.h"

#include <algorithm>
#include <sstream>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

template <typename T>
void appendUnique(std::vector<T>& values, const T& value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

void appendVariables(std::vector<VariableKey>& target,
                     const std::vector<VariableKey>& values) {
    for (const auto& value : values) {
        appendUnique(target, value);
    }
}

std::string sourceAnalyzer(const PropertyClaim& claim) {
    if (!claim.claim_origin.empty()) {
        return claim.claim_origin;
    }
    for (const auto& evidence : claim.evidence) {
        if (!evidence.source.empty()) {
            return evidence.source;
        }
    }
    return "UnknownAnalyzer";
}

bool hasSourceAnalyzer(const PropertyClaim& claim, const char* analyzer) {
    if (claim.claim_origin == analyzer) {
        return true;
    }
    return std::any_of(claim.evidence.begin(), claim.evidence.end(),
                       [analyzer](const PropertyEvidence& evidence) {
                           return evidence.source == analyzer;
                       });
}

AnalysisConfidence strongerConfidence(AnalysisConfidence a, AnalysisConfidence b) {
    auto rank = [](AnalysisConfidence c) {
        switch (c) {
            case AnalysisConfidence::High: return 3;
            case AnalysisConfidence::Medium: return 2;
            case AnalysisConfidence::Low: return 1;
        }
        return 0;
    };
    return rank(b) > rank(a) ? b : a;
}

const FieldDescriptor* fieldDescriptorForClaim(const ProblemAnalysisContext& context,
                                               const PropertyClaim& claim) {
    if (claim.field != INVALID_FIELD_ID) {
        if (const auto* fd = context.fieldDescriptor(claim.field)) {
            return fd;
        }
    }

    for (const auto& variable : claim.variables) {
        if (variable.kind != VariableKind::FieldComponent ||
            variable.field_id == INVALID_FIELD_ID) {
            continue;
        }
        if (const auto* fd = context.fieldDescriptor(variable.field_id)) {
            return fd;
        }
    }

    return nullptr;
}

bool isScalarFieldClaim(const ProblemAnalysisContext& context,
                        const PropertyClaim& claim) {
    const auto* fd = fieldDescriptorForClaim(context, claim);
    if (!fd) {
        return false;
    }
    return fd->value_dimension == 1 && fd->field_type == FieldType::Scalar;
}

std::vector<VariableKey> variablesForFields(const std::vector<FieldId>& fields) {
    std::vector<VariableKey> variables;
    variables.reserve(fields.size());
    for (FieldId field : fields) {
        if (field != INVALID_FIELD_ID) {
            appendUnique(variables, VariableKey::field(field));
        }
    }
    return variables;
}

bool containsVariable(const std::vector<VariableKey>& values,
                      const VariableKey& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool sameVariableSet(const std::vector<VariableKey>& a,
                     const std::vector<VariableKey>& b) {
    return a.size() == b.size() &&
           std::all_of(a.begin(), a.end(), [&](const VariableKey& value) {
               return containsVariable(b, value);
           });
}

bool stabilizedInfSupSurrogateCovers(const ProblemAnalysisReport& report,
                                     const PropertyClaim& claim) {
    return std::any_of(report.claims.begin(), report.claims.end(),
                       [&](const PropertyClaim& other) {
                           return other.kind == PropertyKind::InfSupCondition &&
                                  other.inf_sup_class.has_value() &&
                                  *other.inf_sup_class == InfSupClass::StabilizedSurrogate &&
                                  !other.variables.empty() &&
                                  sameVariableSet(other.variables, claim.variables);
                       });
}

bool shouldRequestStablePairCertification(const ProblemAnalysisReport& report,
                                          const PropertyClaim& claim) {
    if (claim.inf_sup_class.has_value() &&
        *claim.inf_sup_class == InfSupClass::StabilizedSurrogate) {
        return false;
    }
    return !stabilizedInfSupSurrogateCovers(report, claim);
}

void appendVariableId(std::ostringstream& os, const VariableKey& variable) {
    os << static_cast<int>(variable.kind) << ':'
       << variable.field_id << ':'
       << variable.component << ':'
       << variable.name;
}

std::string variablesScopeId(const std::vector<VariableKey>& variables) {
    std::ostringstream os;
    for (const auto& variable : variables) {
        os << '[';
        appendVariableId(os, variable);
        os << ']';
    }
    return os.str();
}

std::string optionalString(const std::optional<std::string>& value) {
    return value ? *value : std::string{};
}

std::string claimScopeId(const PropertyClaim& claim) {
    if (claim.estimate_scope) return *claim.estimate_scope;
    if (claim.tested_block_id) return *claim.tested_block_id;
    if (claim.coefficient_id) return *claim.coefficient_id;
    if (claim.invariant_set_id) return *claim.invariant_set_id;
    if (claim.equilibrium_id) return *claim.equilibrium_id;
    return {};
}

AnalysisSummaryRequest& ensureRequest(AnalysisRequestPlan& plan,
                                      AnalysisSummaryKind kind,
                                      DomainKind domain,
                                      const std::vector<VariableKey>& variables,
                                      const std::string& block_id,
                                      const std::string& contribution_id,
                                      const std::string& scope_id) {
    auto it = std::find_if(plan.summary_requests.begin(),
                           plan.summary_requests.end(),
                           [&](const AnalysisSummaryRequest& request) {
                               return request.summary_kind == kind &&
                                      request.domain == domain &&
                                      sameVariableSet(request.variables,
                                                      variables) &&
                                      request.block_id == block_id &&
                                      request.contribution_id == contribution_id &&
                                      request.scope_id == scope_id;
                           });
    if (it != plan.summary_requests.end()) {
        return *it;
    }

    AnalysisSummaryRequest request;
    request.summary_kind = kind;
    request.domain = domain;
    request.variables = variables;
    request.block_id = block_id;
    request.contribution_id = contribution_id;
    request.scope_id = scope_id;
    request.request_id = std::string(toString(kind)) + ":" + toString(domain);
    if (!block_id.empty()) request.request_id += ":block=" + block_id;
    if (!contribution_id.empty()) {
        request.request_id += ":contribution=" + contribution_id;
    }
    if (!scope_id.empty()) request.request_id += ":scope=" + scope_id;
    if (!variables.empty()) {
        request.request_id += ":vars=" + variablesScopeId(variables);
    }
    plan.summary_requests.push_back(std::move(request));
    return plan.summary_requests.back();
}

void populateRequestVariableRoles(AnalysisSummaryRequest& request,
                                  AnalysisSummaryKind kind,
                                  const std::vector<VariableKey>& variables)
{
    appendVariables(request.state_variables, variables);
    switch (kind) {
        case AnalysisSummaryKind::InfSupEstimate:
        case AnalysisSummaryKind::InfSupPairCertification:
        case AnalysisSummaryKind::SchurComplement:
            if (!variables.empty()) {
                appendUnique(request.test_variables, variables.front());
                appendUnique(request.trial_variables, variables.front());
            }
            if (variables.size() >= 2u) {
                appendUnique(request.multiplier_variables, variables[1]);
            }
            break;
        case AnalysisSummaryKind::BoundarySymbol:
        case AnalysisSummaryKind::FluxBalance:
        case AnalysisSummaryKind::TransferOperator:
            appendVariables(request.test_variables, variables);
            appendVariables(request.trial_variables, variables);
            break;
        default:
            appendVariables(request.state_variables, variables);
            break;
    }
}

bool invariantDescriptorHasSchemeLevelEvidence(
    const InvariantDomainDescriptor& descriptor) noexcept
{
    return !descriptor.theorem_id.empty() ||
           descriptor.limiter_evidence_present ||
           descriptor.cfl_condition_satisfied ||
           descriptor.ssp_time_discretization_evidence_present ||
           descriptor.low_order_invariant_domain_evidence_present ||
           descriptor.convex_limiting_evidence_present ||
           descriptor.spatial_monotonicity_evidence_present ||
           descriptor.mass_positivity_evidence_present;
}

OperatorBlockId requestedBlock(DomainKind domain,
                               const std::vector<VariableKey>& variables,
                               const std::string& block_id,
                               const std::string& contribution_id) {
    OperatorBlockId block;
    block.domain = domain;
    block.operator_tag = block_id;
    block.contribution_id = contribution_id;
    block.test_variables = variables;
    block.trial_variables = variables;
    return block;
}

std::vector<std::string> requestedContributionIds(
    const std::string& contribution_id) {
    if (contribution_id.empty()) {
        return {};
    }
    return {contribution_id};
}

bool scopedBlockCoversRequest(const OperatorBlockId& evidence_block,
                              const std::vector<VariableKey>& evidence_variables,
                              const std::string& evidence_contribution_id,
                              const OperatorBlockId& target_block,
                              const std::vector<std::string>& target_contribution_ids) {
    if (target_block.operator_tag.empty() &&
        !evidence_block.operator_tag.empty() &&
        evidence_block.contribution_id.empty() &&
        evidence_contribution_id.empty()) {
        return false;
    }
    if (target_contribution_ids.empty() &&
        (!evidence_block.contribution_id.empty() ||
         !evidence_contribution_id.empty())) {
        return false;
    }
    return strictScopedEvidenceMatches(evidence_block,
                                       evidence_variables,
                                       evidence_contribution_id,
                                       target_block,
                                       target_contribution_ids);
}

bool matrixSummaryCoversRequest(const DiscreteMatrixSummary& summary,
                                const OperatorBlockId& target_block,
                                const std::vector<std::string>& target_contribution_ids) {
    if (!target_block.contribution_id.empty() &&
        summary.block.contribution_id != target_block.contribution_id &&
        std::find(summary.contribution_ids.begin(),
                  summary.contribution_ids.end(),
                  target_block.contribution_id) ==
            summary.contribution_ids.end()) {
        return false;
    }
    return scopedBlockCoversRequest(summary.block,
                                    variablesForBlock(summary.block),
                                    summary.block.contribution_id,
                                    target_block,
                                    target_contribution_ids);
}

bool variablesCoverRequest(const std::vector<VariableKey>& evidence,
                           DomainKind evidence_domain,
                           const OperatorBlockId& target_block) {
    if (evidence_domain != target_block.domain) {
        return false;
    }
    return variableSetCoversAll(evidence, variablesForBlock(target_block));
}

bool summaryAlreadyAvailable(const ProblemAnalysisContext& context,
                             AnalysisSummaryKind kind,
                             DomainKind domain,
                             const std::vector<VariableKey>& variables,
                             const std::string& block_id,
                             const std::string& contribution_id,
                             const std::string& scope_id) {
    const auto* summaries = context.analysisSummaries();
    if (!summaries) {
        return false;
    }
    const auto target_block =
        requestedBlock(domain, variables, block_id, contribution_id);
    const auto target_contribution_ids =
        requestedContributionIds(contribution_id);
    auto block_covers = [&](const OperatorBlockId& block,
                            const std::vector<VariableKey>& evidence_variables,
                            const std::string& evidence_contribution = {}) {
        return scopedBlockCoversRequest(block,
                                        evidence_variables,
                                        evidence_contribution,
                                        target_block,
                                        target_contribution_ids);
    };
    switch (kind) {
        case AnalysisSummaryKind::CoefficientProperties:
            return std::any_of(
                summaries->coefficient_properties.begin(),
                summaries->coefficient_properties.end(),
                [&](const CoefficientPropertySummary& summary) {
                    return coefficientSummaryCovers(
                        summary, target_block, target_contribution_ids);
                });
        case AnalysisSummaryKind::DiscreteMatrix:
            return std::any_of(
                summaries->discrete_matrices.begin(),
                summaries->discrete_matrices.end(),
                [&](const DiscreteMatrixSummary& summary) {
                    return matrixSummaryCoversRequest(summary,
                                                      target_block,
                                                      target_contribution_ids);
                });
        case AnalysisSummaryKind::ReducedMatrix:
            return std::any_of(
                summaries->reduced_matrices.begin(),
                summaries->reduced_matrices.end(),
                [&](const ReducedMatrixSummary& summary) {
                    return matrixSummaryCoversRequest(summary.free_free_matrix,
                                                      target_block,
                                                      target_contribution_ids);
                });
        case AnalysisSummaryKind::LocalStencil:
            return std::any_of(
                summaries->local_stencils.begin(),
                summaries->local_stencils.end(),
                [&](const LocalStencilSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::FluxBalance:
            return std::any_of(
                summaries->flux_balances.begin(),
                summaries->flux_balances.end(),
                [&](const FluxBalanceSummary& summary) {
                    return block_covers(summary.block, {});
                });
        case AnalysisSummaryKind::TemporalStability:
            return std::any_of(
                summaries->temporal_stability.begin(),
                summaries->temporal_stability.end(),
                [&](const TemporalStabilitySummary& summary) {
                    return block_covers(summary.block,
                                        summary.variables,
                                        summary.contribution_id);
                });
        case AnalysisSummaryKind::InfSupEstimate:
            return std::any_of(
                summaries->inf_sup_estimates.begin(),
                summaries->inf_sup_estimates.end(),
                [&](const InfSupEstimateSummary& summary) {
                    std::vector<VariableKey> pair{
                        summary.primal_variable,
                        summary.multiplier_variable};
                    const bool scope_ok =
                        scope_id.empty() ||
                        summary.estimate_scope.empty() ||
                        summary.estimate_scope == scope_id;
                    return scope_ok && block_covers(summary.block, pair);
                });
        case AnalysisSummaryKind::InfSupPairCertification:
            return std::any_of(
                summaries->inf_sup_pair_certifications.begin(),
                summaries->inf_sup_pair_certifications.end(),
                [&](const InfSupPairCertificationSummary& summary) {
                    std::vector<VariableKey> pair{
                        summary.primal_variable,
                        summary.multiplier_variable};
                    return block_covers(summary.block, pair);
                });
        case AnalysisSummaryKind::ParameterScale:
            return std::any_of(
                summaries->parameter_scales.begin(),
                summaries->parameter_scales.end(),
                [&](const ParameterScaleSummary& summary) {
                    return block_covers(summary.block,
                                        summary.variables,
                                        summary.contribution_id);
                });
        case AnalysisSummaryKind::StabilizationAdequacy:
            return std::any_of(
                summaries->stabilization_adequacy.begin(),
                summaries->stabilization_adequacy.end(),
                [&](const StabilizationAdequacySummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::SchurComplement:
            return std::any_of(
                summaries->schur_complements.begin(),
                summaries->schur_complements.end(),
                [&](const SchurComplementSummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::NullspaceDegeneracy:
            return std::any_of(
                summaries->nullspace_degeneracies.begin(),
                summaries->nullspace_degeneracies.end(),
                [&](const NullspaceDegeneracySummary& summary) {
                    return block_covers(summary.block,
                                        summary.affected_variables);
                });
        case AnalysisSummaryKind::RobustnessTrend:
            return std::any_of(
                summaries->robustness_trends.begin(),
                summaries->robustness_trends.end(),
                [&](const RobustnessTrendSummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::Applicability:
            return std::any_of(
                summaries->applicability.begin(),
                summaries->applicability.end(),
                [&](const ApplicabilitySummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::NumericalErrorBudget:
            return std::any_of(
                summaries->numerical_error_budgets.begin(),
                summaries->numerical_error_budgets.end(),
                [&](const NumericalErrorBudgetSummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::DAEStructureEvidence:
            return std::any_of(
                summaries->dae_structure_evidence.begin(),
                summaries->dae_structure_evidence.end(),
                [&](const DAEStructureEvidenceSummary& summary) {
                    return variablesCoverRequest(summary.variables,
                                                 domain,
                                                 target_block);
                });
        case AnalysisSummaryKind::CompatibleComplex:
            return std::any_of(
                summaries->compatible_complexes.begin(),
                summaries->compatible_complexes.end(),
                [&](const CompatibleComplexSummary& summary) {
                    return variablesCoverRequest(summary.variables,
                                                 domain,
                                                 target_block);
                });
        case AnalysisSummaryKind::NonlinearTangent:
            return std::any_of(
                summaries->nonlinear_tangents.begin(),
                summaries->nonlinear_tangents.end(),
                [&](const NonlinearTangentSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::SpectralStructure:
            return std::any_of(
                summaries->spectral_structures.begin(),
                summaries->spectral_structures.end(),
                [&](const SpectralStructureSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::ErrorEstimator:
            return std::any_of(
                summaries->error_estimators.begin(),
                summaries->error_estimators.end(),
                [&](const ErrorEstimatorSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::QuadratureAdequacy:
            return std::any_of(
                summaries->quadrature_adequacy.begin(),
                summaries->quadrature_adequacy.end(),
                [&](const QuadratureAdequacySummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::MinimumResidualStability:
            return std::any_of(
                summaries->minimum_residual_stability.begin(),
                summaries->minimum_residual_stability.end(),
                [&](const MinimumResidualStabilitySummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::InitialCompatibility:
            return std::any_of(
                summaries->initial_compatibility.begin(),
                summaries->initial_compatibility.end(),
                [&](const InitialCompatibilitySummary& summary) {
                    const bool scope_ok =
                        !scope_id.empty() &&
                        (summary.compatibility_scope == scope_id ||
                         summary.invariant_set_id == scope_id);
                    if (!scope_ok) {
                        return false;
                    }
                    if (variables.empty()) {
                        return !summary.invariant_domain_variables.empty();
                    }
                    return variableSetCoversAll(
                        summary.invariant_domain_variables,
                        variables);
                });
        case AnalysisSummaryKind::BoundarySymbol:
            return std::any_of(
                summaries->boundary_symbols.begin(),
                summaries->boundary_symbols.end(),
                [&](const BoundarySymbolSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::MeshGeometryQuality:
            return std::any_of(
                summaries->mesh_geometry_quality.begin(),
                summaries->mesh_geometry_quality.end(),
                [&](const MeshGeometryQualitySummary& summary) {
                    return summary.domain == domain &&
                           summary.jacobian_bounds_present;
                });
        case AnalysisSummaryKind::EnergyEntropyBalance:
            return !scope_id.empty() &&
                   std::any_of(summaries->energy_entropy.begin(),
                               summaries->energy_entropy.end(),
                               [&](const EnergyEntropySummary& summary) {
                                   return summary.energy_entropy_id == scope_id ||
                                          summary.energy_functional_id == scope_id;
                               });
        case AnalysisSummaryKind::InvariantDomain:
            return std::any_of(
                summaries->invariant_domains.begin(),
                summaries->invariant_domains.end(),
                [&](const InvariantDomainSummary& summary) {
                    const bool scope_ok =
                        scope_id.empty() ||
                        summary.invariant_set_id == scope_id;
                    return scope_ok &&
                           variablesCoverRequest(summary.variables,
                                                 domain,
                                                 target_block);
                });
        case AnalysisSummaryKind::EquilibriumPreservation:
            return !scope_id.empty() &&
                   std::any_of(
                       summaries->equilibrium_preservation.begin(),
                       summaries->equilibrium_preservation.end(),
                       [&](const EquilibriumPreservationSummary& summary) {
                           return summary.equilibrium_id == scope_id ||
                                  summary.equilibrium_family_id == scope_id;
                       });
        case AnalysisSummaryKind::MovingDomain:
            return !scope_id.empty() &&
                   std::any_of(
                       summaries->moving_domain.begin(),
                       summaries->moving_domain.end(),
                       [&](const MovingDomainSummary& summary) {
                           return summary.constant_state_scope == scope_id ||
                                  summary.gcl_theorem_id == scope_id ||
                                  summary.mesh_update_time_scheme == scope_id;
                       });
        case AnalysisSummaryKind::TransferOperator:
            return !scope_id.empty() &&
                   std::any_of(summaries->transfer_operators.begin(),
                               summaries->transfer_operators.end(),
                               [&](const TransferOperatorSummary& summary) {
                                   return summary.interface_pair_id == scope_id ||
                                          summary.projection_space_id == scope_id;
                               });
        case AnalysisSummaryKind::AdjointConsistency:
            return std::any_of(
                summaries->adjoint_consistency.begin(),
                summaries->adjoint_consistency.end(),
                [&](const AdjointConsistencySummary& summary) {
                    return contribution_id.empty()
                        ? (scope_id.empty() ||
                           summary.goal_functional_id == scope_id)
                        : summary.contribution_id == contribution_id;
                });
        case AnalysisSummaryKind::CoupledSystemStability:
            return std::any_of(
                summaries->coupled_system_stability.begin(),
                summaries->coupled_system_stability.end(),
                [&](const CoupledSystemStabilitySummary& summary) {
                    const bool scope_ok =
                        scope_id.empty() ||
                        summary.coupling_group == scope_id;
                    return scope_ok &&
                           variablesCoverRequest(summary.variables,
                                                 domain,
                                                 target_block);
                });
    }
    return false;
}

void addRequest(AnalysisRequestPlan& plan,
                const ProblemAnalysisContext& context,
                AnalysisSummaryKind kind,
                DomainKind domain,
                const std::vector<VariableKey>& variables,
                std::size_t source_claim_index,
                const PropertyClaim& claim,
                const std::string& reason) {
    auto& request = ensureRequest(plan,
                                  kind,
                                  domain,
                                  variables,
                                  optionalString(claim.tested_block_id),
                                  std::string{},
                                  claimScopeId(claim));
    request.already_available =
        request.already_available ||
        summaryAlreadyAvailable(context,
                                kind,
                                domain,
                                variables,
                                request.block_id,
                                request.contribution_id,
                                request.scope_id);
    request.confidence = strongerConfidence(request.confidence, claim.confidence);
    appendVariables(request.variables, variables);
    populateRequestVariableRoles(request, kind, variables);
    appendUnique(request.source_claim_indices, source_claim_index);
    appendUnique(request.source_claim_kinds, claim.kind);
    appendUnique(request.source_analyzers, sourceAnalyzer(claim));
    appendUnique(request.reasons, reason);
}

void addContextRequest(AnalysisRequestPlan& plan,
                       const ProblemAnalysisContext& context,
                       AnalysisSummaryKind kind,
                       DomainKind domain,
                       const std::vector<VariableKey>& variables,
                       const std::string& reason,
                       AnalysisConfidence confidence = AnalysisConfidence::Medium,
                       const std::string& block_id = {},
                       const std::string& contribution_id = {},
                       const std::string& scope_id = {}) {
    auto& request = ensureRequest(plan,
                                  kind,
                                  domain,
                                  variables,
                                  block_id,
                                  contribution_id,
                                  scope_id);
    request.already_available =
        request.already_available ||
        summaryAlreadyAvailable(context,
                                kind,
                                domain,
                                variables,
                                block_id,
                                contribution_id,
                                scope_id);
    request.confidence = strongerConfidence(request.confidence, confidence);
    appendVariables(request.variables, variables);
    populateRequestVariableRoles(request, kind, variables);
    appendUnique(request.source_analyzers, std::string("ProblemAnalysisContext"));
    appendUnique(request.reasons, reason);
}

bool claimShouldSeedSummaryRequests(const PropertyClaim& claim) noexcept {
    return claim.status == PropertyStatus::Exact ||
           claim.status == PropertyStatus::Likely ||
           claim.status == PropertyStatus::Preserved ||
           claim.status == PropertyStatus::Violated ||
           claim.status == PropertyStatus::Unknown;
}

bool isWeakPenaltyLike(EnforcementKind kind) noexcept {
    return kind == EnforcementKind::WeakPenalty ||
           kind == EnforcementKind::WeakNitsche;
}

bool isNonlocalCouplingContribution(const ContributionDescriptor& contribution) noexcept {
    if (contribution.domain == DomainKind::InterfaceFace ||
        contribution.domain == DomainKind::CoupledBoundary ||
        contribution.domain == DomainKind::Global ||
        contribution.domain == DomainKind::AuxiliaryCoupling) {
        return true;
    }
    if (contribution.role == ContributionRole::GlobalCoupling ||
        contribution.role == ContributionRole::FieldToAuxiliary ||
        contribution.role == ContributionRole::AuxiliaryToField ||
        contribution.role == ContributionRole::AuxiliaryToAuxiliary ||
        contribution.role == ContributionRole::AuxiliarySelf) {
        return true;
    }
    return contribution.balance.has_value() &&
           contribution.balance->role == BalanceRole::ExchangeLike;
}

bool contextHasNonlocalCoupling(const ProblemAnalysisContext& context) noexcept {
    for (const auto& contribution : context.contributions()) {
        if (isNonlocalCouplingContribution(contribution)) {
            return true;
        }
    }
    for (const auto& bc : context.bcDescriptors()) {
        if (bc.domain == DomainKind::InterfaceFace ||
            bc.domain == DomainKind::CoupledBoundary ||
            bc.introduces_global_coupling) {
            return true;
        }
    }
    return false;
}

} // namespace

std::string NumericSummaryPlanner::name() const {
    return "NumericSummaryPlanner";
}

void NumericSummaryPlanner::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const {
    for (const auto& record : context.formulationRecords()) {
        for (const auto& descriptor : record.invariant_domain_descriptors) {
            if (!invariantDescriptorHasSchemeLevelEvidence(descriptor)) {
                continue;
            }
            addContextRequest(
                report.request_plan,
                context,
                AnalysisSummaryKind::InvariantDomain,
                descriptor.domain,
                descriptor.variables,
                "FormulationRecord carries scheme-level invariant-domain metadata; request limiter/CFL/theorem and post-step violation summaries",
                AnalysisConfidence::Medium,
                record.operator_tag,
                {},
                descriptor.invariant_set_id);
        }
    }

    for (std::size_t i = 0; i < report.claims.size(); ++i) {
        const auto& claim = report.claims[i];
        if (!claimShouldSeedSummaryRequests(claim)) {
            continue;
        }

        const auto source = sourceAnalyzer(claim);

        if (claim.kind == PropertyKind::OperatorSymmetry ||
            claim.kind == PropertyKind::OperatorDefiniteness ||
            hasSourceAnalyzer(claim, "OperatorClassAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an elliptic/operator block; request sparse matrix symmetry, sign, row-sum, conditioning, finite tolerance, and theorem-specific M-matrix/SPD diagnostics");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an operator block; request reduced free-free evidence after constraints");
            addRequest(report.request_plan, context, AnalysisSummaryKind::CoefficientProperties,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an elliptic/coercive structure; request coefficient positivity and anisotropy metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::MeshGeometryQuality,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an elliptic operator; request native mesh geometry quality evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::Applicability,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an operator block; request theorem-family applicability gates for DMP, M-matrix, inf-sup, and Schur checks");
            addRequest(report.request_plan, context, AnalysisSummaryKind::NumericalErrorBudget,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an operator block; request conditioning-derived numerical error budget and verification tolerance recommendation");

            if (isScalarFieldClaim(context, claim)) {
                addRequest(report.request_plan, context, AnalysisSummaryKind::LocalStencil,
                           claim.domain, claim.variables, i, claim,
                           source + " classified scalar diffusion; request local stencil sign checks for DMP/Z-matrix/M-matrix monotonicity");
            }
        }

        if (claim.kind == PropertyKind::Nullspace &&
            (hasSourceAnalyzer(claim, "KernelAnalyzer") ||
             hasSourceAnalyzer(claim, "MixedOperatorAnalyzer"))) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected a nullspace; request reduced operator and nullspace-handling evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::NullspaceDegeneracy,
                       claim.domain, claim.variables, i, claim,
                       source + " detected a nullspace; request sparse rank/nullity and constraint-mode degeneracy classification");
        }

        if (claim.kind == PropertyKind::MixedSaddlePoint ||
            hasSourceAnalyzer(claim, "MixedOperatorAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupEstimate,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request numerical inf-sup estimate");
            if (shouldRequestStablePairCertification(report, claim)) {
                addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupPairCertification,
                           claim.domain, claim.variables, i, claim,
                           source + " detected saddle-point structure; request known stable-pair, Fortin, mesh, and domain assumption evidence");
            }
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request constrained/reduced block classification");
            addRequest(report.request_plan, context, AnalysisSummaryKind::SchurComplement,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request Schur complement conditioning, equivalence, and preconditioner evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request block matrix diagnostics and Schur-ready evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::NullspaceDegeneracy,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request nullspace degeneracy and constraint-mode classification");
            addRequest(report.request_plan, context, AnalysisSummaryKind::Applicability,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request theorem-family applicability gates");
            addRequest(report.request_plan, context, AnalysisSummaryKind::RobustnessTrend,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request cross-run trend records for inf-sup, Schur, and conditioning metrics");
        }

        if (claim.kind == PropertyKind::InfSupCondition ||
            hasSourceAnalyzer(claim, "InfSupAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupEstimate,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted an inf-sup claim; request estimate value, scope, and nullspace handling");
            if (shouldRequestStablePairCertification(report, claim)) {
                addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupPairCertification,
                           claim.domain, claim.variables, i, claim,
                           source + " emitted an inf-sup claim; request stable-pair/Fortin certification metadata");
            }
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted an inf-sup claim; request reduced saddle-point block evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::NullspaceDegeneracy,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted an inf-sup claim; request degeneracy evidence before classifying near-zero estimates as violations");
            addRequest(report.request_plan, context, AnalysisSummaryKind::RobustnessTrend,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted an inf-sup claim; request multi-run robustness trend evidence");
        }

        if (claim.kind == PropertyKind::OperatorTransportCharacter ||
            hasSourceAnalyzer(claim, "TransportCharacterAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::TemporalStability,
                       claim.domain, claim.variables, i, claim,
                       source + " detected first-order transport; request finite CFL/eigenvalue-scale, stability-region, norm, operator-scope, nonnormal/pseudospectral, accepted-growth, and time-horizon metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                       claim.domain, claim.variables, i, claim,
                       source + " detected first-order transport; request finite Peclet-like and local transport-diffusion scale metadata with theorem/scope identifiers");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected first-order transport; request finite nonnormality/skew split matrix diagnostics and operator-scope provenance");
            addRequest(report.request_plan, context, AnalysisSummaryKind::InvariantDomain,
                       claim.domain, claim.variables, i, claim,
                       source + " detected transport character; request overshoot/undershoot and invariant-domain evidence when available");
        }

        if (claim.kind == PropertyKind::Stabilization ||
            hasSourceAnalyzer(claim, "StabilizationAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                       claim.domain, claim.variables, i, claim,
                       source + " detected stabilization; request stabilization parameter and scaling metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::StabilizationAdequacy,
                       claim.domain, claim.variables, i, claim,
                       source + " detected stabilization; request parameter-formula, consistency, quantitative Peclet-regime, and CFL-bound adequacy evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected stabilization; request retained matrix diagnostics for consistency and conditioning");
            addRequest(report.request_plan, context, AnalysisSummaryKind::RobustnessTrend,
                       claim.domain, claim.variables, i, claim,
                       source + " detected stabilization; request cross-run stabilization parameter and conditioning trend evidence");
        }

        if (claim.kind == PropertyKind::ConservationStructure ||
            hasSourceAnalyzer(claim, "ConservationAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a conservation/balance claim; request finite declared tolerance plus finite local, global, and interface flux-balance residuals with closure/source/orientation/time-scope metadata");
        }

        if (claim.kind == PropertyKind::DifferentialAlgebraicStructure ||
            hasSourceAnalyzer(claim, "DAEStructureAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::TemporalStability,
                       claim.domain, claim.variables, i, claim,
                       source + " classified temporal/DAE structure; request time scheme stability metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DAEStructureEvidence,
                       claim.domain, claim.variables, i, claim,
                       source + " classified temporal/DAE structure; request mass-rank, algebraic-Jacobian rank, hidden-constraint, regular descriptor-pencil/index theorem, projector, and finite consistent-initialization residual/tolerance evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::InitialCompatibility,
                       claim.domain, claim.variables, i, claim,
                       source + " classified temporal/DAE structure; request initial constraint and boundary residual evidence");
        }

        if (claim.kind == PropertyKind::SpaceCompatibility ||
            hasSourceAnalyzer(claim, "SpaceCompatibilityAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a space/trace compatibility claim; request trace coverage and boundary-symbol evidence");
            if (claim.variables.size() >= 2) {
                addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupEstimate,
                           claim.domain, claim.variables, i, claim,
                           source + " emitted a mixed space compatibility claim; request inf-sup evidence for the space pair");
                if (shouldRequestStablePairCertification(report, claim)) {
                    addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupPairCertification,
                               claim.domain, claim.variables, i, claim,
                               source + " emitted a mixed space compatibility claim; request known stable-pair and Fortin metadata");
                }
            }
        }

        if (claim.kind == PropertyKind::UnderConstraint ||
            claim.kind == PropertyKind::OverConstraint ||
            hasSourceAnalyzer(claim, "ConstraintRankAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a constraint-rank claim; request constrained/free DOF and reduced matrix evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::InitialCompatibility,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a constraint-rank claim; request initial constraint residual evidence for transient/DAE use");
        }

        if (claim.kind == PropertyKind::TopologyScopedKernel ||
            hasSourceAnalyzer(claim, "TopologyScopeAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::MeshGeometryQuality,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted topology-scoped kernel evidence; request mesh region quality and revision metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted topology-scoped kernel evidence; request per-region reduced constraint evidence");
        }

        if (claim.kind == PropertyKind::CompatibilityCondition ||
            hasSourceAnalyzer(claim, "CompatibilityAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InitialCompatibility,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a compatibility condition; request initial/boundary compatibility residual evidence");
        }

        if (claim.kind == PropertyKind::InterfaceCondition ||
            claim.kind == PropertyKind::WeakBoundaryCoercivity ||
            claim.kind == PropertyKind::BoundaryComplementingCondition) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted interface/boundary coercivity evidence; request boundary-symbol and penalty adequacy metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted interface/boundary evidence; request numerical flux-balance metadata");
        }

        if (claim.kind == PropertyKind::CompatibleComplexStructure) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::CompatibleComplex,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted compatible-complex evidence; request exact-sequence, bounded commuting-projection, projection-bound, mesh-family, and shape-regularity metadata");
        }

        if (claim.kind == PropertyKind::TemporalStability) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::TemporalStability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted temporal-stability evidence; request finite CFL, eigenvalue scale, amplification-radius, norm/theorem/scope, time-horizon, and nonnormal/pseudospectral metadata");
        }

        if (claim.kind == PropertyKind::EnergyStability ||
            claim.kind == PropertyKind::EntropyStability) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::EnergyEntropyBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted energy/entropy law evidence; request finite balance/production/tolerance, theorem, functional/norm, coercivity, norm-equivalence, dissipation, entropy-variable, entropy-flux, convexity, and boundary/source accounting summaries");
        }

        if (claim.kind == PropertyKind::DiscreteMaximumPrinciple ||
            claim.kind == PropertyKind::MMatrixStructure ||
            claim.kind == PropertyKind::ZMatrixStructure ||
            claim.kind == PropertyKind::MatrixMonotonicityRisk) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::Applicability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted monotonicity/DMP evidence; request scalar-DMP and M-matrix theorem applicability gates");
        }

        if (claim.kind == PropertyKind::CoefficientPositivity ||
            claim.kind == PropertyKind::ParameterRobustness) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::CoefficientProperties,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted coefficient/parameter robustness evidence; request coefficient spectral bounds and contrast metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted parameter robustness evidence; request nondimensional parameter-scale summaries");
            addRequest(report.request_plan, context, AnalysisSummaryKind::RobustnessTrend,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted parameter robustness evidence; request comparable run trend evidence before certification");
        }

        if (claim.kind == PropertyKind::NonlinearTangentStructure) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::NonlinearTangent,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted nonlinear tangent evidence; request exact/approximate tangent and finite-difference action summaries");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted nonlinear tangent evidence; request tangent block matrix diagnostics");
        }

        if (claim.kind == PropertyKind::LockingRisk) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupEstimate,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted locking/overstiffness evidence; request inf-sup and constraint-space estimates");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted locking/overstiffness evidence; request reduced block conditioning metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted locking/overstiffness evidence; request parameter and resolution scales");
        }

        if (claim.kind == PropertyKind::SpectralCorrectness) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::SpectralStructure,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted spectral correctness evidence; request self-adjointness, compactness, and spurious-mode summaries");
            addRequest(report.request_plan, context, AnalysisSummaryKind::CompatibleComplex,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted spectral correctness evidence; request compatible-complex support where applicable");
        }

        if (claim.kind == PropertyKind::ErrorEstimatorEligibility) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::ErrorEstimator,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted estimator eligibility evidence; request residual, jump, flux-reconstruction, goal-weight, norm scope, PDE class, boundary residual, data oscillation, coefficient/source regularity, shape-regularity, reliability/efficiency, effectivity, and refinement metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted estimator eligibility evidence; request flux-balance residuals used by estimator checks");
        }

        if (claim.kind == PropertyKind::QuadratureAdequacy) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::QuadratureAdequacy,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted quadrature adequacy evidence; request exactness, aliasing, and hourglass-control metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::LocalStencil,
                              claim.domain, claim.variables, i, claim,
                              source + " emitted quadrature adequacy evidence; request local stencil diagnostics for zero-energy modes");
        }

        if (claim.kind == PropertyKind::MinimumResidualStability) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::MinimumResidualStability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted minimum-residual/Petrov-Galerkin evidence; request trial/test space, Riesz map, Fortin/optimal-test, enrichment, residual-control, conditioning estimates, accepted conditioning bounds, and condition-scope metadata");
        }

        if (claim.kind == PropertyKind::InvariantDomainPreservation) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InvariantDomain,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted invariant-domain evidence; request bound, limiter, quantitative CFL/wave-speed, theorem, and post-step violation summaries");
            addRequest(report.request_plan, context, AnalysisSummaryKind::Applicability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted invariant-domain evidence; request operator applicability gate before theorem classification");
        }

        if (claim.kind == PropertyKind::EquilibriumPreservation) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::EquilibriumPreservation,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted equilibrium-preservation evidence; request finite residual/tolerance plus flux/source, reconstruction, quadrature, and boundary-compatibility metadata");
        }

        if (claim.kind == PropertyKind::GeometricConservation) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::MovingDomain,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted moving-domain geometric conservation evidence; request finite mapping Jacobian interval, declared GCL tolerance, theorem, metric-identity, free-stream residual, mesh-update, mesh-velocity, time-integration, and remap summaries");
        }

        if (claim.kind == PropertyKind::TransferOperatorCompatibility) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::TransferOperator,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted transfer-operator evidence; request finite residual/tolerance, projection conservation, constant preservation, interface scope, projection consistency, mortar/dual consistency, interface mass conditioning, action-reaction flux, and rank metadata");
        }

        if (claim.kind == PropertyKind::AdjointConsistency) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::AdjointConsistency,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted adjoint-consistency evidence; request transpose-backend, goal-functional, boundary-adjoint, stabilization-adjoint, goal-linearization, and finite discrete-adjoint residual/tolerance summaries");
        }

        if (claim.kind == PropertyKind::CoupledSystemStructure &&
            hasSourceAnalyzer(claim, "CoupledSystemStabilityAnalyzer") &&
            contextHasNonlocalCoupling(context)) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::CoupledSystemStability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted coupled-system stability evidence; request finite exchange residual, constraint drift, tolerance, partition spectral radius, contraction/coercive/nonnormal bounds, coupled-operator evidence, relaxation, added-mass, and partition metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::TemporalStability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted coupled-system stability evidence; request coupled temporal stability metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted coupled-system stability evidence; request exchange balance summaries");
        }
    }

    for (const auto& contribution : context.contributions()) {
        const auto variables = contribution.test_variables.empty()
            ? contribution.trial_variables
            : contribution.test_variables;

        if (contribution.domain == DomainKind::InteriorFace ||
            contribution.domain == DomainKind::InterfaceFace) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has face contribution '" +
                                  contribution.operator_tag +
                                  "'; request trace/penalty boundary-symbol metadata",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has face contribution '" +
                                  contribution.operator_tag +
                                  "'; request numerical flux and interface-pair balance residuals",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::LocalStencil,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has DG/interface face contribution '" +
                                  contribution.operator_tag +
                                  "'; request local face stencil diagnostics",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
        }

        if (contribution.domain == DomainKind::InterfaceFace) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::TransferOperator,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has interface contribution '" +
                                  contribution.operator_tag +
                                  "'; request projection/mortar transfer compatibility evidence",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
        }

        if (contribution.role == ContributionRole::BoundaryConstraint ||
            contribution.domain == DomainKind::Boundary) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has boundary contribution '" +
                                  contribution.operator_tag +
                                  "'; request weak-boundary coercivity and trace metadata",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
            if (contribution.adjoint_consistency.has_value()) {
                addContextRequest(report.request_plan, context, AnalysisSummaryKind::AdjointConsistency,
                                  contribution.domain, variables,
                                  "ProblemAnalysisContext has boundary contribution '" +
                                      contribution.operator_tag +
                                      "' with adjoint-consistency metadata; request adjoint consistency summary",
                                  AnalysisConfidence::Medium,
                                  contribution.operator_tag,
                                  contribution.contribution_id);
            }
        }
    }

    for (const auto& record : context.formulationRecords()) {
        const auto variables = record.active_variables.empty()
            ? variablesForFields(record.active_fields)
            : record.active_variables;

        bool has_interior_face = record.has_interior_face_terms;
        bool has_interface_face = false;
        bool has_boundary_face = false;
        bool has_cell_terms = false;
        for (DomainKind domain : record.active_domains) {
            has_cell_terms = has_cell_terms || domain == DomainKind::Cell;
            has_interior_face = has_interior_face || domain == DomainKind::InteriorFace;
            has_interface_face = has_interface_face || domain == DomainKind::InterfaceFace;
            has_boundary_face = has_boundary_face || domain == DomainKind::Boundary;
        }

        if (has_cell_terms || record.active_domains.empty()) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::CoefficientProperties,
                              DomainKind::Cell, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has cell operator terms; request coefficient, constitutive, or implicit-form metadata before matrix-based operator classification",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
        }

        if (has_interior_face) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              DomainKind::InteriorFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interior-face/DG terms; request penalty and trace-symbol metadata",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              DomainKind::InteriorFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interior-face/DG terms; request numerical flux-balance summaries",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::LocalStencil,
                              DomainKind::InteriorFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interior-face/DG terms; request local face-stencil diagnostics",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
        }

        if (has_interface_face) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              DomainKind::InterfaceFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interface-face terms; request interface trace and penalty metadata",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              DomainKind::InterfaceFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interface-face terms; request interface flux-balance summaries",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::TransferOperator,
                              DomainKind::InterfaceFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interface-face terms; request transfer/projection compatibility evidence",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
        }

        if (has_boundary_face) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              DomainKind::Boundary, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has boundary-face terms; request boundary-symbol metadata",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
        }
    }

    for (const auto& bc : context.bcDescriptors()) {
        std::vector<VariableKey> variables{bc.primary_variable};
        appendVariables(variables, bc.related_variables);

        if (bc.domain == DomainKind::InterfaceFace) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              DomainKind::InterfaceFace, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' targets an interface; request interface boundary-symbol metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              DomainKind::InterfaceFace, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' targets an interface; request interface flux-balance metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::TransferOperator,
                              DomainKind::InterfaceFace, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' targets an interface; request transfer/projection compatibility metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
        }

        if (isWeakPenaltyLike(bc.enforcement_kind)) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              bc.domain, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' uses weak penalty/Nitsche enforcement; request penalty adequacy and trace metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                              bc.domain, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' uses weak penalty/Nitsche enforcement; request penalty scaling metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              bc.domain, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' uses weak enforcement; request boundary/interface flux-balance metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
        }

        if (bc.adjoint_consistency.has_value()) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::AdjointConsistency,
                              bc.domain, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' provides adjoint-consistency metadata; request adjoint consistency summary",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp

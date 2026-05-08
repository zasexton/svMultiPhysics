/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_ANALYSIS_SUMMARY_MATCHING_H
#define SVMP_FE_ANALYSIS_ANALYSIS_SUMMARY_MATCHING_H

#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryTypes.h"

#include <algorithm>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

struct CertificationGate {
    PropertyStatus status{PropertyStatus::Unknown};
    CertificationClass certification{CertificationClass::Unknown};
    AnalysisConfidence confidence{AnalysisConfidence::Medium};
};

[[nodiscard]] inline CertificationGate certificationGate(
    bool violated,
    bool certified,
    bool numerically_or_structurally_supported) noexcept
{
    if (violated) {
        return {PropertyStatus::Violated,
                CertificationClass::Violated,
                AnalysisConfidence::High};
    }
    if (certified) {
        return {PropertyStatus::Preserved,
                CertificationClass::Certified,
                AnalysisConfidence::High};
    }
    if (numerically_or_structurally_supported) {
        return {PropertyStatus::Likely,
                CertificationClass::NotCertified,
                AnalysisConfidence::Medium};
    }
    return {PropertyStatus::Unknown,
            CertificationClass::NotCertified,
            AnalysisConfidence::Medium};
}

inline void appendUniqueVariable(std::vector<VariableKey>& values,
                                 const VariableKey& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

[[nodiscard]] inline std::vector<VariableKey>
variablesForBlock(const OperatorBlockId& block)
{
    std::vector<VariableKey> variables;
    for (const auto& v : block.test_variables) {
        appendUniqueVariable(variables, v);
    }
    for (const auto& v : block.trial_variables) {
        appendUniqueVariable(variables, v);
    }
    return variables;
}

[[nodiscard]] inline bool variableSetsIntersect(
    const std::vector<VariableKey>& a,
    const std::vector<VariableKey>& b)
{
    for (const auto& av : a) {
        if (std::find(b.begin(), b.end(), av) != b.end()) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool variableSetCoversAll(
    const std::vector<VariableKey>& evidence,
    const std::vector<VariableKey>& target)
{
    if (target.empty()) {
        return true;
    }
    if (evidence.empty()) {
        return false;
    }
    return std::all_of(target.begin(), target.end(),
                       [&](const VariableKey& variable) {
                           return std::find(evidence.begin(),
                                            evidence.end(),
                                            variable) != evidence.end();
                       });
}

[[nodiscard]] inline bool hasBlockScope(const OperatorBlockId& block)
{
    return !block.contribution_id.empty() ||
           !block.operator_tag.empty() ||
           block.marker >= 0 ||
           !block.test_variables.empty() ||
           !block.trial_variables.empty();
}

[[nodiscard]] inline bool contributionIdScopeMatches(
    const OperatorBlockId& evidence_block,
    const OperatorBlockId& target_block,
    bool require_coverage)
{
    if (evidence_block.contribution_id.empty() ||
        target_block.contribution_id.empty() ||
        evidence_block.contribution_id != target_block.contribution_id) {
        return false;
    }

    if (evidence_block.domain != target_block.domain) {
        return false;
    }

    if (evidence_block.marker >= 0 || target_block.marker >= 0) {
        if (evidence_block.marker != target_block.marker) {
            return false;
        }
    }

    if (!evidence_block.operator_tag.empty() &&
        !target_block.operator_tag.empty() &&
        evidence_block.operator_tag != target_block.operator_tag) {
        return false;
    }

    const auto evidence_vars = variablesForBlock(evidence_block);
    const auto target_vars = variablesForBlock(target_block);
    if (require_coverage) {
        return variableSetCoversAll(evidence_vars, target_vars);
    }
    return target_vars.empty() ||
           evidence_vars.empty() ||
           variableSetsIntersect(evidence_vars, target_vars);
}

[[nodiscard]] inline bool blockEvidenceMatches(
    const OperatorBlockId& evidence_block,
    const OperatorBlockId& target_block)
{
    if (!hasBlockScope(evidence_block)) {
        return false;
    }

    if (!evidence_block.contribution_id.empty()) {
        return contributionIdScopeMatches(evidence_block, target_block, false);
    }

    if (!evidence_block.operator_tag.empty()) {
        if (target_block.operator_tag.empty() ||
            evidence_block.operator_tag != target_block.operator_tag) {
            return false;
        }
    }

    if (evidence_block.marker >= 0) {
        if (target_block.marker != evidence_block.marker) {
            return false;
        }
    }

    if (evidence_block.domain != target_block.domain) {
        return false;
    }

    const auto evidence_vars = variablesForBlock(evidence_block);
    if (!evidence_vars.empty()) {
        const auto target_vars = variablesForBlock(target_block);
        if (target_vars.empty() ||
            !variableSetsIntersect(evidence_vars, target_vars)) {
            return false;
        }
    }

    return true;
}

[[nodiscard]] inline bool blockEvidenceCovers(
    const OperatorBlockId& evidence_block,
    const OperatorBlockId& target_block)
{
    if (!hasBlockScope(evidence_block)) {
        return false;
    }

    if (!evidence_block.contribution_id.empty()) {
        return contributionIdScopeMatches(evidence_block, target_block, true);
    }

    if (!target_block.contribution_id.empty()) {
        return false;
    }

    if (!evidence_block.operator_tag.empty()) {
        if (target_block.operator_tag.empty() ||
            evidence_block.operator_tag != target_block.operator_tag) {
            return false;
        }
    } else if (!target_block.operator_tag.empty()) {
        return false;
    }

    if (evidence_block.marker >= 0) {
        if (target_block.marker != evidence_block.marker) {
            return false;
        }
    } else if (target_block.marker >= 0) {
        return false;
    }

    if (evidence_block.domain != target_block.domain) {
        return false;
    }

    return variableSetCoversAll(variablesForBlock(evidence_block),
                                variablesForBlock(target_block));
}

[[nodiscard]] inline bool variableEvidenceMatches(
    const std::vector<VariableKey>& evidence_variables,
    DomainKind evidence_domain,
    const OperatorBlockId& target_block)
{
    if (evidence_variables.empty()) {
        return false;
    }
    if (evidence_domain != target_block.domain) {
        return false;
    }
    const auto target_vars = variablesForBlock(target_block);
    return !target_vars.empty() &&
           variableSetsIntersect(evidence_variables, target_vars);
}

[[nodiscard]] inline bool containsContributionId(
    const std::vector<std::string>& contribution_ids,
    const std::string& contribution_id)
{
    return !contribution_id.empty() &&
           std::find(contribution_ids.begin(), contribution_ids.end(),
                     contribution_id) != contribution_ids.end();
}

[[nodiscard]] inline bool scopedEvidenceMatches(
    const OperatorBlockId& evidence_block,
    const std::vector<VariableKey>& evidence_variables,
    const std::string& evidence_contribution_id,
    const OperatorBlockId& target_block,
    const std::vector<std::string>& target_contribution_ids)
{
    const std::string& explicit_contribution_id =
        !evidence_contribution_id.empty()
            ? evidence_contribution_id
            : evidence_block.contribution_id;
    if (!explicit_contribution_id.empty()) {
        if (!target_block.contribution_id.empty()) {
            return explicit_contribution_id == target_block.contribution_id;
        }
        return containsContributionId(target_contribution_ids,
                                      explicit_contribution_id);
    }
    return blockEvidenceMatches(evidence_block, target_block) ||
           variableEvidenceMatches(evidence_variables,
                                   evidence_block.domain,
                                   target_block);
}

[[nodiscard]] inline bool strictScopedEvidenceMatches(
    const OperatorBlockId& evidence_block,
    const std::vector<VariableKey>& evidence_variables,
    const std::string& evidence_contribution_id,
    const OperatorBlockId& target_block,
    const std::vector<std::string>& target_contribution_ids)
{
    const std::string& explicit_contribution_id =
        !evidence_contribution_id.empty()
            ? evidence_contribution_id
            : evidence_block.contribution_id;
    if (!explicit_contribution_id.empty()) {
        if (!target_block.contribution_id.empty()) {
            return explicit_contribution_id == target_block.contribution_id;
        }
        return containsContributionId(target_contribution_ids,
                                      explicit_contribution_id);
    }
    if (blockEvidenceCovers(evidence_block, target_block)) {
        return true;
    }
    if (evidence_block.domain != target_block.domain) {
        return false;
    }
    const auto target_variables = variablesForBlock(target_block);
    return !target_variables.empty() &&
           variableSetCoversAll(evidence_variables, target_variables);
}

[[nodiscard]] inline bool scopedEvidenceMatches(
    const OperatorBlockId& evidence_block,
    const std::vector<VariableKey>& evidence_variables,
    const std::string& evidence_contribution_id,
    const OperatorBlockId& target_block)
{
    static const std::vector<std::string> empty_contribution_ids;
    return scopedEvidenceMatches(evidence_block,
                                 evidence_variables,
                                 evidence_contribution_id,
                                 target_block,
                                 empty_contribution_ids);
}

[[nodiscard]] inline OperatorBlockId requestTargetBlock(
    const AnalysisSummaryRequest& request)
{
    OperatorBlockId block;
    block.domain = request.domain;
    block.operator_tag = request.block_id;
    block.contribution_id = request.contribution_id;
    block.test_variables = request.test_variables.empty()
        ? request.variables
        : request.test_variables;
    block.trial_variables = request.trial_variables.empty()
        ? request.variables
        : request.trial_variables;
    return block;
}

[[nodiscard]] inline std::vector<std::string> requestContributionIds(
    const AnalysisSummaryRequest& request)
{
    if (request.contribution_id.empty()) {
        return {};
    }
    return {request.contribution_id};
}

[[nodiscard]] inline bool variablesCoverRequest(
    const std::vector<VariableKey>& evidence,
    DomainKind evidence_domain,
    const AnalysisSummaryRequest& request)
{
    if (evidence_domain != request.domain) {
        return false;
    }
    const auto target_block = requestTargetBlock(request);
    return variableSetCoversAll(evidence, variablesForBlock(target_block));
}

[[nodiscard]] inline bool summaryBlockCoversRequest(
    const OperatorBlockId& evidence_block,
    const std::vector<VariableKey>& evidence_variables,
    const std::string& evidence_contribution_id,
    const AnalysisSummaryRequest& request)
{
    const auto target_block = requestTargetBlock(request);
    const auto target_contribution_ids = requestContributionIds(request);
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

[[nodiscard]] inline bool matrixSummaryCoversRequest(
    const DiscreteMatrixSummary& summary,
    const AnalysisSummaryRequest& request)
{
    if (!request.contribution_id.empty() &&
        summary.block.contribution_id != request.contribution_id &&
        std::find(summary.contribution_ids.begin(),
                  summary.contribution_ids.end(),
                  request.contribution_id) == summary.contribution_ids.end()) {
        return false;
    }
    return summaryBlockCoversRequest(summary.block,
                                     variablesForBlock(summary.block),
                                     summary.block.contribution_id,
                                     request);
}

[[nodiscard]] inline bool scopeIdCompatible(const std::string& evidence_scope,
                                            const AnalysisSummaryRequest& request)
{
    return request.scope_id.empty() ||
           evidence_scope.empty() ||
           evidence_scope == request.scope_id;
}

[[nodiscard]] inline bool analysisSummarySetCoversRequest(
    const AnalysisSummarySet& summaries,
    const AnalysisSummaryRequest& request)
{
    const auto target_block = requestTargetBlock(request);
    const auto target_contribution_ids = requestContributionIds(request);
    auto block_covers = [&](const OperatorBlockId& block,
                            const std::vector<VariableKey>& variables,
                            const std::string& contribution_id = {}) {
        return summaryBlockCoversRequest(block,
                                         variables,
                                         contribution_id,
                                         request);
    };

    switch (request.summary_kind) {
        case AnalysisSummaryKind::NormMetadata:
            return std::any_of(
                summaries.norm_metadata.begin(),
                summaries.norm_metadata.end(),
                [&](const NormMetadataSummary& summary) {
                    const auto& summary_scope =
                        summary.scope_id.empty()
                            ? summary.norm_id
                            : summary.scope_id;
                    return scopeIdCompatible(summary_scope, request) &&
                           block_covers(summary.block,
                                        summary.variables,
                                        {});
                });
        case AnalysisSummaryKind::CoefficientProperties:
            return std::any_of(
                summaries.coefficient_properties.begin(),
                summaries.coefficient_properties.end(),
                [&](const CoefficientPropertySummary& summary) {
                    return strictScopedEvidenceMatches(summary.block,
                                                       summary.variables,
                                                       summary.contribution_id,
                                                       target_block,
                                                       target_contribution_ids);
                });
        case AnalysisSummaryKind::DiscreteMatrix:
            return std::any_of(
                summaries.discrete_matrices.begin(),
                summaries.discrete_matrices.end(),
                [&](const DiscreteMatrixSummary& summary) {
                    return matrixSummaryCoversRequest(summary, request);
                });
        case AnalysisSummaryKind::ReducedMatrix:
            return std::any_of(
                summaries.reduced_matrices.begin(),
                summaries.reduced_matrices.end(),
                [&](const ReducedMatrixSummary& summary) {
                    return matrixSummaryCoversRequest(summary.free_free_matrix,
                                                      request);
                });
        case AnalysisSummaryKind::LocalStencil:
            return std::any_of(
                summaries.local_stencils.begin(),
                summaries.local_stencils.end(),
                [&](const LocalStencilSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::MeshGeometryQuality:
            return std::any_of(
                summaries.mesh_geometry_quality.begin(),
                summaries.mesh_geometry_quality.end(),
                [&](const MeshGeometryQualitySummary& summary) {
                    return summary.domain == request.domain &&
                           (summary.jacobian_bounds_present ||
                            summary.shape_regular_evidence_present ||
                            summary.mesh_family_scope_present);
                });
        case AnalysisSummaryKind::FluxBalance:
            return std::any_of(
                summaries.flux_balances.begin(),
                summaries.flux_balances.end(),
                [&](const FluxBalanceSummary& summary) {
                    return block_covers(summary.block, {});
                });
        case AnalysisSummaryKind::TemporalStability:
            return std::any_of(
                summaries.temporal_stability.begin(),
                summaries.temporal_stability.end(),
                [&](const TemporalStabilitySummary& summary) {
                    return block_covers(summary.block,
                                        summary.variables,
                                        summary.contribution_id);
                });
        case AnalysisSummaryKind::BoundarySymbol:
            return std::any_of(
                summaries.boundary_symbols.begin(),
                summaries.boundary_symbols.end(),
                [&](const BoundarySymbolSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::InfSupEstimate:
            return std::any_of(
                summaries.inf_sup_estimates.begin(),
                summaries.inf_sup_estimates.end(),
                [&](const InfSupEstimateSummary& summary) {
                    std::vector<VariableKey> pair{
                        summary.primal_variable,
                        summary.multiplier_variable};
                    return scopeIdCompatible(summary.estimate_scope, request) &&
                           block_covers(summary.block, pair);
                });
        case AnalysisSummaryKind::InfSupPairCertification:
            return std::any_of(
                summaries.inf_sup_pair_certifications.begin(),
                summaries.inf_sup_pair_certifications.end(),
                [&](const InfSupPairCertificationSummary& summary) {
                    std::vector<VariableKey> pair{
                        summary.primal_variable,
                        summary.multiplier_variable};
                    return block_covers(summary.block, pair);
                });
        case AnalysisSummaryKind::EnergyEntropyBalance:
            return !request.scope_id.empty() &&
                   std::any_of(summaries.energy_entropy.begin(),
                               summaries.energy_entropy.end(),
                               [&](const EnergyEntropySummary& summary) {
                                   return summary.energy_entropy_id == request.scope_id ||
                                          summary.energy_functional_id == request.scope_id;
                               });
        case AnalysisSummaryKind::InvariantDomain:
            return std::any_of(
                summaries.invariant_domains.begin(),
                summaries.invariant_domains.end(),
                [&](const InvariantDomainSummary& summary) {
                    const bool scope_ok =
                        request.scope_id.empty() ||
                        summary.invariant_set_id == request.scope_id;
                    return scope_ok &&
                           variablesCoverRequest(summary.variables,
                                                 request.domain,
                                                 request);
                });
        case AnalysisSummaryKind::EquilibriumPreservation:
            return !request.scope_id.empty() &&
                   std::any_of(
                       summaries.equilibrium_preservation.begin(),
                       summaries.equilibrium_preservation.end(),
                       [&](const EquilibriumPreservationSummary& summary) {
                           return summary.equilibrium_id == request.scope_id ||
                                  summary.equilibrium_family_id == request.scope_id;
                       });
        case AnalysisSummaryKind::MovingDomain:
            return !request.scope_id.empty() &&
                   std::any_of(
                       summaries.moving_domain.begin(),
                       summaries.moving_domain.end(),
                       [&](const MovingDomainSummary& summary) {
                           return summary.constant_state_scope == request.scope_id ||
                                  summary.gcl_theorem_id == request.scope_id ||
                                  summary.mesh_update_time_scheme == request.scope_id;
                       });
        case AnalysisSummaryKind::TransferOperator:
            return !request.scope_id.empty() &&
                   std::any_of(summaries.transfer_operators.begin(),
                               summaries.transfer_operators.end(),
                               [&](const TransferOperatorSummary& summary) {
                                   return summary.interface_pair_id == request.scope_id ||
                                          summary.projection_space_id == request.scope_id;
                               });
        case AnalysisSummaryKind::AdjointConsistency:
            return std::any_of(
                summaries.adjoint_consistency.begin(),
                summaries.adjoint_consistency.end(),
                [&](const AdjointConsistencySummary& summary) {
                    return request.contribution_id.empty()
                        ? (request.scope_id.empty() ||
                           summary.goal_functional_id == request.scope_id)
                        : summary.contribution_id == request.contribution_id;
                });
        case AnalysisSummaryKind::ParameterScale:
            return std::any_of(
                summaries.parameter_scales.begin(),
                summaries.parameter_scales.end(),
                [&](const ParameterScaleSummary& summary) {
                    return block_covers(summary.block,
                                        summary.variables,
                                        summary.contribution_id);
                });
        case AnalysisSummaryKind::StabilizationAdequacy:
            return std::any_of(
                summaries.stabilization_adequacy.begin(),
                summaries.stabilization_adequacy.end(),
                [&](const StabilizationAdequacySummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::InitialCompatibility:
            return std::any_of(
                summaries.initial_compatibility.begin(),
                summaries.initial_compatibility.end(),
                [&](const InitialCompatibilitySummary& summary) {
                    const bool scope_ok =
                        request.scope_id.empty()
                            ? (!summary.compatibility_scope.empty() ||
                               !summary.invariant_set_id.empty())
                            : (summary.compatibility_scope == request.scope_id ||
                               summary.invariant_set_id == request.scope_id);
                    if (!scope_ok) {
                        return false;
                    }
                    if (request.variables.empty()) {
                        return !summary.invariant_domain_variables.empty() ||
                               summary.algebraic_constraint_metadata_present ||
                               summary.boundary_constraint_metadata_present ||
                               summary.invariant_domain_metadata_present;
                    }
                    return variableSetCoversAll(
                        summary.invariant_domain_variables,
                        request.variables);
                });
        case AnalysisSummaryKind::CompatibleComplex:
            return std::any_of(
                summaries.compatible_complexes.begin(),
                summaries.compatible_complexes.end(),
                [&](const CompatibleComplexSummary& summary) {
                    return variablesCoverRequest(summary.variables,
                                                 request.domain,
                                                 request);
                });
        case AnalysisSummaryKind::NonlinearTangent:
            return std::any_of(
                summaries.nonlinear_tangents.begin(),
                summaries.nonlinear_tangents.end(),
                [&](const NonlinearTangentSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::SpectralStructure:
            return std::any_of(
                summaries.spectral_structures.begin(),
                summaries.spectral_structures.end(),
                [&](const SpectralStructureSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::ErrorEstimator:
            return std::any_of(
                summaries.error_estimators.begin(),
                summaries.error_estimators.end(),
                [&](const ErrorEstimatorSummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::QuadratureAdequacy:
            return std::any_of(
                summaries.quadrature_adequacy.begin(),
                summaries.quadrature_adequacy.end(),
                [&](const QuadratureAdequacySummary& summary) {
                    return block_covers(summary.block,
                                        variablesForBlock(summary.block));
                });
        case AnalysisSummaryKind::CoupledSystemStability:
            return std::any_of(
                summaries.coupled_system_stability.begin(),
                summaries.coupled_system_stability.end(),
                [&](const CoupledSystemStabilitySummary& summary) {
                    const bool scope_ok =
                        request.scope_id.empty() ||
                        summary.coupling_group == request.scope_id;
                    return scope_ok &&
                           variablesCoverRequest(summary.variables,
                                                 request.domain,
                                                 request);
                });
        case AnalysisSummaryKind::DAEStructureEvidence:
            return std::any_of(
                summaries.dae_structure_evidence.begin(),
                summaries.dae_structure_evidence.end(),
                [&](const DAEStructureEvidenceSummary& summary) {
                    return variablesCoverRequest(summary.variables,
                                                 request.domain,
                                                 request);
                });
        case AnalysisSummaryKind::SchurComplement:
            return std::any_of(
                summaries.schur_complements.begin(),
                summaries.schur_complements.end(),
                [&](const SchurComplementSummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::MinimumResidualStability:
            return std::any_of(
                summaries.minimum_residual_stability.begin(),
                summaries.minimum_residual_stability.end(),
                [&](const MinimumResidualStabilitySummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::NullspaceDegeneracy:
            return std::any_of(
                summaries.nullspace_degeneracies.begin(),
                summaries.nullspace_degeneracies.end(),
                [&](const NullspaceDegeneracySummary& summary) {
                    return block_covers(summary.block,
                                        summary.affected_variables);
                });
        case AnalysisSummaryKind::RobustnessTrend:
            return std::any_of(
                summaries.robustness_trends.begin(),
                summaries.robustness_trends.end(),
                [&](const RobustnessTrendSummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::Applicability:
            return std::any_of(
                summaries.applicability.begin(),
                summaries.applicability.end(),
                [&](const ApplicabilitySummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
        case AnalysisSummaryKind::NumericalErrorBudget:
            return std::any_of(
                summaries.numerical_error_budgets.begin(),
                summaries.numerical_error_budgets.end(),
                [&](const NumericalErrorBudgetSummary& summary) {
                    return block_covers(summary.block, summary.variables);
                });
    }
    return false;
}

[[nodiscard]] inline bool matchesParameterRole(
    const ParameterScaleSummary& summary,
    ParameterScaleRole expected)
{
    if (expected == ParameterScaleRole::Unknown ||
        expected == ParameterScaleRole::Generic) {
        return summary.role == expected ||
               summary.role == ParameterScaleRole::Unknown ||
               summary.role == ParameterScaleRole::Generic;
    }
    return summary.role == expected;
}

[[nodiscard]] inline bool parameterScaleMatches(
    const ParameterScaleSummary& summary,
    const OperatorBlockId& target_block,
    ParameterScaleRole expected,
    const std::vector<std::string>& target_contribution_ids)
{
    return matchesParameterRole(summary, expected) &&
           scopedEvidenceMatches(summary.block,
                                 summary.variables,
                                 summary.contribution_id,
                                 target_block,
                                 target_contribution_ids);
}

[[nodiscard]] inline bool parameterScaleMatches(
    const ParameterScaleSummary& summary,
    const OperatorBlockId& target_block,
    ParameterScaleRole expected)
{
    static const std::vector<std::string> empty_contribution_ids;
    return parameterScaleMatches(summary,
                                 target_block,
                                 expected,
                                 empty_contribution_ids);
}

[[nodiscard]] inline bool parameterScaleCovers(
    const ParameterScaleSummary& summary,
    const OperatorBlockId& target_block,
    ParameterScaleRole expected,
    const std::vector<std::string>& target_contribution_ids)
{
    return matchesParameterRole(summary, expected) &&
           strictScopedEvidenceMatches(summary.block,
                                       summary.variables,
                                       summary.contribution_id,
                                       target_block,
                                       target_contribution_ids);
}

[[nodiscard]] inline bool parameterScaleCovers(
    const ParameterScaleSummary& summary,
    const OperatorBlockId& target_block,
    ParameterScaleRole expected)
{
    static const std::vector<std::string> empty_contribution_ids;
    return parameterScaleCovers(summary,
                                target_block,
                                expected,
                                empty_contribution_ids);
}

[[nodiscard]] inline bool temporalSummaryMatches(
    const TemporalStabilitySummary& summary,
    const OperatorBlockId& target_block,
    const std::vector<std::string>& target_contribution_ids)
{
    return scopedEvidenceMatches(summary.block,
                                 summary.variables,
                                 summary.contribution_id,
                                 target_block,
                                 target_contribution_ids);
}

[[nodiscard]] inline bool temporalSummaryMatches(
    const TemporalStabilitySummary& summary,
    const OperatorBlockId& target_block)
{
    static const std::vector<std::string> empty_contribution_ids;
    return temporalSummaryMatches(summary,
                                  target_block,
                                  empty_contribution_ids);
}

[[nodiscard]] inline bool coefficientSummaryMatches(
    const CoefficientPropertySummary& summary,
    const OperatorBlockId& target_block,
    const std::vector<std::string>& target_contribution_ids)
{
    return scopedEvidenceMatches(summary.block,
                                 summary.variables,
                                 summary.contribution_id,
                                 target_block,
                                 target_contribution_ids);
}

[[nodiscard]] inline bool coefficientSummaryMatches(
    const CoefficientPropertySummary& summary,
    const OperatorBlockId& target_block)
{
    static const std::vector<std::string> empty_contribution_ids;
    return coefficientSummaryMatches(summary,
                                     target_block,
                                     empty_contribution_ids);
}

[[nodiscard]] inline bool coefficientSummaryCovers(
    const CoefficientPropertySummary& summary,
    const OperatorBlockId& target_block,
    const std::vector<std::string>& target_contribution_ids)
{
    return strictScopedEvidenceMatches(summary.block,
                                       summary.variables,
                                       summary.contribution_id,
                                       target_block,
                                       target_contribution_ids);
}

[[nodiscard]] inline bool coefficientSummaryCovers(
    const CoefficientPropertySummary& summary,
    const DiscreteMatrixSummary& target_matrix)
{
    return coefficientSummaryCovers(summary,
                                    target_matrix.block,
                                    target_matrix.contribution_ids);
}

[[nodiscard]] inline bool coefficientSummaryMatches(
    const CoefficientPropertySummary& summary,
    const DiscreteMatrixSummary& target_matrix)
{
    return coefficientSummaryMatches(summary,
                                     target_matrix.block,
                                     target_matrix.contribution_ids);
}

[[nodiscard]] inline bool coefficientEigenvalueBoundsOrdered(
    const CoefficientPropertySummary& summary) noexcept
{
    return numeric::finiteOrdered(summary.min_eigenvalue,
                                  summary.max_eigenvalue);
}

[[nodiscard]] inline bool coefficientLowerBoundMatchesPositivity(
    const CoefficientPropertySummary& summary) noexcept
{
    const Real tol = summary.positivity_tolerance > Real{}
        ? summary.positivity_tolerance
        : Real{1.0e-12};
    if (!coefficientEigenvalueBoundsOrdered(summary)) {
        return false;
    }
    switch (summary.positivity) {
        case PositivityClass::Positive:
            return summary.min_eigenvalue > tol;
        case PositivityClass::Nonnegative:
            return summary.min_eigenvalue >= -tol;
        case PositivityClass::Negative:
            return summary.max_eigenvalue < -tol;
        case PositivityClass::Nonpositive:
            return summary.max_eigenvalue <= tol;
        case PositivityClass::Indefinite:
        case PositivityClass::Unknown:
            return true;
    }
    return false;
}

[[nodiscard]] inline bool coefficientDeclaredBoundContradictsPositivity(
    const CoefficientPropertySummary& summary) noexcept
{
    return summary.lower_bound_valid_for_all_samples &&
           summary.tolerance_metadata_present &&
           coefficientEigenvalueBoundsOrdered(summary) &&
           !coefficientLowerBoundMatchesPositivity(summary);
}

[[nodiscard]] inline bool fluxClosureMetadataComplete(
    const FluxBalanceSummary& summary) noexcept
{
    const bool face_evidence =
        summary.interface_pair_count == 0u ||
        summary.face_pair_residual_evidence_present;
    return summary.flux_variable_metadata_present &&
           summary.element_residual_evidence_present &&
           face_evidence &&
           summary.source_quadrature_consistency_present &&
           summary.orientation_consistency_present &&
           summary.boundary_flux_accounted_for;
}

[[nodiscard]] inline bool fluxBalanceScopeDeclared(
    const FluxBalanceSummary& summary) noexcept
{
    return summary.steady_balance_scope ||
           summary.transient_balance_scope;
}

[[nodiscard]] inline bool fluxTimeScopeComplete(
    const FluxBalanceSummary& summary) noexcept
{
    return summary.steady_balance_scope ||
           (summary.transient_balance_scope &&
            summary.time_update_balance_present);
}

[[nodiscard]] inline bool fluxClosureCertificationMetadataComplete(
    const FluxBalanceSummary& summary) noexcept
{
    return fluxClosureMetadataComplete(summary) &&
           fluxBalanceScopeDeclared(summary) &&
           fluxTimeScopeComplete(summary);
}

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_ANALYSIS_SUMMARY_MATCHING_H

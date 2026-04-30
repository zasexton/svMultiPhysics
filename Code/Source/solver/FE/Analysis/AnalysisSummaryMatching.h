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

[[nodiscard]] inline bool blockEvidenceMatches(
    const OperatorBlockId& evidence_block,
    const OperatorBlockId& target_block)
{
    if (!hasBlockScope(evidence_block)) {
        return false;
    }

    if (!evidence_block.contribution_id.empty()) {
        if (target_block.contribution_id.empty() ||
            evidence_block.contribution_id != target_block.contribution_id) {
            return false;
        }
        return true;
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
        return !target_block.contribution_id.empty() &&
               evidence_block.contribution_id == target_block.contribution_id;
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

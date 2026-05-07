/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ConservationAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void appendUnique(std::vector<VariableKey>& values, const VariableKey& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::vector<VariableKey> blockVariables(const OperatorBlockId& block)
{
    std::vector<VariableKey> variables;
    for (const auto& v : block.test_variables) {
        appendUnique(variables, v);
    }
    for (const auto& v : block.trial_variables) {
        appendUnique(variables, v);
    }
    return variables;
}

bool localVariableSetsIntersect(const std::vector<VariableKey>& a,
                                const std::vector<VariableKey>& b)
{
    for (const auto& av : a) {
        if (std::find(b.begin(), b.end(), av) != b.end()) {
            return true;
        }
    }
    return false;
}

std::vector<VariableKey> contributionVariables(
    const ContributionDescriptor& contribution)
{
    std::vector<VariableKey> variables;
    for (const auto& v : contribution.test_variables) {
        appendUnique(variables, v);
    }
    for (const auto& v : contribution.trial_variables) {
        appendUnique(variables, v);
    }
    return variables;
}

bool hasMatchingSymbolicBalance(const ProblemAnalysisContext& context,
                                const FluxBalanceSummary& summary)
{
    if (summary.symbolic_balance_evidence_present) {
        const auto summary_variables = blockVariables(summary.block);
        const bool group_scoped =
            !summary.symbolic_balance_group.empty() &&
            (summary.balance_group.empty() ||
             summary.symbolic_balance_group == summary.balance_group);
        const bool contribution_scoped =
            !summary.symbolic_balance_contribution_id.empty() &&
            (summary.block.contribution_id.empty() ||
             summary.symbolic_balance_contribution_id ==
                 summary.block.contribution_id);
        const bool block_scoped =
            !summary_variables.empty() &&
            (!summary.block.operator_tag.empty() ||
             !summary.block.contribution_id.empty() ||
             !summary.balance_group.empty());
        if (group_scoped || contribution_scoped || block_scoped) {
            return true;
        }
    }
    const auto summary_variables = blockVariables(summary.block);
    for (const auto& contribution : context.contributions()) {
        if (!contribution.balance.has_value()) {
            continue;
        }
        const auto& balance = *contribution.balance;
        if (!summary.balance_group.empty() &&
            balance.balance_group != summary.balance_group) {
            continue;
        }
        if (summary.balance_group.empty() &&
            !balance.balance_group.empty() &&
            !summary.block.operator_tag.empty() &&
            contribution.operator_tag != summary.block.operator_tag) {
            continue;
        }
        if (contribution.domain != summary.block.domain) {
            continue;
        }
        if (!summary_variables.empty() &&
            !localVariableSetsIntersect(contributionVariables(contribution),
                                        summary_variables)) {
            continue;
        }
        return true;
    }
    return false;
}

void emitFluxBalanceSummaryClaims(const ProblemAnalysisContext& context,
                                  ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->flux_balances) {
        const bool tolerance_declared =
            numeric::finiteDeclaredTolerance(summary.balance_tolerance);
        const Real tol = tolerance_declared
            ? summary.balance_tolerance
            : Real{1.0e-10};
        const bool residuals_finite =
            numeric::finiteAbsResidualTriple(
                summary.local_residual_norm,
                summary.global_residual_norm,
                summary.interface_pair_residual_norm);
        const bool numeric_evidence_valid =
            tolerance_declared && residuals_finite;
        const Real residual = residuals_finite
            ? numeric::maxAbsTriple(summary.local_residual_norm,
                                    summary.global_residual_norm,
                                    summary.interface_pair_residual_norm)
            : Real{};
        const bool violated =
            (numeric_evidence_valid && residual > tol) ||
            summary.local_violation_count > 0u;
        const bool symbolic_balance_present =
            hasMatchingSymbolicBalance(context, summary);
        const bool closure_metadata_complete =
            fluxClosureCertificationMetadataComplete(summary);

        PropertyClaim claim;
        claim.kind = PropertyKind::ConservationStructure;
        if (violated) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.conservation_class = ConservationClass::ClosureBroken;
        } else if (numeric_evidence_valid && symbolic_balance_present &&
                   closure_metadata_complete) {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.conservation_class = summary.interface_pair_count > 0u
                ? ConservationClass::ExchangeBalanced
                : ConservationClass::LocalClosureExpected;
        } else if (numeric_evidence_valid && symbolic_balance_present) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.conservation_class = summary.interface_pair_count > 0u
                ? ConservationClass::PotentialExchangeBalance
                : ConservationClass::GlobalClosureExpected;
        } else {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.conservation_class = ConservationClass::Unknown;
        }
        claim.domain = summary.block.domain;
        claim.variables = blockVariables(summary.block);
        claim.local_balance_residual = summary.local_residual_norm;
        claim.global_balance_residual = summary.global_residual_norm;
        claim.interface_balance_residual = summary.interface_pair_residual_norm;
        claim.flux_balance_residual = residual;
        claim.tested_block_id = summary.block.operator_tag;
        if (violated) {
            claim.description =
                "Numeric flux-balance summary violates the declared tolerance";
        } else if (numeric_evidence_valid && symbolic_balance_present &&
                   closure_metadata_complete) {
            claim.description =
                "Numeric flux-balance summary satisfies the declared tolerance with matching symbolic, flux, source, orientation, and steady/transient time-scope closure evidence";
        } else if (numeric_evidence_valid && symbolic_balance_present) {
            claim.description =
                "Numeric flux-balance summary satisfies residual checks but lacks complete flux/source/orientation or steady/transient time-scope closure metadata";
        } else {
            claim.description =
                "Numeric flux-balance summary lacks finite declared tolerance, finite residuals, or matching symbolic balance evidence";
        }
        claim.claim_origin = "ConservationAnalyzer";
        claim.addEvidence("ConservationAnalyzer",
            "FluxBalanceSummary local=" +
            std::to_string(summary.local_residual_norm) +
            ", global=" +
            std::to_string(summary.global_residual_norm) +
            ", interface=" +
            std::to_string(summary.interface_pair_residual_norm) +
            ", tolerance=" + std::to_string(tol) +
            ", local_violations=" +
            std::to_string(summary.local_violation_count) +
            ", tolerance_declared=" +
            std::string(tolerance_declared ? "true" : "false") +
            ", finite_residuals=" +
            std::string(residuals_finite ? "true" : "false") +
            ", symbolic_balance=" +
            std::string(symbolic_balance_present ? "true" : "false") +
            ", closure_metadata=" +
            std::string(closure_metadata_complete ? "true" : "false") +
            ", steady_scope=" +
            std::string(summary.steady_balance_scope ? "true" : "false") +
            ", transient_scope=" +
            std::string(summary.transient_balance_scope ? "true" : "false") +
            ", time_update_balance=" +
            std::string(summary.time_update_balance_present ? "true" : "false"));
        if (violated) {
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Warning;
            issue.message =
                "Flux-balance residual exceeds tolerance for block '" +
                summary.block.operator_tag + "'";
            report.issues.push_back(std::move(issue));
        } else if (!numeric_evidence_valid) {
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Warning;
            issue.message =
                "Flux-balance summary for block '" +
                summary.block.operator_tag +
                "' has non-finite residual evidence or invalid tolerance";
            report.issues.push_back(std::move(issue));
        }
        report.claims.push_back(std::move(claim));
    }
}

} // namespace

std::string ConservationAnalyzer::name() const {
    return "ConservationAnalyzer";
}

void ConservationAnalyzer::run(const ProblemAnalysisContext& context,
                               ProblemAnalysisReport& report) const
{
    const auto& contributions = context.contributions();
    emitFluxBalanceSummaryClaims(context, report);

    // =====================================================================
    // Group contributions by balance_group
    // =====================================================================

    struct BalanceGroupEntry {
        const ContributionDescriptor* contrib;
        BalanceDescriptor balance;
    };

    std::unordered_map<std::string, std::vector<BalanceGroupEntry>> groups;

    for (const auto& contrib : contributions) {
        if (!contrib.balance.has_value()) continue;
        const auto& bd = *contrib.balance;
        if (bd.balance_group.empty()) continue;

        BalanceGroupEntry entry;
        entry.contrib = &contrib;
        entry.balance = bd;
        groups[bd.balance_group].push_back(std::move(entry));
    }

    // No balance descriptors populated => graceful no-op
    if (groups.empty()) return;

    // =====================================================================
    // Analyze each balance group
    // =====================================================================

    for (const auto& [group_name, entries] : groups) {
        // Count contributions by sign and role
        int positive_count = 0;
        int negative_count = 0;
        bool has_local_closure_expected = false;
        bool has_boundary_flux = false;

        // Collect variables involved
        std::vector<VariableKey> involved_vars;

        for (const auto& entry : entries) {
            if (entry.balance.sign > 0) {
                ++positive_count;
            } else if (entry.balance.sign < 0) {
                ++negative_count;
            }

            if (entry.balance.role == BalanceRole::FluxLike) {
                if (entry.contrib->domain == DomainKind::Boundary ||
                    entry.contrib->domain == DomainKind::CoupledBoundary) {
                    has_boundary_flux = true;
                }
            }

            if (entry.balance.local_closure_expected) {
                has_local_closure_expected = true;
            }

            for (const auto& tv : entry.contrib->test_variables) {
                involved_vars.push_back(tv);
            }
            for (const auto& tv : entry.contrib->trial_variables) {
                involved_vars.push_back(tv);
            }
        }

        // Check for local closure: if local_closure_expected and the field
        // supports it (e.g. HDiv flux)
        if (has_local_closure_expected) {
            bool field_supports_closure = false;

            for (const auto& entry : entries) {
                for (const auto& tv : entry.contrib->test_variables) {
                    if (tv.kind == VariableKind::FieldComponent) {
                        const auto* fd = context.fieldDescriptor(tv.field_id);
                        if (fd && fd->supports_local_balance_closure) {
                            field_supports_closure = true;
                            break;
                        }
                    }
                }
                if (field_supports_closure) break;
            }

            if (field_supports_closure) {
                PropertyClaim claim;
                claim.kind = PropertyKind::ConservationStructure;
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::NotCertified;
                claim.conservation_class = ConservationClass::LocalClosureExpected;
                claim.variables = involved_vars;
                claim.description =
                    "Local conservation closure expected for balance group '" +
                    group_name + "' (field supports local balance closure, but exact closure requires flux/source residual metadata)";
                claim.claim_origin = "ConservationAnalyzer";
                claim.addEvidence("ConservationAnalyzer",
                    "local_closure_expected=true and field has"
                    " supports_local_balance_closure=true",
                    AnalysisConfidence::Medium);
                report.claims.push_back(std::move(claim));
                continue;
            }
        }

        // Check if sources balance sinks (positive and negative contributions match)
        if (positive_count > 0 && negative_count > 0) {
            PropertyClaim claim;
            claim.kind = PropertyKind::ConservationStructure;
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.conservation_class = ConservationClass::PotentialExchangeBalance;
            claim.variables = involved_vars;
            claim.description =
                "Potential exchange balance for balance group '" + group_name +
                "': " + std::to_string(positive_count) + " positive and " +
                std::to_string(negative_count) + " negative contributions; "
                "closure requires flux/source residual metadata";
            claim.claim_origin = "ConservationAnalyzer";
            claim.addEvidence("ConservationAnalyzer",
                "Both positive and negative signed contributions present, but residual closure was not certified",
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(claim));
            continue;
        }

        // Flux contributions on boundaries but no matching conservation structure.
        // Require at least 2 entries in the group: a single boundary flux
        // contribution is just an ordinary Neumann BC, not a broken conservation
        // law. ClosureBroken is only meaningful when multiple flux-like
        // contributions form a balance group without closure structure.
        if (has_boundary_flux && !has_local_closure_expected &&
            entries.size() >= 2) {
            PropertyClaim claim;
            claim.kind = PropertyKind::ConservationStructure;
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Low;
            claim.certification_class = CertificationClass::NotCertified;
            claim.conservation_class = ConservationClass::Unknown;
            claim.variables = involved_vars;
            claim.description =
                "Conservation closure unresolved for balance group '" +
                group_name + "': boundary flux contributions present but no"
                " matching local closure structure";
            claim.claim_origin = "ConservationAnalyzer";
            claim.addEvidence("ConservationAnalyzer",
                "Boundary flux contributions exist without local_closure_expected",
                AnalysisConfidence::Low);
            report.claims.push_back(std::move(claim));
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp

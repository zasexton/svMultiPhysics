/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ConservationAnalyzer.h"
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

void emitFluxBalanceSummaryClaims(const ProblemAnalysisContext& context,
                                  ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& summary : summaries->flux_balances) {
        const Real tol = summary.balance_tolerance > Real{}
            ? summary.balance_tolerance
            : Real{1.0e-10};
        const Real residual = std::max({
            std::abs(summary.local_residual_norm),
            std::abs(summary.global_residual_norm),
            std::abs(summary.interface_pair_residual_norm)});
        const bool violated =
            residual > tol ||
            summary.local_violation_count > 0u;

        PropertyClaim claim;
        claim.kind = PropertyKind::ConservationStructure;
        claim.status = violated ? PropertyStatus::Violated
                                : PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::High;
        claim.certification_class = violated ? CertificationClass::Violated
                                             : CertificationClass::Certified;
        claim.conservation_class = violated
            ? ConservationClass::ClosureBroken
            : (summary.interface_pair_count > 0u
                ? ConservationClass::ExchangeBalanced
                : ConservationClass::GlobalClosureExpected);
        claim.domain = summary.block.domain;
        claim.variables = blockVariables(summary.block);
        claim.local_balance_residual = summary.local_residual_norm;
        claim.global_balance_residual = summary.global_residual_norm;
        claim.interface_balance_residual = summary.interface_pair_residual_norm;
        claim.flux_balance_residual = residual;
        claim.tested_block_id = summary.block.operator_tag;
        claim.description = violated
            ? "Numeric flux-balance summary violates the declared tolerance"
            : "Numeric flux-balance summary satisfies the declared tolerance";
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
            std::to_string(summary.local_violation_count));
        if (violated) {
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Warning;
            issue.message =
                "Flux-balance residual exceeds tolerance for block '" +
                summary.block.operator_tag + "'";
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
                claim.status = PropertyStatus::Exact;
                claim.confidence = AnalysisConfidence::High;
                claim.conservation_class = ConservationClass::LocalClosureExpected;
                claim.variables = involved_vars;
                claim.description =
                    "Local conservation closure expected for balance group '" +
                    group_name + "' (field supports local balance closure)";
                claim.claim_origin = "ConservationAnalyzer";
                claim.addEvidence("ConservationAnalyzer",
                    "local_closure_expected=true and field has"
                    " supports_local_balance_closure=true");
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
            claim.conservation_class = ConservationClass::ExchangeBalanced;
            claim.variables = involved_vars;
            claim.description =
                "Exchange balanced for balance group '" + group_name +
                "': " + std::to_string(positive_count) + " positive and " +
                std::to_string(negative_count) + " negative contributions";
            claim.claim_origin = "ConservationAnalyzer";
            claim.addEvidence("ConservationAnalyzer",
                "Both positive and negative signed contributions present",
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
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Low;
            claim.conservation_class = ConservationClass::ClosureBroken;
            claim.variables = involved_vars;
            claim.description =
                "Conservation closure potentially broken for balance group '" +
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

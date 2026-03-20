/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ConservationAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

std::string ConservationAnalyzer::name() const {
    return "ConservationAnalyzer";
}

void ConservationAnalyzer::run(const ProblemAnalysisContext& context,
                               ProblemAnalysisReport& report) const
{
    const auto& contributions = context.contributions();

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
        bool has_flux = false;
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
                has_flux = true;
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

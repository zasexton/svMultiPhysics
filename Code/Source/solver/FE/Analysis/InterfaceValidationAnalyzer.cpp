/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/InterfaceValidationAnalyzer.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/InterfaceTopologyContext.h"

#include <algorithm>
#include <cmath>
#include <set>
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

bool isFaceOrBoundary(DomainKind domain) noexcept
{
    return domain == DomainKind::Boundary ||
           domain == DomainKind::InteriorFace ||
           domain == DomainKind::InterfaceFace ||
           domain == DomainKind::CoupledBoundary;
}

void emitBoundaryAndFluxEvidence(const ProblemAnalysisContext& context,
                                 ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    Real max_penalty_scale = Real{};
    bool has_penalty_scale = false;
    for (const auto& scale : summaries->parameter_scales) {
        if (!has_penalty_scale || scale.max_scale_value > max_penalty_scale) {
            max_penalty_scale = scale.max_scale_value;
            has_penalty_scale = true;
        }
    }

    for (const auto& boundary : summaries->boundary_symbols) {
        if (boundary.complementing_condition_satisfied.has_value()) {
            const bool satisfied = *boundary.complementing_condition_satisfied;
            PropertyClaim claim;
            claim.kind = PropertyKind::BoundaryComplementingCondition;
            claim.status = satisfied ? PropertyStatus::Preserved
                                     : PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = satisfied ? CertificationClass::Certified
                                                  : CertificationClass::Violated;
            claim.boundary_complementing_condition_satisfied = satisfied;
            claim.domain = boundary.block.domain;
            claim.variables = blockVariables(boundary.block);
            claim.tested_block_id = boundary.block.operator_tag;
            claim.description = satisfied
                ? "Boundary-symbol summary satisfies the complementing condition"
                : "Boundary-symbol summary violates the complementing condition";
            claim.claim_origin = "InterfaceValidationAnalyzer";
            claim.addEvidence("InterfaceValidationAnalyzer",
                "BoundarySymbolSummary principal_order=" +
                std::to_string(boundary.principal_operator_order) +
                ", boundary_order=" +
                std::to_string(boundary.boundary_operator_order) +
                ", evidence_scope='" + boundary.evidence_scope + "'");
            report.claims.push_back(std::move(claim));
        }

        if (isFaceOrBoundary(boundary.block.domain) && has_penalty_scale) {
            const bool adequate = max_penalty_scale >= Real{1};
            PropertyClaim claim;
            claim.kind = PropertyKind::WeakBoundaryCoercivity;
            claim.status = adequate ? PropertyStatus::Preserved
                                    : PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = adequate ? CertificationClass::Certified
                                                 : CertificationClass::Violated;
            claim.domain = boundary.block.domain;
            claim.variables = blockVariables(boundary.block);
            claim.penalty_scale = max_penalty_scale;
            claim.weak_coercivity_lower_bound = max_penalty_scale - Real{1};
            claim.tested_block_id = boundary.block.operator_tag;
            claim.description = adequate
                ? "Weak-boundary penalty scale is adequate for boundary-symbol evidence"
                : "Weak-boundary penalty scale is below the generic coercivity threshold";
            claim.claim_origin = "InterfaceValidationAnalyzer";
            claim.addEvidence("InterfaceValidationAnalyzer",
                "BoundarySymbolSummary paired with ParameterScaleSummary max_scale=" +
                std::to_string(max_penalty_scale),
                AnalysisConfidence::Medium);
            report.claims.push_back(std::move(claim));
        }
    }

    for (const auto& flux : summaries->flux_balances) {
        if (!isFaceOrBoundary(flux.block.domain)) continue;
        const Real tol = flux.balance_tolerance > Real{}
            ? flux.balance_tolerance
            : Real{1.0e-10};
        const Real residual = std::max({
            std::abs(flux.local_residual_norm),
            std::abs(flux.global_residual_norm),
            std::abs(flux.interface_pair_residual_norm)});
        const bool balanced =
            residual <= tol &&
            flux.local_violation_count == 0u;

        PropertyClaim claim;
        claim.kind = PropertyKind::InterfaceCondition;
        claim.status = balanced ? PropertyStatus::Preserved
                                : PropertyStatus::Violated;
        claim.confidence = AnalysisConfidence::High;
        claim.certification_class = balanced ? CertificationClass::Certified
                                             : CertificationClass::Violated;
        claim.domain = flux.block.domain;
        claim.variables = blockVariables(flux.block);
        claim.interface_balance_residual = flux.interface_pair_residual_norm;
        claim.flux_balance_residual = residual;
        claim.tested_block_id = flux.block.operator_tag;
        claim.description = balanced
            ? "Boundary/interface flux summary is balanced within tolerance"
            : "Boundary/interface flux summary violates the declared tolerance";
        claim.claim_origin = "InterfaceValidationAnalyzer";
        claim.addEvidence("InterfaceValidationAnalyzer",
            "FluxBalanceSummary residual=" +
            std::to_string(residual) +
            ", tolerance=" + std::to_string(tol));
        report.claims.push_back(std::move(claim));
    }
}

} // namespace

std::string InterfaceValidationAnalyzer::name() const {
    return "InterfaceValidationAnalyzer";
}

void InterfaceValidationAnalyzer::run(const ProblemAnalysisContext& context,
                                       ProblemAnalysisReport& report) const
{
    const auto* itopo = context.interfaceTopologyContext();
    const auto& contributions = context.contributions();

    emitBoundaryAndFluxEvidence(context, report);

    // Collect all interface-face contributions and their markers
    std::set<int> specific_markers_referenced;
    bool has_wildcard_contribution = false;
    bool has_any_interface_contribution = false;

    for (const auto& c : contributions) {
        if (c.domain != DomainKind::InterfaceFace) continue;
        has_any_interface_contribution = true;

        if (c.interface_scope == InterfaceScope::AllRegisteredInterfaces) {
            has_wildcard_contribution = true;
        } else if (c.interface_marker >= 0) {
            specific_markers_referenced.insert(c.interface_marker);
        }
    }

    // Also check formulation records for interface integrals (.dI(marker))
    for (const auto& rec : context.formulationRecords()) {
        for (const auto& dom : rec.active_domains) {
            if (dom == DomainKind::InterfaceFace) {
                has_any_interface_contribution = true;
            }
        }
    }

    // If no interface contributions, nothing to validate
    if (!has_any_interface_contribution) return;

    bool post_setup = (itopo != nullptr);

    if (post_setup) {
        // --- Post-setup: strict validation ---
        auto registered_markers = itopo->markers();

        // Check SpecificMarker contributions against registered meshes
        for (int marker : specific_markers_referenced) {
            if (!itopo->hasMarker(marker)) {
                AnalysisIssue issue;
                issue.severity = IssueSeverity::Error;
                issue.message =
                    "Interface contribution references marker " +
                    std::to_string(marker) +
                    " but no InterfaceMesh is registered for that marker";
                report.issues.push_back(std::move(issue));
            }
        }

        // Check AllRegisteredInterfaces contributions
        if (has_wildcard_contribution && itopo->empty()) {
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Error;
            issue.message =
                "Interface contribution uses AllRegisteredInterfaces scope "
                "but no InterfaceMesh objects are registered";
            report.issues.push_back(std::move(issue));
        }

        // Check for unused InterfaceMesh objects
        for (int reg_marker : registered_markers) {
            bool targeted = specific_markers_referenced.count(reg_marker) > 0;
            // Wildcard contributions target all registered meshes
            if (has_wildcard_contribution) targeted = true;

            if (!targeted) {
                AnalysisIssue issue;
                issue.severity = IssueSeverity::Info;
                issue.message =
                    "InterfaceMesh registered for marker " +
                    std::to_string(reg_marker) +
                    " but no interface contribution targets it";
                report.issues.push_back(std::move(issue));
            }
        }
    } else {
        // --- Pre-setup: provisional warnings ---
        // Interface topology is not yet available. Accept contributions
        // but emit low-confidence warnings for known-marker references.
        for (int marker : specific_markers_referenced) {
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Warning;
            issue.message =
                "Interface contribution references marker " +
                std::to_string(marker) +
                " but interface topology is not yet available "
                "(will be validated after setup)";
            report.issues.push_back(std::move(issue));
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp

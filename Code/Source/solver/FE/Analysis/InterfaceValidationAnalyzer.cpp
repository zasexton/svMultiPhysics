/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/InterfaceValidationAnalyzer.h"
#include "Analysis/AnalysisSummaryMatching.h"
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

bool isFaceOrBoundary(DomainKind domain) noexcept
{
    return domain == DomainKind::Boundary ||
           domain == DomainKind::InteriorFace ||
           domain == DomainKind::InterfaceFace ||
           domain == DomainKind::CoupledBoundary;
}

bool boundaryComplementingCertificateComplete(
    const BoundarySymbolSummary& boundary) noexcept
{
    const bool count_ok =
        boundary.required_boundary_condition_count == 0u ||
        boundary.boundary_condition_count >=
            boundary.required_boundary_condition_count;
    const bool margin_ok =
        boundary.complementing_margin_present &&
        std::isfinite(static_cast<double>(boundary.complementing_margin)) &&
        boundary.complementing_margin > Real{};
    const bool lopatinskii_symbol_evidence =
        boundary.tangential_frequency_coverage_present &&
        boundary.decaying_root_count_evidence_present &&
        boundary.stable_subspace_dimension_evidence_present &&
        boundary.parameter_ellipticity_evidence_present &&
        margin_ok &&
        !boundary.complementing_theorem_id.empty() &&
        boundary.root_subspace_mismatch_count == 0u;
    return boundary.principal_symbol_rank_evidence_present &&
           boundary.boundary_symbol_rank_evidence_present &&
           lopatinskii_symbol_evidence &&
           boundary.component_coverage_complete &&
           boundary.dof_coverage_complete &&
           boundary.missing_symbol_count == 0u &&
           count_ok;
}

void emitBoundaryAndFluxEvidence(const ProblemAnalysisContext& context,
                                 ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) return;

    for (const auto& boundary : summaries->boundary_symbols) {
        if (boundary.complementing_condition_satisfied.has_value()) {
            const bool satisfied = *boundary.complementing_condition_satisfied;
            const bool symbol_mismatch =
                boundary.root_subspace_mismatch_count > 0u;
            const bool certificate_complete =
                satisfied && !symbol_mismatch &&
                boundaryComplementingCertificateComplete(boundary);
            PropertyClaim claim;
            claim.kind = PropertyKind::BoundaryComplementingCondition;
            claim.status = (!satisfied || symbol_mismatch)
                ? PropertyStatus::Violated
                : (certificate_complete ? PropertyStatus::Preserved
                                        : PropertyStatus::Likely);
            claim.confidence = certificate_complete || !satisfied || symbol_mismatch
                ? AnalysisConfidence::High
                : AnalysisConfidence::Medium;
            claim.certification_class = (!satisfied || symbol_mismatch)
                ? CertificationClass::Violated
                : (certificate_complete ? CertificationClass::Certified
                                        : CertificationClass::NotCertified);
            claim.boundary_complementing_condition_satisfied = satisfied;
            claim.domain = boundary.block.domain;
            claim.variables = variablesForBlock(boundary.block);
            claim.tested_block_id = boundary.block.operator_tag;
            claim.description = !satisfied
                ? "Boundary-symbol summary violates the complementing condition"
                : (symbol_mismatch
                    ? "Boundary-symbol summary reports inconsistent decaying-root and stable-subspace evidence"
                : (certificate_complete
                    ? "Boundary-symbol summary satisfies the complementing condition with rank, count, tangential-frequency, root/subspace, margin, component, and DOF coverage evidence"
                    : "Boundary-symbol summary reports the complementing condition but lacks complete rank, count, tangential-frequency, root/subspace, margin, component, or DOF coverage evidence"));
            claim.claim_origin = "InterfaceValidationAnalyzer";
            claim.addEvidence("InterfaceValidationAnalyzer",
                "BoundarySymbolSummary principal_order=" +
                std::to_string(boundary.principal_operator_order) +
                ", boundary_order=" +
                std::to_string(boundary.boundary_operator_order) +
                ", evidence_scope='" + boundary.evidence_scope +
                "', principal_rank=" +
                std::string(boundary.principal_symbol_rank_evidence_present ? "true" : "false") +
                ", boundary_rank=" +
                std::string(boundary.boundary_symbol_rank_evidence_present ? "true" : "false") +
                ", component_coverage=" +
                std::string(boundary.component_coverage_complete ? "true" : "false") +
                ", dof_coverage=" +
                std::string(boundary.dof_coverage_complete ? "true" : "false") +
                ", tangential_frequency=" +
                std::string(boundary.tangential_frequency_coverage_present ? "true" : "false") +
                ", decaying_roots=" +
                std::string(boundary.decaying_root_count_evidence_present ? "true" : "false") +
                ", stable_subspace=" +
                std::string(boundary.stable_subspace_dimension_evidence_present ? "true" : "false") +
                ", parameter_ellipticity=" +
                std::string(boundary.parameter_ellipticity_evidence_present ? "true" : "false") +
                ", margin=" +
                std::to_string(boundary.complementing_margin) +
                ", theorem='" + boundary.complementing_theorem_id + "'" +
                ", root_subspace_mismatches=" +
                std::to_string(boundary.root_subspace_mismatch_count) +
                ", missing_symbols=" +
                std::to_string(boundary.missing_symbol_count));
            report.claims.push_back(std::move(claim));
        }

        if (isFaceOrBoundary(boundary.block.domain)) {
            Real max_penalty_scale = Real{};
            Real required_lower_bound = Real{};
            bool has_penalty_scale = false;
            bool has_required_lower_bound = false;
            bool has_trace_metadata = false;
            for (const auto& scale : summaries->parameter_scales) {
                if (!parameterScaleMatches(
                        scale, boundary.block,
                        ParameterScaleRole::WeakBoundaryPenalty)) {
                    continue;
                }
                if (!has_penalty_scale ||
                    scale.max_scale_value > max_penalty_scale) {
                    max_penalty_scale = scale.max_scale_value;
                    has_penalty_scale = true;
                }
                if (scale.required_lower_bound_present &&
                    (!has_required_lower_bound ||
                     scale.required_lower_bound > required_lower_bound)) {
                    required_lower_bound = scale.required_lower_bound;
                    has_required_lower_bound = true;
                }
                has_trace_metadata =
                    has_trace_metadata || scale.trace_inverse_metadata_present;
            }

            PropertyClaim claim;
            claim.kind = PropertyKind::WeakBoundaryCoercivity;
            claim.domain = boundary.block.domain;
            claim.variables = variablesForBlock(boundary.block);
            claim.tested_block_id = boundary.block.operator_tag;
            claim.claim_origin = "InterfaceValidationAnalyzer";
            if (has_penalty_scale) {
                claim.penalty_scale = max_penalty_scale;
                if (has_required_lower_bound) {
                    const bool adequate =
                        max_penalty_scale + Real{1.0e-14} >= required_lower_bound;
                    if (adequate && has_trace_metadata) {
                        claim.status = PropertyStatus::Preserved;
                        claim.confidence = AnalysisConfidence::High;
                        claim.certification_class = CertificationClass::Certified;
                    } else if (adequate) {
                        claim.status = PropertyStatus::Unknown;
                        claim.confidence = AnalysisConfidence::Medium;
                        claim.certification_class = CertificationClass::NotCertified;
                    } else {
                        claim.status = PropertyStatus::Violated;
                        claim.confidence = AnalysisConfidence::High;
                        claim.certification_class = CertificationClass::Violated;
                    }
                    claim.weak_coercivity_lower_bound =
                        max_penalty_scale - required_lower_bound;
                    claim.description = adequate
                        ? (has_trace_metadata
                              ? "Scoped weak-boundary penalty scale satisfies the trace-backed coercivity lower bound"
                              : "Scoped weak-boundary penalty scale satisfies the reported lower bound but lacks trace/inverse metadata")
                        : "Scoped weak-boundary penalty scale is below the reported coercivity lower bound";
                    claim.addEvidence("InterfaceValidationAnalyzer",
                        "ParameterScaleSummary max_scale=" +
                        std::to_string(max_penalty_scale) +
                        ", required_lower_bound=" +
                        std::to_string(required_lower_bound) +
                        ", trace_inverse_metadata=" +
                        std::string(has_trace_metadata ? "true" : "false"),
                        claim.confidence);
                } else {
                    claim.status = PropertyStatus::Unknown;
                    claim.confidence = AnalysisConfidence::Medium;
                    claim.certification_class = CertificationClass::NotCertified;
                    claim.description =
                        "Weak-boundary penalty scale is present but lacks a theorem-specific coercivity lower bound";
                    claim.addEvidence("InterfaceValidationAnalyzer",
                        "ParameterScaleSummary max_scale=" +
                        std::to_string(max_penalty_scale) +
                        " without required_lower_bound metadata",
                        AnalysisConfidence::Medium);
                }
            } else {
                claim.status = PropertyStatus::Unknown;
                claim.confidence = AnalysisConfidence::Medium;
                claim.certification_class = CertificationClass::Unknown;
                claim.description =
                    "Weak-boundary coercivity is unknown because no scoped penalty-scale summary matched this boundary";
                claim.addEvidence("InterfaceValidationAnalyzer",
                    "BoundarySymbolSummary had no matching WeakBoundaryPenalty ParameterScaleSummary",
                    AnalysisConfidence::Medium);
            }
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
        claim.variables = variablesForBlock(flux.block);
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

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/InterfaceValidationAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
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

bool isWeakPenaltyLike(EnforcementKind kind) noexcept
{
    return kind == EnforcementKind::WeakPenalty ||
           kind == EnforcementKind::WeakNitsche;
}

bool isWeakBoundaryCoercivityRoute(
    WeakBoundaryEnforcementRoute route) noexcept
{
    return route == WeakBoundaryEnforcementRoute::PenaltyCoercivity ||
           route == WeakBoundaryEnforcementRoute::NitscheCoercivity ||
           route == WeakBoundaryEnforcementRoute::StabilizedTraceCoupling;
}

bool boundaryDescriptorUsesWeakCoercivityRoute(
    const BoundaryConditionDescriptor& bc) noexcept
{
    if (isWeakPenaltyLike(bc.enforcement_kind)) {
        return bc.weak_boundary_route ==
                   WeakBoundaryEnforcementRoute::Unknown ||
               isWeakBoundaryCoercivityRoute(bc.weak_boundary_route);
    }
    if (bc.enforcement_kind != EnforcementKind::WeakInequality) {
        return false;
    }
    return isWeakBoundaryCoercivityRoute(bc.weak_boundary_route);
}

bool parameterScaleUsesWeakCoercivityRoute(
    const ParameterScaleSummary& scale) noexcept
{
    return scale.weak_boundary_route ==
               WeakBoundaryEnforcementRoute::Unknown ||
           isWeakBoundaryCoercivityRoute(scale.weak_boundary_route);
}

enum class MarkerMatchMode : std::uint8_t {
    LoosePlanning,
    StrictCertification,
};

bool markerMatchesBoundary(const BoundaryConditionDescriptor& bc,
                           const OperatorBlockId& block,
                           MarkerMatchMode mode) noexcept
{
    if (block.domain == DomainKind::InterfaceFace) {
        if (mode == MarkerMatchMode::StrictCertification) {
            return block.marker >= 0 &&
                   bc.interface_marker >= 0 &&
                   block.marker == bc.interface_marker;
        }
        return block.marker < 0 || bc.interface_marker < 0 ||
               block.marker == bc.interface_marker;
    }
    if (block.domain == DomainKind::Boundary ||
        block.domain == DomainKind::CoupledBoundary) {
        if (mode == MarkerMatchMode::StrictCertification) {
            return block.marker >= 0 &&
                   bc.boundary_marker >= 0 &&
                   block.marker == bc.boundary_marker;
        }
        return block.marker < 0 || bc.boundary_marker < 0 ||
               block.marker == bc.boundary_marker;
    }
    return true;
}

bool weakBoundaryCoercivityApplies(const ProblemAnalysisContext& context,
                                   const AnalysisSummarySet& summaries,
                                   const BoundarySymbolSummary& boundary)
{
    if (!isFaceOrBoundary(boundary.block.domain)) {
        return false;
    }

    const auto boundary_variables = variablesForBlock(boundary.block);
    bool explicit_noncoercive_inequality = false;
    for (const auto& bc : context.bcDescriptors()) {
        if (bc.domain != boundary.block.domain ||
            !markerMatchesBoundary(bc,
                                   boundary.block,
                                   MarkerMatchMode::StrictCertification)) {
            continue;
        }
        std::vector<VariableKey> bc_variables{bc.primary_variable};
        for (const auto& variable : bc.related_variables) {
            appendUniqueVariable(bc_variables, variable);
        }
        if (!variableSetsIntersect(bc_variables, boundary_variables)) {
            continue;
        }
        if (boundaryDescriptorUsesWeakCoercivityRoute(bc)) {
            return true;
        }
        if (bc.enforcement_kind == EnforcementKind::WeakInequality) {
            explicit_noncoercive_inequality = true;
        }
    }

    if (explicit_noncoercive_inequality) {
        return false;
    }

    for (const auto& scale : summaries.parameter_scales) {
        if (parameterScaleCovers(scale,
                                 boundary.block,
                                 ParameterScaleRole::WeakBoundaryPenalty) &&
            parameterScaleUsesWeakCoercivityRoute(scale)) {
            return true;
        }
    }
    return false;
}

bool hasScopedSymbolicBalance(const FluxBalanceSummary& summary)
{
    if (!summary.symbolic_balance_evidence_present) {
        return false;
    }
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
        !variablesForBlock(summary.block).empty() &&
        (!summary.block.operator_tag.empty() ||
         !summary.block.contribution_id.empty() ||
         !summary.balance_group.empty());
    return group_scoped || contribution_scoped || block_scoped;
}

void addInterfaceTopologyIssue(ProblemAnalysisReport& report,
                               IssueSeverity severity,
                               int marker,
                               const std::string& message)
{
    AnalysisIssue issue;
    issue.severity = severity;
    issue.message = "Interface marker " + std::to_string(marker) +
                    ": " + message;
    report.issues.push_back(std::move(issue));
}

bool boundaryComplementingCertificateComplete(
    const BoundarySymbolSummary& boundary) noexcept
{
    const bool count_ok =
        (boundary.required_boundary_condition_count_present ||
         boundary.required_boundary_condition_count > 0u) &&
        boundary.boundary_condition_count >=
            boundary.required_boundary_condition_count;
    const bool margin_ok =
        boundary.complementing_margin_present &&
        numeric::finitePositive(boundary.complementing_margin);
    const bool lopatinskii_symbol_evidence =
        boundary.tangential_frequency_coverage_present &&
        boundary.decaying_root_count_evidence_present &&
        boundary.stable_subspace_dimension_evidence_present &&
        boundary.parameter_ellipticity_evidence_present &&
        margin_ok &&
        !boundary.complementing_theorem_id.empty() &&
        boundary.root_subspace_mismatch_count == 0u;
    const bool mixed_corner_scope_ok =
        !boundary.mixed_boundary_or_corner_scope_present ||
        boundary.mixed_corner_edge_coverage_present;
    return boundary.principal_symbol_rank_evidence_present &&
           boundary.boundary_symbol_rank_evidence_present &&
           lopatinskii_symbol_evidence &&
           boundary.component_coverage_complete &&
           boundary.dof_coverage_complete &&
           boundary.missing_symbol_count == 0u &&
           mixed_corner_scope_ok &&
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
            claim.evidence_level = certificate_complete
                ? EvidenceLevel::CertifiedNumericTheorem
                : EvidenceLevel::ScopedNumericSummary;
            claim.variables = variablesForBlock(boundary.block);
            claim.tested_block_id = boundary.block.operator_tag;
            claim.description = !satisfied
                ? "Boundary-symbol summary violates the complementing condition"
                : (symbol_mismatch
                    ? "Boundary-symbol summary reports inconsistent decaying-root and stable-subspace evidence"
                : (certificate_complete
                    ? "Boundary-symbol summary satisfies the complementing condition with rank, count, tangential-frequency, root/subspace, margin, component, DOF, and mixed-corner coverage evidence"
                    : "Boundary-symbol summary reports the complementing condition but lacks complete rank, count, tangential-frequency, root/subspace, margin, component, DOF, or mixed-corner coverage evidence"));
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
                ", mixed_boundary_or_corner_scope=" +
                std::string(boundary.mixed_boundary_or_corner_scope_present ? "true" : "false") +
                ", mixed_corner_edge_coverage=" +
                std::string(boundary.mixed_corner_edge_coverage_present ? "true" : "false") +
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
                ", required_count_present=" +
                std::string(boundary.required_boundary_condition_count_present ? "true" : "false") +
                ", required_count=" +
                std::to_string(boundary.required_boundary_condition_count) +
                ", provided_count=" +
                std::to_string(boundary.boundary_condition_count) +
                ", root_subspace_mismatches=" +
                std::to_string(boundary.root_subspace_mismatch_count) +
                ", missing_symbols=" +
                std::to_string(boundary.missing_symbol_count));
            report.claims.push_back(std::move(claim));
        }

        if (weakBoundaryCoercivityApplies(context, *summaries, boundary)) {
            Real reported_penalty_scale = Real{};
            Real reported_required_lower_bound = Real{};
            bool has_penalty_scale = false;
            bool has_required_lower_bound = false;
            bool has_complete_certificate = false;
            bool has_lower_bound_violation = false;
            bool has_adequate_incomplete_certificate = false;
            bool invalid_penalty_numeric_evidence = false;
            for (const auto& scale : summaries->parameter_scales) {
                if (!parameterScaleCovers(
                        scale, boundary.block,
                        ParameterScaleRole::WeakBoundaryPenalty)) {
                    continue;
                }
                has_penalty_scale = true;
                if (scale.required_lower_bound_present) {
                    if (!parameterScaleUsesWeakCoercivityRoute(scale)) {
                        continue;
                    }
                    has_required_lower_bound = true;
                    const bool scale_values_valid =
                        numeric::finitePositive(scale.min_scale_value) &&
                        numeric::finitePositive(scale.max_scale_value) &&
                        scale.min_scale_value <= scale.max_scale_value;
                    const bool lower_bound_valid =
                        numeric::finitePositive(scale.required_lower_bound);
                    const bool trace_constant_valid =
                        !scale.trace_inverse_metadata_present ||
                        numeric::finitePositive(scale.trace_inverse_constant);
                    if (!scale_values_valid ||
                        !lower_bound_valid ||
                        !trace_constant_valid) {
                        invalid_penalty_numeric_evidence = true;
                        continue;
                    }
                    reported_penalty_scale = scale.min_scale_value;
                    reported_required_lower_bound = scale.required_lower_bound;
                    const bool adequate =
                        scale.min_scale_value + Real{1.0e-14} >=
                        scale.required_lower_bound;
                    if (!adequate) {
                        has_lower_bound_violation = true;
                    } else if (scale.trace_inverse_metadata_present &&
                               !scale.scale_theorem_id.empty()) {
                        has_complete_certificate = true;
                    } else {
                        has_adequate_incomplete_certificate = true;
                    }
                } else if (numeric::finitePositive(scale.max_scale_value)) {
                    reported_penalty_scale = scale.max_scale_value;
                } else {
                    invalid_penalty_numeric_evidence = true;
                }
            }

            PropertyClaim claim;
            claim.kind = PropertyKind::WeakBoundaryCoercivity;
            claim.domain = boundary.block.domain;
            claim.variables = variablesForBlock(boundary.block);
            claim.tested_block_id = boundary.block.operator_tag;
            claim.claim_origin = "InterfaceValidationAnalyzer";
            claim.weak_boundary_route = boundary.weak_boundary_route;
            claim.evidence_level = EvidenceLevel::ScopedNumericSummary;
            if (has_penalty_scale) {
                claim.penalty_scale = reported_penalty_scale;
                if (invalid_penalty_numeric_evidence) {
                    claim.status = PropertyStatus::Violated;
                    claim.confidence = AnalysisConfidence::High;
                    claim.certification_class = CertificationClass::Violated;
                    claim.description =
                        "Weak-boundary penalty scale, trace inverse constant, or required lower bound is non-finite, non-positive, or unordered";
                    claim.addEvidence("InterfaceValidationAnalyzer",
                        "ParameterScaleSummary penalty_scale=" +
                        std::to_string(reported_penalty_scale) +
                        ", required_lower_bound=" +
                        std::to_string(reported_required_lower_bound) +
                        ", invalid_numeric_evidence=true",
                        claim.confidence);
                } else if (has_lower_bound_violation) {
                    claim.status = PropertyStatus::Violated;
                    claim.confidence = AnalysisConfidence::High;
                    claim.certification_class = CertificationClass::Violated;
                    claim.weak_coercivity_lower_bound =
                        reported_penalty_scale - reported_required_lower_bound;
                    claim.description =
                        "Scoped weak-boundary minimum penalty scale is below the reported coercivity lower bound";
                    claim.addEvidence("InterfaceValidationAnalyzer",
                        "ParameterScaleSummary min_scale=" +
                        std::to_string(reported_penalty_scale) +
                        ", required_lower_bound=" +
                        std::to_string(reported_required_lower_bound) +
                        ", trace_inverse_and_theorem_on_same_summary=false",
                        claim.confidence);
                } else if (has_complete_certificate) {
                    claim.status = PropertyStatus::Preserved;
                    claim.confidence = AnalysisConfidence::High;
                    claim.certification_class = CertificationClass::Certified;
                    claim.weak_coercivity_lower_bound =
                        reported_penalty_scale - reported_required_lower_bound;
                    claim.description =
                        "Scoped weak-boundary minimum penalty scale satisfies the trace-backed theorem-specific coercivity lower bound";
                    claim.addEvidence("InterfaceValidationAnalyzer",
                        "ParameterScaleSummary min_scale=" +
                        std::to_string(reported_penalty_scale) +
                        ", required_lower_bound=" +
                        std::to_string(reported_required_lower_bound) +
                        ", trace_inverse_metadata=true, scale_theorem=true",
                        claim.confidence);
                } else if (has_required_lower_bound) {
                    (void)has_adequate_incomplete_certificate;
                    claim.status = PropertyStatus::Unknown;
                    claim.confidence = AnalysisConfidence::Medium;
                    claim.certification_class = CertificationClass::NotCertified;
                    claim.weak_coercivity_lower_bound =
                        reported_penalty_scale - reported_required_lower_bound;
                    claim.description =
                        "Scoped weak-boundary minimum penalty scale satisfies the reported lower bound but lacks trace/inverse or theorem metadata on the same parameter-scale summary";
                    claim.addEvidence("InterfaceValidationAnalyzer",
                        "ParameterScaleSummary min_scale=" +
                        std::to_string(reported_penalty_scale) +
                        ", required_lower_bound=" +
                        std::to_string(reported_required_lower_bound) +
                        ", complete_same_summary=false",
                        claim.confidence);
                } else {
                    claim.status = PropertyStatus::Unknown;
                    claim.confidence = AnalysisConfidence::Medium;
                    claim.certification_class = CertificationClass::NotCertified;
                    claim.description =
                        "Weak-boundary penalty scale is present but lacks a theorem-specific coercivity lower bound";
                    claim.addEvidence("InterfaceValidationAnalyzer",
                        "ParameterScaleSummary max_scale=" +
                        std::to_string(reported_penalty_scale) +
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
        const bool tolerance_declared =
            numeric::finiteDeclaredTolerance(flux.balance_tolerance);
        const Real tol = tolerance_declared
            ? flux.balance_tolerance
            : Real{1.0e-10};
        const bool residuals_finite =
            numeric::finiteAbsResidualTriple(
                flux.local_residual_norm,
                flux.global_residual_norm,
                flux.interface_pair_residual_norm);
        const bool numeric_evidence_valid =
            tolerance_declared && residuals_finite;
        const Real residual = residuals_finite
            ? numeric::maxAbsTriple(flux.local_residual_norm,
                                    flux.global_residual_norm,
                                    flux.interface_pair_residual_norm)
            : Real{};
        const bool balanced =
            numeric_evidence_valid &&
            residual <= tol &&
            flux.local_violation_count == 0u;
        const bool violated =
            flux.local_violation_count > 0u ||
            (numeric_evidence_valid && residual > tol);
        const bool closure_metadata_complete =
            fluxClosureCertificationMetadataComplete(flux);
        const bool symbolic_balance_present =
            hasScopedSymbolicBalance(flux);
        const bool certified =
            balanced &&
            tolerance_declared &&
            closure_metadata_complete &&
            symbolic_balance_present;

        PropertyClaim claim;
        claim.kind = PropertyKind::InterfaceCondition;
        claim.status = violated
            ? PropertyStatus::Violated
            : (certified ? PropertyStatus::Preserved
                         : (numeric_evidence_valid ? PropertyStatus::Likely
                                                   : PropertyStatus::Unknown));
        claim.confidence = violated || certified
            ? AnalysisConfidence::High
            : AnalysisConfidence::Medium;
        claim.certification_class = violated
            ? CertificationClass::Violated
            : (certified ? CertificationClass::Certified
                         : CertificationClass::NotCertified);
        claim.domain = flux.block.domain;
        claim.variables = variablesForBlock(flux.block);
        claim.interface_balance_residual = flux.interface_pair_residual_norm;
        claim.flux_balance_residual = residual;
        claim.tested_block_id = flux.block.operator_tag;
        claim.description = violated
            ? "Boundary/interface flux summary violates the declared tolerance"
            : (certified
                ? "Boundary/interface flux summary is balanced with matching symbolic balance and complete flux closure metadata"
                : (numeric_evidence_valid
                    ? "Boundary/interface flux summary is residual-balanced but lacks symbolic balance or complete flux closure metadata"
                    : "Boundary/interface flux summary lacks finite declared tolerance or finite residual evidence"));
        claim.claim_origin = "InterfaceValidationAnalyzer";
        claim.addEvidence("InterfaceValidationAnalyzer",
            "FluxBalanceSummary residual=" +
            std::to_string(residual) +
            ", tolerance=" + std::to_string(tol) +
            ", tolerance_declared=" +
            std::string(tolerance_declared ? "true" : "false") +
            ", finite_residuals=" +
            std::string(residuals_finite ? "true" : "false") +
            ", symbolic_balance=" +
            std::string(symbolic_balance_present ? "true" : "false") +
            ", closure_metadata=" +
            std::string(closure_metadata_complete ? "true" : "false"),
            claim.confidence);
        if (!violated && !numeric_evidence_valid) {
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Warning;
            issue.message =
                "Boundary/interface flux summary for block '" +
                flux.block.operator_tag +
                "' has non-finite residual evidence or invalid tolerance";
            report.issues.push_back(std::move(issue));
        }
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
        for (int marker : itopo->interfaceCoverageForReferencedMarkers(
                 specific_markers_referenced)) {
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Error;
            issue.message =
                "Interface contribution references marker " +
                std::to_string(marker) +
                " but no InterfaceMesh faces are registered for that marker";
            report.issues.push_back(std::move(issue));
        }

        auto validateMarkerTopology = [&](int marker, bool targeted) {
            if (!targeted) {
                return;
            }
            if (!itopo->markerRecordsMatchKey(marker)) {
                addInterfaceTopologyIssue(
                    report, IssueSeverity::Error, marker,
                    "marker_to_faces contains face records with mismatched marker ids");
            }
            if (itopo->numFacesForMarker(marker) == 0u) {
                addInterfaceTopologyIssue(
                    report, IssueSeverity::Error, marker,
                    "referenced interface marker has no face records");
            }
            if (!itopo->markerHasTwoSidedFaces(marker)) {
                addInterfaceTopologyIssue(
                    report, IssueSeverity::Error, marker,
                    "one or more interface faces are missing minus/plus cell incidence for two-sided coupling");
            }
            if (!itopo->markerHasOrientation(marker)) {
                addInterfaceTopologyIssue(
                    report, IssueSeverity::Warning, marker,
                    "orientation metadata is missing for jump/flux/transfer validation");
            }
            if (!itopo->markerHasValidLocalFaceIndices(marker)) {
                addInterfaceTopologyIssue(
                    report, IssueSeverity::Error, marker,
                    "minus/plus local face indices are missing or invalid");
            }
            if (!itopo->markerHasConsistentRegions(marker)) {
                addInterfaceTopologyIssue(
                    report, IssueSeverity::Warning, marker,
                    "minus/plus region metadata is incomplete for region-scoped conservation or nullspace analysis");
            }
            if (itopo->markerHasNonmanifoldIncidence(marker)) {
                addInterfaceTopologyIssue(
                    report, IssueSeverity::Warning, marker,
                    "duplicate face incidence suggests duplicate or nonmanifold interface topology; explicit nonmanifold support metadata is not available");
            }
        };

        std::set<int> validated_markers;
        for (int marker : specific_markers_referenced) {
            if (itopo->markerExists(marker) &&
                validated_markers.insert(marker).second) {
                validateMarkerTopology(marker, true);
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
            } else if (validated_markers.insert(reg_marker).second) {
                validateMarkerTopology(reg_marker, true);
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

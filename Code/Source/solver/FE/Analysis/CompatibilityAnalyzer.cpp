/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/CompatibilityAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

[[nodiscard]] bool isPrimalTraceAnchoringBC(const BoundaryConditionDescriptor& bc) noexcept
{
    const bool value_like_trace =
        bc.trace_kind == TraceKind::Value ||
        bc.trace_kind == TraceKind::NormalComponent ||
        bc.trace_kind == TraceKind::TangentialComponent;

    if (!value_like_trace) {
        return false;
    }

    return bc.enforcement_kind == EnforcementKind::Strong ||
           bc.enforcement_kind == EnforcementKind::WeakPenalty ||
           bc.enforcement_kind == EnforcementKind::WeakNitsche ||
           bc.enforcement_kind == EnforcementKind::WeakInequality;
}

void emitInitialCompatibilityClaims(const ProblemAnalysisContext& context,
                                    ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) {
        return;
    }

    for (const auto& summary : summaries->initial_compatibility) {
        const bool constraint_residual_finite =
            numeric::finite(summary.initial_constraint_residual);
        const bool boundary_residual_finite =
            numeric::finite(summary.initial_boundary_residual);
        const auto constraint_residual = constraint_residual_finite
            ? std::abs(summary.initial_constraint_residual)
            : summary.initial_constraint_residual;
        const auto boundary_residual = boundary_residual_finite
            ? std::abs(summary.initial_boundary_residual)
            : summary.initial_boundary_residual;
        const bool invariant_residual_finite =
            numeric::finite(summary.invariant_domain_admissibility_residual);
        const auto invariant_residual = invariant_residual_finite
            ? std::abs(summary.invariant_domain_admissibility_residual)
            : summary.invariant_domain_admissibility_residual;
        const bool tolerance_declared =
            summary.residual_tolerance_declared &&
            numeric::finiteDeclaredTolerance(summary.residual_tolerance);
        const auto tolerance = tolerance_declared
            ? summary.residual_tolerance
            : 0.0;
        const bool algebraic_scope_checked =
            summary.algebraic_constraint_metadata_present &&
            summary.checked_constraint_family_count > 0u;
        const bool boundary_scope_checked =
            summary.boundary_constraint_metadata_present &&
            summary.checked_boundary_condition_count > 0u;
        const bool invariant_scope_checked =
            summary.invariant_domain_metadata_present &&
            summary.checked_invariant_state_count > 0u &&
            !summary.invariant_set_id.empty() &&
            !summary.invariant_domain_variables.empty() &&
            summary.invariant_domain_admissibility_residual_present;
        const bool compatibility_scope_declared =
            !summary.compatibility_scope.empty() &&
            (algebraic_scope_checked ||
             boundary_scope_checked ||
             invariant_scope_checked);
        const bool residuals_finite_for_checked_scopes =
            (!algebraic_scope_checked || constraint_residual_finite) &&
            (!boundary_scope_checked || boundary_residual_finite) &&
            (!invariant_scope_checked || invariant_residual_finite);
        const bool violated =
            tolerance_declared &&
            residuals_finite_for_checked_scopes &&
            ((algebraic_scope_checked && constraint_residual > tolerance) ||
             (boundary_scope_checked && boundary_residual > tolerance) ||
             (invariant_scope_checked && invariant_residual > tolerance) ||
             (invariant_scope_checked &&
              summary.invariant_domain_initial_violation_count > 0u));

        PropertyClaim claim;
        claim.kind = PropertyKind::InitialDataCompatibility;
        claim.domain = DomainKind::Global;
        claim.constraint_drift_norm = constraint_residual;
        if (!tolerance_declared || !compatibility_scope_declared ||
            !residuals_finite_for_checked_scopes) {
            claim.status = PropertyStatus::Unknown;
            claim.confidence = AnalysisConfidence::Medium;
            claim.certification_class = CertificationClass::NotCertified;
            claim.description =
                "Initial-data compatibility summary lacks declared tolerance, finite checked residuals, or checked constraint-family scope";
        } else if (violated) {
            claim.status = PropertyStatus::Violated;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Violated;
            claim.initial_data_compatible = false;
            claim.description =
                "Initial data violates algebraic, boundary, or invariant-domain compatibility";
        } else {
            claim.status = PropertyStatus::Preserved;
            claim.confidence = AnalysisConfidence::High;
            claim.certification_class = CertificationClass::Certified;
            claim.initial_data_compatible = true;
            claim.description =
                "Initial data satisfies declared algebraic, boundary, or invariant-domain compatibility checks";
        }
        claim.claim_origin = "CompatibilityAnalyzer";
        claim.addEvidence("CompatibilityAnalyzer",
            "InitialCompatibilitySummary constraint_residual=" +
            std::to_string(summary.initial_constraint_residual) +
            ", boundary_residual=" +
            std::to_string(summary.initial_boundary_residual) +
            ", tolerance=" + std::to_string(summary.residual_tolerance) +
            ", tolerance_declared=" +
            std::string(tolerance_declared ? "true" : "false") +
            ", finite_constraint_residual=" +
            std::string(constraint_residual_finite ? "true" : "false") +
            ", finite_boundary_residual=" +
            std::string(boundary_residual_finite ? "true" : "false") +
            ", scope='" + summary.compatibility_scope + "'" +
            ", algebraic_scope_checked=" +
            std::string(algebraic_scope_checked ? "true" : "false") +
            ", boundary_scope_checked=" +
            std::string(boundary_scope_checked ? "true" : "false") +
            ", invariant_scope_checked=" +
            std::string(invariant_scope_checked ? "true" : "false") +
            ", invariant_violations=" +
            std::to_string(summary.invariant_domain_initial_violation_count) +
            ", invariant_set='" + summary.invariant_set_id + "'" +
            ", checked_invariant_states=" +
            std::to_string(summary.checked_invariant_state_count) +
            ", invariant_residual_present=" +
            std::string(summary.invariant_domain_admissibility_residual_present ? "true" : "false") +
            ", invariant_residual=" +
            std::to_string(summary.invariant_domain_admissibility_residual) +
            ", invariant_residual_finite=" +
            std::string(invariant_residual_finite ? "true" : "false"));
        report.claims.push_back(std::move(claim));
    }
}

} // namespace

std::string CompatibilityAnalyzer::name() const {
    return "CompatibilityAnalyzer";
}

void CompatibilityAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
    emitInitialCompatibilityClaims(context, report);

    auto nullspace_claims = report.claimsOfKind(PropertyKind::Nullspace);
    if (nullspace_claims.empty()) return;

    const auto& bcs = context.bcDescriptors();
    const auto& contributions = context.contributions();

    // Build a per-field map of NullspaceLifting and NullspacePreserving
    // contributions for enhanced compatibility analysis.
    // NullspacePreserving (e.g., periodic BC) does NOT suppress compatibility warnings.
    // NullspaceLifting (e.g., Dirichlet BC) DOES suppress compatibility warnings.
    std::unordered_map<FieldId, bool> fields_with_lifting;
    for (const auto& contrib : contributions) {
        if (hasFlag(contrib.traits, OperatorTraitFlags::NullspaceLifting)) {
            for (const auto& tv : contrib.test_variables) {
                if (tv.kind == VariableKind::FieldComponent &&
                    tv.field_id != INVALID_FIELD_ID) {
                    fields_with_lifting[tv.field_id] = true;
                }
            }
        }
        // NullspacePreserving contributions are intentionally NOT added to
        // the lifting map — they preserve the nullspace and thus do not
        // remove the compatibility condition.
    }

    for (const auto* ns_claim : nullspace_claims) {
        FieldId fid = ns_claim->field;
        if (fid == INVALID_FIELD_ID) continue;

        if (ns_claim->status != PropertyStatus::Exact &&
            ns_claim->status != PropertyStatus::Likely) {
            continue;
        }

        // Check if NullspaceLifting contributions anchor this field
        if (fields_with_lifting.count(fid)) {
            continue;  // Lifting suppresses the compatibility warning
        }

        // Collect BCs targeting this field.
        // Key distinction: algebraic relations (periodic, MPC) PRESERVE the
        // nullspace and do NOT count as anchoring/value BCs. A periodic-only
        // Poisson still has a compatibility condition.
        bool has_anchoring_bc = false;
        bool has_any_non_algebraic_bc = false;
        bool all_non_algebraic_are_flux = true;

        for (const auto& bc : bcs) {
            if (bc.primary_variable.kind != VariableKind::FieldComponent) continue;
            if (bc.primary_variable.field_id != fid) continue;

            // Skip algebraic relations — they preserve nullspace
            if (bc.enforcement_kind == EnforcementKind::AffineRelation) continue;

            has_any_non_algebraic_bc = true;

            // Check if this BC anchors the constant mode
            if (bc.anchors_constant_mode || bc.anchors_rigid_body_translation) {
                has_anchoring_bc = true;
                all_non_algebraic_are_flux = false;
            }

            if (isPrimalTraceAnchoringBC(bc)) {
                has_anchoring_bc = true;
                all_non_algebraic_are_flux = false;
            } else if (bc.enforcement_kind != EnforcementKind::WeakConsistent) {
                all_non_algebraic_are_flux = false;
            }
        }

        // Compatibility condition arises when:
        // 1. Field has a nullspace mode
        // 2. No non-algebraic BCs anchor the mode (all are Neumann/flux, or
        //    only algebraic relations like periodicity exist)
        if (!has_anchoring_bc && (all_non_algebraic_are_flux || !has_any_non_algebraic_bc)) {
            PropertyClaim claim;
            claim.kind = PropertyKind::CompatibilityCondition;
            claim.status = ns_claim->status;
            claim.confidence = ns_claim->confidence;
            claim.field = fid;
            claim.component = ns_claim->component;
            claim.region = ns_claim->region;
            claim.domain = DomainKind::Global;
            claim.variables = ns_claim->variables;

            if (!has_any_non_algebraic_bc) {
                claim.description =
                    "No anchoring boundary conditions found for field " +
                    std::to_string(fid) +
                    " which has a nullspace mode — solvability condition "
                    "requires the right-hand side to be orthogonal to "
                    "the nullspace";
            } else {
                claim.description =
                    "All non-algebraic boundary conditions for field " +
                    std::to_string(fid) +
                    " are flux-type (Neumann) — solvability condition "
                    "requires the right-hand side integral to satisfy "
                    "compatibility (e.g., integral of f = 0 for "
                    "pure Neumann Poisson)";
            }

            claim.addEvidence("CompatibilityAnalyzer",
                "Nullspace: " + ns_claim->description,
                ns_claim->confidence);

            report.claims.push_back(std::move(claim));
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/CompatibilityAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

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
           bc.enforcement_kind == EnforcementKind::WeakNitsche;
}

} // namespace

std::string CompatibilityAnalyzer::name() const {
    return "CompatibilityAnalyzer";
}

void CompatibilityAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
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

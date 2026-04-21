/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ConstraintRankAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

namespace svmp {
namespace FE {
namespace analysis {

std::string ConstraintRankAnalyzer::name() const {
    return "ConstraintRankAnalyzer";
}

void ConstraintRankAnalyzer::run(const ProblemAnalysisContext& context,
                                 ProblemAnalysisReport& report) const
{
    const auto* cs = context.constraintSummary();
    const auto& bcs = context.bcDescriptors();
    const auto& contributions = context.contributions();

    // --- Check for constraint conflicts → OverConstraint ---
    // This runs unconditionally — over-constraint can exist without nullspace.
    if (cs && cs->hasConflicts()) {
        for (const auto& conflict : cs->conflicts) {
            PropertyClaim claim;
            claim.kind = PropertyKind::OverConstraint;
            // Structural anomaly, not a definitive error — masters + nonzero
            // inhomogeneity can be legitimate (periodic with offset).
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
            claim.domain = DomainKind::Global;
            claim.description = "Possible constraint conflict: " +
                                conflict.description;
            for (const auto& src : conflict.conflicting_sources) {
                claim.addEvidence("ConstraintRankAnalyzer",
                    "Source: " + src, AnalysisConfidence::Medium);
            }
            report.claims.push_back(std::move(claim));
        }
    }

    // --- Build per-field maps for NullspaceLifting contributions ---
    // Distinguish ExactlyRemoves (strong Dirichlet) from WeaklyLifts (Robin, Nitsche).
    // WeaklyLifts counts as anchoring but may warrant a lower-confidence claim.
    std::unordered_map<FieldId, bool> fields_with_exact_lifting;
    std::unordered_map<FieldId, bool> fields_with_weak_lifting;
    for (const auto& contrib : contributions) {
        if (hasFlag(contrib.traits, OperatorTraitFlags::NullspaceLifting)) {
            for (const auto& tv : contrib.test_variables) {
                if (tv.kind == VariableKind::FieldComponent &&
                    tv.field_id != INVALID_FIELD_ID) {
                    if (contrib.nullspace_effect.has_value() &&
                        *contrib.nullspace_effect == NullspaceEffect::ExactlyRemoves) {
                        fields_with_exact_lifting[tv.field_id] = true;
                    } else {
                        fields_with_weak_lifting[tv.field_id] = true;
                    }
                }
            }
        }
    }

    // --- Build a set of fields with NullspacePreserving contributions ---
    // These preserve nullspace (e.g., periodic BCs) — do NOT count as anchoring.
    std::unordered_map<FieldId, bool> fields_with_preserving;
    for (const auto& contrib : contributions) {
        if (hasFlag(contrib.traits, OperatorTraitFlags::NullspacePreserving)) {
            for (const auto& tv : contrib.test_variables) {
                if (tv.kind == VariableKind::FieldComponent &&
                    tv.field_id != INVALID_FIELD_ID) {
                    fields_with_preserving[tv.field_id] = true;
                }
            }
        }
    }

    // --- For each Nullspace claim, check if BCs anchor the mode ---
    auto nullspace_claims = report.claimsOfKind(PropertyKind::Nullspace);
    for (const auto* ns_claim : nullspace_claims) {
        FieldId fid = ns_claim->field;
        if (fid == INVALID_FIELD_ID) continue;

        bool is_rigid_body = ns_claim->description.find("rigid-body") !=
                             std::string::npos ||
                             ns_claim->description.find("rigid") !=
                             std::string::npos;

        // Check if any BC truly anchors this field's nullspace mode.
        // Algebraic relations (periodic, MPC) PRESERVE the nullspace — skip them.
        // For per-component claims, only BCs targeting that specific component
        // (or all components, bc.component == -1) count as anchoring.
        int claim_comp = ns_claim->component;
        bool anchors_constant = false;
        bool anchors_translation = false;
        bool anchors_rotation = false;

        // First check contributions with structured NullspaceEffect to distinguish
        // exact anchoring from weak regularization. This takes priority over BC
        // boolean flags which conflate the two.
        bool weak_only = false;
        if (fields_with_exact_lifting.count(fid)) {
            anchors_constant = true;
        } else if (fields_with_weak_lifting.count(fid)) {
            anchors_constant = true;
            weak_only = true;
        }

        // Then check BC descriptor flags as fallback for BCs that haven't been
        // lowered into contributions (or whose contributions lack NullspaceEffect).
        // Only strong enforcement (Strong, not WeakPenalty/WeakNitsche) sets
        // anchoring from boolean flags — weak enforcement is already handled above
        // via the NullspaceEffect contribution path.
        for (const auto& bc : bcs) {
            if (bc.primary_variable.kind != VariableKind::FieldComponent) continue;
            if (bc.primary_variable.field_id != fid) continue;
            if (bc.enforcement_kind == EnforcementKind::AffineRelation) continue;
            if (claim_comp >= 0 && bc.component >= 0 && bc.component != claim_comp) continue;

            if (bc.enforcement_kind == EnforcementKind::Strong) {
                if (bc.anchors_constant_mode) anchors_constant = true;
                if (bc.anchors_rigid_body_translation) anchors_translation = true;
                if (bc.anchors_rigid_body_rotation) anchors_rotation = true;
            }
            // WeakPenalty/WeakNitsche anchoring is handled via NullspaceEffect above.
            // If no contributions were lowered, fall back to BC flags for these too.
            if (!anchors_constant &&
                (bc.enforcement_kind == EnforcementKind::WeakPenalty ||
                 bc.enforcement_kind == EnforcementKind::WeakNitsche ||
                 bc.enforcement_kind == EnforcementKind::WeakInequality)) {
                if (bc.anchors_constant_mode) {
                    anchors_constant = true;
                    weak_only = true;
                }
            }
        }

        // Check constraint summary for non-algebraic Dirichlet DOFs.
        // For per-component claims (component >= 0), check the specific component.
        bool has_dirichlet_dofs = false;
        if (cs) {
            for (const auto& cset : cs->constrained_sets) {
                if (cset.field != fid) continue;
                if (cset.region != -1) continue;
                // Match component: -1 (aggregate) for field-wide claims,
                // specific component for per-component claims
                if (claim_comp >= 0 && cset.component != claim_comp) continue;
                if (claim_comp < 0 && cset.component != -1) continue;
                if (cset.num_constrained_dofs > 0 &&
                    cset.constraint_source != "AffineRelation") {
                    has_dirichlet_dofs = true;
                }
                break;
            }
        }

        bool anchored = false;
        if (is_rigid_body) {
            // Rigid-body modes need BOTH translation AND rotation anchoring.
            anchored = anchors_translation && anchors_rotation;
        } else {
            // Scalar/componentwise constant: anchoring flags or Dirichlet DOFs.
            // For per-component claims, only that component's DOFs count.
            anchored = anchors_constant || has_dirichlet_dofs;
        }

        if (!anchored) {
            PropertyClaim claim;
            claim.kind = PropertyKind::UnderConstraint;
            claim.status = PropertyStatus::Violated;
            claim.confidence = ns_claim->confidence;
            claim.field = fid;
            claim.component = ns_claim->component;
            claim.region = ns_claim->region;
            claim.domain = DomainKind::Global;
            claim.variables = ns_claim->variables;
            claim.description =
                "Nullspace mode is not anchored by boundary conditions: " +
                ns_claim->description;
            claim.addEvidence("ConstraintRankAnalyzer",
                "No anchoring BC found for field " + std::to_string(fid),
                ns_claim->confidence);
            report.claims.push_back(std::move(claim));
        } else if (weak_only) {
            // Anchored, but only through weak enforcement (Robin/Nitsche penalty).
            // Emit an informational issue (not an UnderConstraint claim) so the
            // user knows the anchoring is regularization-based rather than exact
            // Dirichlet elimination. The mode IS anchored, just weakly.
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Info;
            issue.message =
                "Nullspace mode for field " + std::to_string(fid) +
                " is weakly anchored (penalty/Nitsche only, no exact Dirichlet"
                " elimination) — anchoring depends on penalty parameter"
                " magnitude: " + ns_claim->description;
            report.issues.push_back(std::move(issue));
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp

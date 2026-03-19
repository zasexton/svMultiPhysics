/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "NullspaceAnalyzer.h"

namespace svmp {
namespace FE {
namespace forms {

// ============================================================================
// analyze — top-level entry point
// ============================================================================

std::vector<gauge::GaugeCandidate>
NullspaceAnalyzer::analyze(const FormExpr& residual, std::span<const FieldId> fields) const
{
    std::vector<gauge::GaugeCandidate> candidates;

    if (!residual.isValid() || !residual.node()) {
        return candidates;
    }

    const auto& root = *residual.node();

    for (FieldId fid : fields) {
        auto classification = classifyField(root, fid);

        if (classification.occurrence_count == 0) {
            // Field not referenced in the residual — skip
            continue;
        }

        // Determine confidence reduction factors
        const bool has_stab = classification.has_stabilization;
        const bool has_dt = classification.has_time_derivative;

        auto make_confidence = [&]() -> gauge::Confidence {
            if (has_stab) return gauge::Confidence::Medium;
            return gauge::Confidence::High;
        };

        auto make_reason = [&](const char* mode_desc) -> std::string {
            std::string reason = mode_desc;
            if (has_stab) {
                reason += " (stabilization terms weakly break the nullspace)";
            }
            return reason;
        };

        // Skip if the field has absolute terms — mode is fully anchored
        if (classification.has_absolute_terms) {
            continue;
        }

        if (!classification.only_through_annihilating_ops) {
            continue;
        }

        // When a field has a time derivative AND appears only through
        // annihilating ops, the mass matrix from the time discretization
        // (M * (u^{n+1} - u^n) / dt) anchors the constant mode exactly.
        // Do NOT produce a gauge candidate — the system is not singular.
        if (has_dt) {
            continue;
        }

        // --- Rigid-body modes: sym(grad(u)) only, vector field ---
        // Must check BEFORE scalar/componentwise constant since sym_grad is
        // more specific.
        if (classification.only_through_sym_grad &&
            !classification.has_plain_grad &&
            classification.field_value_dimension > 1) {
            gauge::GaugeCandidate c;
            c.field = fid;
            c.component = -1;
            c.family = gauge::NullspaceModeFamily::KernelOfSymGrad;
            c.source = gauge::CandidateSource::FormsInference;
            c.confidence = make_confidence();
            c.reason = make_reason("Vector field appears only through sym(grad(u)) "
                                   "— rigid-body modes (translations + rotations) "
                                   "are in the operator nullspace");
            candidates.push_back(std::move(c));
            continue;
        }

        // --- Componentwise vector constant or scalar constant ---
        if (classification.field_value_dimension > 1) {
            // Vector field with all paths through annihilating ops →
            // componentwise constant modes
            gauge::GaugeCandidate c;
            c.field = fid;
            c.component = -1;
            c.family = gauge::NullspaceModeFamily::ComponentwiseConstant;
            c.source = gauge::CandidateSource::FormsInference;
            c.confidence = make_confidence();
            c.reason = make_reason("Vector field appears only through gradient-like "
                                   "operators — per-component constant shifts are "
                                   "in the operator nullspace");
            candidates.push_back(std::move(c));
        } else {
            // Scalar field → scalar constant mode
            gauge::GaugeCandidate c;
            c.field = fid;
            c.component = -1;
            c.family = gauge::NullspaceModeFamily::ScalarConstant;
            c.source = gauge::CandidateSource::FormsInference;
            c.confidence = make_confidence();
            c.reason = make_reason("Field appears only through gradient-like operators "
                                   "— constant shift is in the operator nullspace");
            candidates.push_back(std::move(c));
        }
    }

    return candidates;
}

// ============================================================================
// classifyField — classify how a single field appears in the DAG
// ============================================================================

FieldAppearanceClassification
NullspaceAnalyzer::classifyField(const FormExprNode& root, FieldId target_field) const
{
    FieldAppearanceClassification result;
    result.field = target_field;

    WalkState initial_state{};
    walkNode(root, target_field, initial_state, result);

    // If no occurrences were found, reset the "only_through" flags to false
    // since they are vacuously true.
    if (result.occurrence_count == 0) {
        result.only_through_annihilating_ops = false;
        result.only_through_sym_grad = false;
    }

    return result;
}

// ============================================================================
// walkNode — recursive DAG walk
// ============================================================================

void NullspaceAnalyzer::walkNode(const FormExprNode& node,
                                  FieldId target_field,
                                  const WalkState& state,
                                  FieldAppearanceClassification& result) const
{
    const auto ty = node.type();

    // ---- Check if this node is a leaf referencing our target field ----
    switch (ty) {
        case FormExprType::DiscreteField:
        case FormExprType::StateField:
        case FormExprType::TrialFunction: {
            auto fid = node.fieldId();
            if (fid.has_value() && *fid == target_field) {
                result.occurrence_count++;

                // Capture field dimension from space signature if available
                const auto* sig = node.spaceSignature();
                if (sig && sig->value_dimension > result.field_value_dimension) {
                    result.field_value_dimension = sig->value_dimension;
                }

                // Check whether any annihilating operator was applied on this path
                if (!state.hasAnnihilatingOp() && !state.under_time_derivative) {
                    // Field appears without differential operators → absolute term
                    result.only_through_annihilating_ops = false;
                    result.has_absolute_terms = true;
                }

                if (state.under_time_derivative) {
                    result.has_time_derivative = true;
                }

                if (state.near_cell_diameter && state.hasAnnihilatingOp()) {
                    result.has_stabilization = true;
                }

                // Track sym(grad) vs plain grad for rigid-body analysis
                if (state.under_gradient) {
                    if (state.under_sym_part) {
                        // This path goes through sym(grad(field))
                        // only_through_sym_grad remains true
                    } else {
                        // Plain grad without sym — breaks rigid-body invariance
                        result.has_plain_grad = true;
                        result.only_through_sym_grad = false;
                    }
                } else if (state.hasAnnihilatingOp()) {
                    // Other annihilating ops (div, curl, hessian) — not sym(grad)
                    result.only_through_sym_grad = false;
                } else if (!state.under_time_derivative) {
                    // Not under any annihilating op and not time derivative
                    result.only_through_sym_grad = false;
                }
            }
            return;  // leaf — no children
        }

        // TestFunction is not a trial field, skip
        case FormExprType::TestFunction:
            return;

        // PreviousSolutionRef references the same field as the trial (for time
        // integration).  In the DAG it has no fieldId(); the mapping is implicit.
        // For nullspace analysis we treat it as the field appearing with a time
        // derivative operator (it comes from dt(u)).
        case FormExprType::PreviousSolutionRef:
            return;

        default:
            break;
    }

    // ---- Propagate differential-operator context to children ----
    auto children = node.children();

    switch (ty) {
        case FormExprType::Gradient: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState child_state = state;
                child_state.under_gradient = true;
                walkNode(*child, target_field, child_state, result);
            }
            return;
        }

        case FormExprType::Divergence: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState child_state = state;
                child_state.under_divergence = true;
                walkNode(*child, target_field, child_state, result);
            }
            return;
        }

        case FormExprType::Curl: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState child_state = state;
                child_state.under_curl = true;
                walkNode(*child, target_field, child_state, result);
            }
            return;
        }

        case FormExprType::Hessian: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState child_state = state;
                child_state.under_hessian = true;
                walkNode(*child, target_field, child_state, result);
            }
            return;
        }

        case FormExprType::TimeDerivative: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState child_state = state;
                child_state.under_time_derivative = true;
                walkNode(*child, target_field, child_state, result);
            }
            return;
        }

        case FormExprType::SymmetricPart: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState child_state = state;
                child_state.under_sym_part = true;
                walkNode(*child, target_field, child_state, result);
            }
            return;
        }

        // Stabilization heuristic: CellDiameter in the same subtree as a field
        // suggests h-scaled penalty terms (PSPG, LSIC, GLS).
        case FormExprType::CellDiameter: {
            // CellDiameter is a leaf (no children) but we set the flag for
            // siblings in the parent multiplication.  The flag is set in the
            // parent's state propagation below.
            return;
        }

        default:
            break;
    }

    // ---- For non-differential operators, check for stabilization context ----
    // If this subtree contains CellDiameter among its children, flag the
    // stabilization context for all field occurrences in sibling subtrees.
    bool subtree_has_cell_diameter = false;
    if (ty == FormExprType::Multiply || ty == FormExprType::InnerProduct ||
        ty == FormExprType::DoubleContraction) {
        for (const auto* child : children) {
            if (child && child->type() == FormExprType::CellDiameter) {
                subtree_has_cell_diameter = true;
                break;
            }
        }
    }

    // ---- Default: recurse into children with inherited state ----
    for (const auto* child : children) {
        if (!child) continue;
        WalkState child_state = state;
        if (subtree_has_cell_diameter) {
            child_state.near_cell_diameter = true;
        }
        walkNode(*child, target_field, child_state, result);
    }
}

} // namespace forms
} // namespace FE
} // namespace svmp

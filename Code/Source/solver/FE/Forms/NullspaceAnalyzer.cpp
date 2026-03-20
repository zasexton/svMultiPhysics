/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "NullspaceAnalyzer.h"
#include "Analysis/FormStructureAnalyzer.h"

namespace svmp {
namespace FE {
namespace forms {

// ============================================================================
// analyze — delegates to FormStructureAnalyzer, then maps via analyzeFromSummary
// ============================================================================

std::vector<gauge::GaugeCandidate>
NullspaceAnalyzer::analyze(const FormExpr& residual, std::span<const FieldId> fields) const
{
    if (!residual.isValid() || !residual.node()) {
        return {};
    }

    analysis::FormStructureAnalyzer fsa;
    auto summary = fsa.analyze(residual, fields);
    return analyzeFromSummary(summary);
}

// ============================================================================
// analyzeFromSummary — maps FieldOperatorSummary → GaugeCandidate
// ============================================================================

std::vector<gauge::GaugeCandidate>
NullspaceAnalyzer::analyzeFromSummary(const analysis::FormStructureSummary& summary) const
{
    std::vector<gauge::GaugeCandidate> candidates;

    for (const auto& fs : summary.per_field) {
        if (fs.occurrence_count == 0) continue;

        const bool has_stab = fs.has_stabilization;
        const bool has_dt = fs.has_time_derivative;

        auto make_confidence = [&]() -> gauge::Confidence {
            return has_stab ? gauge::Confidence::Medium : gauge::Confidence::High;
        };

        auto make_reason = [&](const char* mode_desc) -> std::string {
            std::string reason = mode_desc;
            if (has_stab) {
                reason += " (stabilization terms weakly break the nullspace)";
            }
            return reason;
        };

        // Skip if field has absolute terms — mode is fully anchored
        if (fs.has_absolute_value) continue;

        if (!fs.only_through_annihilating_ops) continue;

        // Time derivative mass matrix anchors the constant mode
        if (has_dt) continue;

        // Rigid-body modes: sym(grad(u)) only, vector field
        if (fs.only_through_sym_grad && !fs.has_plain_grad &&
            fs.value_dimension > 1) {
            gauge::GaugeCandidate c;
            c.field = fs.field;
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

        // Componentwise vector constant or scalar constant
        if (fs.value_dimension > 1) {
            gauge::GaugeCandidate c;
            c.field = fs.field;
            c.component = -1;
            c.family = gauge::NullspaceModeFamily::ComponentwiseConstant;
            c.source = gauge::CandidateSource::FormsInference;
            c.confidence = make_confidence();
            c.reason = make_reason("Vector field appears only through gradient-like "
                                   "operators — per-component constant shifts are "
                                   "in the operator nullspace");
            candidates.push_back(std::move(c));
        } else {
            gauge::GaugeCandidate c;
            c.field = fs.field;
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
// classifyField — backward-compatible wrapper
// ============================================================================

FieldAppearanceClassification
NullspaceAnalyzer::classifyField(const FormExprNode& root, FieldId target_field) const
{
    analysis::FormStructureAnalyzer fsa;
    auto fs = fsa.analyzeField(root, target_field);

    FieldAppearanceClassification result;
    result.field = fs.field;
    result.only_through_annihilating_ops = fs.only_through_annihilating_ops;
    result.has_absolute_terms = fs.has_absolute_value;
    result.has_stabilization = fs.has_stabilization;
    result.only_through_sym_grad = fs.only_through_sym_grad;
    result.has_plain_grad = fs.has_plain_grad;
    result.has_time_derivative = fs.has_time_derivative;
    result.occurrence_count = fs.occurrence_count;
    result.field_value_dimension = fs.value_dimension;

    return result;
}

} // namespace forms
} // namespace FE
} // namespace svmp

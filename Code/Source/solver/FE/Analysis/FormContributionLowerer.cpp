/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/FormContributionLowerer.h"
#include "Analysis/FormStructureAnalyzer.h"
#include "Analysis/FormExprScanner.h"
#include "Forms/FormExpr.h"

#include <map>

namespace svmp {
namespace FE {
namespace analysis {

std::vector<ContributionDescriptor>
lowerFormulation(const FormulationRecord& rec) {
    std::vector<ContributionDescriptor> contributions;

    if (!rec.residual_expr) return contributions;

    FormStructureAnalyzer fsa;

    // Run full analysis on the residual once to get per-field summaries
    // including test-side structure (self_adjoint_pattern, etc.)
    // This provides accurate trait flags for all fields.
    forms::FormExpr residual_expr(std::const_pointer_cast<forms::FormExprNode>(rec.residual_expr));
    std::map<FieldId, FieldOperatorSummary> field_summaries;
    if (residual_expr.isValid()) {
        auto full_summary = fsa.analyze(residual_expr, rec.active_fields);
        for (const auto& fs : full_summary.per_field) {
            field_summaries[fs.field] = fs;
        }
    }

    if (!rec.block_residual_exprs.empty()) {
        for (const auto& [block_key, block_node] : rec.block_residual_exprs) {
            if (!block_node) continue;
            FieldId test_fid = block_key.first;
            FieldId nominal_trial_fid = block_key.second;

            // For multi-field per-test blocks stored as (test, test) pseudo-blocks,
            // analyze ALL active fields to detect cross-coupling contributions.
            // This produces true (test, trial) contributions for each trial field
            // that appears in this test block.
            std::vector<FieldId> trial_fields_to_analyze;
            if (nominal_trial_fid == test_fid && rec.active_fields.size() > 1) {
                // Multi-field pseudo-block: check all active fields
                for (FieldId fid : rec.active_fields) {
                    auto trial_fs = fsa.analyzeField(*block_node, fid);
                    if (trial_fs.occurrence_count > 0) {
                        trial_fields_to_analyze.push_back(fid);
                    }
                }
            } else {
                // Single-field or explicit (test, trial) block: check if the
                // nominal trial field actually appears before analyzing it.
                // analyzeField counts StateField/TrialFunction nodes by FieldId,
                // but TrialFunction(space, "name") nodes have no FieldId. Use
                // hasTrial() on the block expression as a fallback.
                auto trial_fs = fsa.analyzeField(*block_node, nominal_trial_fid);
                if (trial_fs.occurrence_count > 0) {
                    trial_fields_to_analyze.push_back(nominal_trial_fid);
                } else {
                    // Fallback: check for untagged TrialFunction nodes
                    forms::FormExpr block_expr(
                        std::const_pointer_cast<forms::FormExprNode>(block_node));
                    if (block_expr.hasTrial()) {
                        trial_fields_to_analyze.push_back(nominal_trial_fid);
                    }
                }
            }

            // Pure-source rows (e.g., f*v) reference no active fields as
            // trial. Emit a source-like contribution for the test field so
            // analysis is aware of the forcing term.
            if (trial_fields_to_analyze.empty()) {
                ContributionDescriptor d;
                d.operator_tag = rec.operator_tag;
                d.test_variables = {VariableKey::field(test_fid)};
                d.trial_variables = {};
                d.role = ContributionRole::DiagonalBlock;
                d.traits = OperatorTraitFlags::HasMass;
                d.confidence = AnalysisConfidence::Medium;
                d.source_block_key = block_key;
                d.source_expression = block_node;

                {
                    std::string test_name;
                    for (const auto& [fid, name] : rec.field_names) {
                        if (fid == test_fid) { test_name = name; break; }
                    }
                    if (!test_name.empty()) {
                        d.origin = "FormsInstaller(source, test=" + test_name + ")";
                        d.block_context = "source term for test=" + test_name;
                    } else {
                        d.origin = "FormsInstaller(source, field " + std::to_string(test_fid) + ")";
                        d.block_context = "source term for field(" + std::to_string(test_fid) + ")";
                    }
                }

                // Scan for domain info
                auto scan_src = scanFormExpr(*block_node);
                if (!scan_src.boundary_markers.empty()) {
                    d.boundary_marker = scan_src.boundary_markers[0];
                    d.domain = DomainKind::Boundary;
                }

                contributions.push_back(std::move(d));
                continue;
            }

            // Scan for boundary/interface markers on the block node
            auto scan = scanFormExpr(*block_node);

            for (FieldId trial_fid : trial_fields_to_analyze) {

            // Analyze the trial field within THIS specific block node, not
            // the full residual. This ensures off-diagonal blocks get traits
            // from the actual coupling terms, not from other test blocks.
            // For test-side info (self_adjoint_pattern), merge from the full summary.
            FieldOperatorSummary fs = fsa.analyzeField(*block_node, trial_fid);
            auto full_it = field_summaries.find(trial_fid);
            if (full_it != field_summaries.end()) {
                // Inherit test-side flags from the full analysis
                fs.test_has_gradient = full_it->second.test_has_gradient;
                fs.test_has_sym_grad = full_it->second.test_has_sym_grad;
                fs.test_has_absolute_value = full_it->second.test_has_absolute_value;
                fs.self_adjoint_pattern = full_it->second.self_adjoint_pattern;
            }

            ContributionDescriptor d;
            d.operator_tag = rec.operator_tag;
            d.test_variables = {VariableKey::field(test_fid)};
            d.trial_variables = {VariableKey::field(trial_fid)};

            // Source provenance: link back to the block expression
            d.source_block_key = block_key;
            d.source_expression = block_node;

            // Build origin and block_context with field names when available
            {
                std::string test_name, trial_name;
                for (const auto& [fid, name] : rec.field_names) {
                    if (fid == test_fid) test_name = name;
                    if (fid == trial_fid) trial_name = name;
                }
                if (!test_name.empty() || !trial_name.empty()) {
                    d.origin = "FormsInstaller(test=" + test_name + ", trial=" + trial_name + ")";
                    d.block_context = "test=" + test_name + " (field " +
                        std::to_string(test_fid) + "), trial=" + trial_name +
                        " (field " + std::to_string(trial_fid) + ")";
                } else {
                    d.origin = "FormsInstaller(block " + std::to_string(test_fid) +
                               "," + std::to_string(trial_fid) + ")";
                    d.block_context = "block(" + std::to_string(test_fid) +
                                      ", " + std::to_string(trial_fid) + ")";
                }
            }

            // Classify role
            bool is_diagonal = (test_fid == trial_fid);
            if (is_diagonal) {
                if (rec.has_stabilization_terms && fs.has_stabilization) {
                    d.role = ContributionRole::StabilizationBlock;
                } else if (fs.only_through_annihilating_ops && !fs.has_absolute_value) {
                    d.role = ContributionRole::DiagonalBlock;
                } else if (fs.has_absolute_value && !fs.has_gradient && !fs.has_sym_grad) {
                    // No own gradient — constraint/multiplier (e.g., pressure pp block)
                    d.role = ContributionRole::ConstraintBlock;
                } else {
                    d.role = ContributionRole::DiagonalBlock;
                }
            } else {
                // Off-diagonal: check if this is a constraint coupling
                if (fs.has_absolute_value && !fs.has_gradient) {
                    d.role = ContributionRole::ConstraintBlock;
                } else {
                    d.role = ContributionRole::OffDiagonalBlock;
                }
            }

            // Set traits from FieldOperatorSummary
            auto flags = OperatorTraitFlags::None;

            if (fs.only_through_annihilating_ops && !fs.has_absolute_value) {
                flags = flags | OperatorTraitFlags::HasSecondOrder;
                if (fs.self_adjoint_pattern) {
                    flags = flags | OperatorTraitFlags::SymmetricLike;
                    flags = flags | OperatorTraitFlags::PositiveSemiDefiniteLike;
                }
            }
            if (fs.has_absolute_value && fs.has_gradient) {
                flags = flags | OperatorTraitFlags::HasFirstOrder;
            }
            if (fs.has_absolute_value && !fs.has_gradient && !fs.has_sym_grad &&
                !fs.has_divergence && !fs.has_curl) {
                flags = flags | OperatorTraitFlags::HasMass;
            }
            if (fs.only_through_sym_grad && !fs.has_plain_grad && fs.self_adjoint_pattern) {
                flags = flags | OperatorTraitFlags::SymmetricLike;
            }

            d.traits = flags;
            d.confidence = AnalysisConfidence::High;

            // Infer temporal metadata from time-derivative flags
            if (fs.has_time_derivative) {
                d.temporal = TemporalDescriptor{
                    fs.time_derivative_order > 0 ? fs.time_derivative_order : 1,
                    TemporalContributionKind::MassLike};
                d.traits = d.traits | OperatorTraitFlags::HasMass;
            } else if (d.role == ContributionRole::ConstraintBlock) {
                // Constraint blocks (diagonal or off-diagonal) are algebraic —
                // mark as PureConstraint so DAEStructureAnalyzer sees constraint
                // variables even when they only appear in off-diagonal blocks.
                d.temporal = TemporalDescriptor{0, TemporalContributionKind::PureConstraint};
            } else if (is_diagonal) {
                // Steady contribution (no time derivative): mark as algebraic/none
                // so DAEStructureAnalyzer can distinguish steady PDE fields from
                // dynamic ODE fields in coupled PDE-ODE systems.
                d.temporal = TemporalDescriptor{0, TemporalContributionKind::None};
            }

            // Infer transport character from first-order operators
            if (fs.has_absolute_value && fs.has_gradient && !fs.only_through_annihilating_ops) {
                d.transport_character = TransportCharacter::DirectionalFirstOrder;
                d.traits = d.traits | OperatorTraitFlags::HasFirstOrder;
            }

            // Infer pairing for off-diagonal and constraint blocks.
            // Use operator_tag + field IDs as pairing_group to distinguish
            // unrelated constraint pairs in multi-physics systems.
            // Classify as FormalAdjointPair when the TRIAL side of THIS BLOCK
            // couples through differential operators (grad/div/sym_grad).
            // NOTE: we only use trial-side analysis here because `fs` is
            // block-local (from analyzeField(*block_node, trial_fid)), while
            // test-side flags in field_summaries are whole-form and would
            // incorrectly promote unrelated off-diagonal pairings. In a
            // saddle-point system (e.g. NS), the PV block (div(u)*q_test) gets
            // FormalAdjointPair from trial-side divergence; the VP block
            // (p*div(v_test)) gets ConstraintPair since pressure enters
            // algebraically. InfSupAnalyzer accepts both kinds.
            if (d.role == ContributionRole::ConstraintBlock ||
                (d.role == ContributionRole::OffDiagonalBlock && !is_diagonal)) {
                PairingDescriptor pd;
                pd.row_var = VariableKey::field(test_fid);
                pd.col_var = VariableKey::field(trial_fid);

                // Block-local trial-side differential coupling detection
                bool has_differential_coupling =
                    fs.has_gradient || fs.has_divergence || fs.has_sym_grad ||
                    fs.has_curl;

                pd.kind = has_differential_coupling
                    ? PairingKind::FormalAdjointPair
                    : PairingKind::ConstraintPair;

                // Record whether the trial field also appears undifferentiated.
                // This is true for ConstraintPair (always) and for mixed blocks
                // like the NS-VMS VP block where p appears both in p*div(v)
                // (undifferentiated) and in τ_m*grad(v)·grad(p) (differentiated).
                pd.trial_has_undifferentiated = fs.has_absolute_value;

                pd.pairing_group = rec.operator_tag + "_" +
                    std::to_string(test_fid) + "_" + std::to_string(trial_fid);
                d.pairings.push_back(std::move(pd));
            }

            // Set boundary/interface markers from scan
            if (!scan.boundary_markers.empty()) {
                d.boundary_marker = scan.boundary_markers[0];
                d.domain = DomainKind::Boundary;
            }
            if (!scan.interface_markers.empty()) {
                d.interface_marker = scan.interface_markers[0];
                d.domain = DomainKind::InterfaceFace;
            }

            // Emit nullspace hints for fields through annihilating ops
            if (is_diagonal && fs.only_through_annihilating_ops &&
                !fs.has_absolute_value && !fs.has_time_derivative) {
                NullspaceHint nh;
                nh.field = trial_fid;
                nh.confidence = fs.has_stabilization ? AnalysisConfidence::Medium
                                                     : AnalysisConfidence::High;
                if (fs.only_through_sym_grad && !fs.has_plain_grad && fs.value_dimension > 1) {
                    nh.family = NullspaceFamily::KernelOfSymGrad;
                    nh.reason = "Vector field only through sym(grad) — rigid-body nullspace";
                } else if (fs.value_dimension > 1) {
                    nh.family = NullspaceFamily::ComponentwiseConstant;
                    nh.reason = "Vector field only through annihilating ops — componentwise constant nullspace";
                } else {
                    nh.family = NullspaceFamily::ScalarConstant;
                    nh.reason = "Scalar field only through annihilating ops — constant nullspace";
                }
                d.nullspace_hints.push_back(std::move(nh));
            }

            contributions.push_back(std::move(d));
            } // end for trial_fields_to_analyze
        }
    } else if (!rec.active_fields.empty()) {
        // No per-block expressions — analyze the full residual per field.
        // Must emit the same Phase 21-24 metadata as the block path above:
        // temporal, transport, nullspace hints, first-order/mass trait flags.
        for (FieldId fid : rec.active_fields) {
            auto fs = fsa.analyzeField(*rec.residual_expr, fid);
            if (fs.occurrence_count == 0) continue;

            ContributionDescriptor d;
            d.operator_tag = rec.operator_tag;
            d.test_variables = {VariableKey::field(fid)};
            d.trial_variables = {VariableKey::field(fid)};
            d.role = ContributionRole::DiagonalBlock;

            // Source provenance: diagonal block from full residual
            d.source_block_key = std::make_pair(fid, fid);
            d.source_expression = rec.residual_expr;

            {
                std::string field_name;
                for (const auto& [f, name] : rec.field_names) {
                    if (f == fid) { field_name = name; break; }
                }
                if (!field_name.empty()) {
                    d.origin = "FormsInstaller(field=" + field_name + ")";
                    d.block_context = "field=" + field_name + " (field " + std::to_string(fid) + ")";
                } else {
                    d.origin = "FormsInstaller(field " + std::to_string(fid) + ")";
                    d.block_context = "field(" + std::to_string(fid) + ")";
                }
            }

            auto flags = OperatorTraitFlags::None;
            if (fs.only_through_annihilating_ops && !fs.has_absolute_value) {
                flags = flags | OperatorTraitFlags::HasSecondOrder;
                if (fs.self_adjoint_pattern) {
                    flags = flags | OperatorTraitFlags::SymmetricLike
                          | OperatorTraitFlags::PositiveSemiDefiniteLike;
                }
            }
            if (fs.has_absolute_value && fs.has_gradient) {
                flags = flags | OperatorTraitFlags::HasFirstOrder;
            }
            if (fs.has_absolute_value && !fs.has_gradient && !fs.has_sym_grad &&
                !fs.has_divergence && !fs.has_curl) {
                flags = flags | OperatorTraitFlags::HasMass;
            }
            d.traits = flags;
            d.confidence = AnalysisConfidence::High;

            // Temporal metadata
            if (fs.has_time_derivative) {
                d.temporal = TemporalDescriptor{
                    fs.time_derivative_order > 0 ? fs.time_derivative_order : 1,
                    TemporalContributionKind::MassLike};
                d.traits = d.traits | OperatorTraitFlags::HasMass;
            } else {
                d.temporal = TemporalDescriptor{0, TemporalContributionKind::None};
            }

            // Transport character
            if (fs.has_absolute_value && fs.has_gradient &&
                !fs.only_through_annihilating_ops) {
                d.transport_character = TransportCharacter::DirectionalFirstOrder;
                d.traits = d.traits | OperatorTraitFlags::HasFirstOrder;
            }

            // Nullspace hints
            if (fs.only_through_annihilating_ops &&
                !fs.has_absolute_value && !fs.has_time_derivative) {
                NullspaceHint nh;
                nh.field = fid;
                nh.confidence = fs.has_stabilization ? AnalysisConfidence::Medium
                                                     : AnalysisConfidence::High;
                if (fs.only_through_sym_grad && !fs.has_plain_grad &&
                    fs.value_dimension > 1) {
                    nh.family = NullspaceFamily::KernelOfSymGrad;
                    nh.reason = "Vector field only through sym(grad) — "
                                "rigid-body nullspace";
                } else if (fs.value_dimension > 1) {
                    nh.family = NullspaceFamily::ComponentwiseConstant;
                    nh.reason = "Vector field only through annihilating ops — "
                                "componentwise constant nullspace";
                } else {
                    nh.family = NullspaceFamily::ScalarConstant;
                    nh.reason = "Scalar field only through annihilating ops — "
                                "constant nullspace";
                }
                d.nullspace_hints.push_back(std::move(nh));
            }

            contributions.push_back(std::move(d));
        }

        // NOTE: No cross-field pairing emission here. Without block-split
        // expressions, analyzeField(*residual, trial_fid) scans the WHOLE
        // residual and cannot tell which test equation a trial field appears
        // in, so every field pair would get a phantom coupling. For multi-field
        // inf-sup detection on the fallback path, InfSupAnalyzer's
        // MixedSaddlePoint fallback (from MixedOperatorAnalyzer) is the
        // correct mechanism — it operates on structural claims rather than
        // fabricated contribution metadata.
    }

    return contributions;
}

} // namespace analysis
} // namespace FE
} // namespace svmp

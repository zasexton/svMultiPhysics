/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/FormContributionLowerer.h"
#include "Analysis/FormStructureAnalyzer.h"
#include "Analysis/FormExprScanner.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

[[nodiscard]] bool appendNullspaceHintIfPresent(ContributionDescriptor& d,
                                                FieldId field,
                                                const FieldOperatorSummary& fs)
{
    if (!fs.only_through_annihilating_ops || fs.has_absolute_value || fs.has_time_derivative) {
        return false;
    }

    NullspaceHint nh;
    nh.field = field;
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
    return true;
}

[[nodiscard]] forms::FormExpr extractNonBoundaryTerms(const forms::FormExpr& expr)
{
    if (!expr.isValid() || !expr.hasTest() || expr.hasTrial()) {
        return {};
    }

    try {
        forms::FormCompiler compiler;
        const auto ir = compiler.compileLinear(expr);

        forms::FormExpr filtered;
        for (const auto& term : ir.terms()) {
            if (term.domain == forms::IntegralDomain::Boundary) {
                continue;
            }

            forms::FormExpr wrapped;
            switch (term.domain) {
                case forms::IntegralDomain::Cell:
                    wrapped = term.integrand.dx();
                    break;
                case forms::IntegralDomain::InteriorFace:
                    wrapped = term.integrand.dS();
                    break;
                case forms::IntegralDomain::InterfaceFace:
                    wrapped = term.integrand.dI(term.interface_marker);
                    break;
                case forms::IntegralDomain::Boundary:
                    break;
            }

            if (!wrapped.isValid()) {
                continue;
            }
            filtered = filtered.isValid() ? (filtered + wrapped) : wrapped;
        }

        return filtered;
    } catch (const std::exception&) {
        return {};
    }
}

void appendBoundaryInsensitiveNullspaceHint(ContributionDescriptor& d,
                                            FieldId field,
                                            const FieldOperatorSummary& fs,
                                            const forms::FormExpr& expr,
                                            bool has_boundary_terms,
                                            FormStructureAnalyzer& fsa)
{
    if (appendNullspaceHintIfPresent(d, field, fs) || !has_boundary_terms || !fs.has_absolute_value) {
        return;
    }

    const auto interior_expr = extractNonBoundaryTerms(expr);
    if (!interior_expr.isValid() || !interior_expr.node()) {
        return;
    }

    const auto filtered = fsa.analyzeField(*interior_expr.node(), field);
    appendNullspaceHintIfPresent(d, field, filtered);
}

[[nodiscard]] bool isFieldVariable(const VariableKey& key)
{
    return key.kind == VariableKind::FieldComponent &&
           key.field_id != INVALID_FIELD_ID;
}

void attachStabilizationSurrogates(std::vector<ContributionDescriptor>& contributions)
{
    std::set<FieldId> stabilized_fields;

    for (const auto& contribution : contributions) {
        if (contribution.role != ContributionRole::StabilizationBlock) {
            continue;
        }

        for (const auto& key : contribution.test_variables) {
            if (isFieldVariable(key)) {
                stabilized_fields.insert(key.field_id);
            }
        }
        for (const auto& key : contribution.trial_variables) {
            if (isFieldVariable(key)) {
                stabilized_fields.insert(key.field_id);
            }
        }
    }

    if (stabilized_fields.empty()) {
        return;
    }

    for (auto& contribution : contributions) {
        for (auto& pairing : contribution.pairings) {
            const bool row_stabilized =
                isFieldVariable(pairing.row_var) &&
                stabilized_fields.count(pairing.row_var.field_id) != 0;
            const bool col_stabilized =
                isFieldVariable(pairing.col_var) &&
                stabilized_fields.count(pairing.col_var.field_id) != 0;
            if (row_stabilized || col_stabilized) {
                pairing.has_stabilizing_surrogate = true;
            }
        }
    }
}

void attachRuntimeMetadata(ContributionDescriptor& d,
                           const FormExprScanResult& scan)
{
    d.parameter_usages = scan.parameter_usages;
    d.coefficient_usages = scan.coefficient_usages;
    d.scale_usages = scan.scale_usages;

    if (!scan.scale_usages.empty()) {
        ScalingDescriptor scaling;
        for (const auto& scale : scan.scale_usages) {
            if (std::abs(scale.h_power) > std::abs(scaling.h_power)) {
                scaling.h_power = scale.h_power;
            }
            if (std::abs(scale.dt_power) > std::abs(scaling.dt_power)) {
                scaling.dt_power = scale.dt_power;
            }
            scaling.parameter_scaled =
                scaling.parameter_scaled ||
                !scale.parameter_names.empty() ||
                !scale.parameter_slots.empty();
            scaling.coefficient_scaled =
                scaling.coefficient_scaled || !scale.coefficient_names.empty();
        }
        d.scaling = scaling;
    } else if (!scan.parameter_usages.empty() ||
               !scan.coefficient_usages.empty() ||
               scan.has_cell_diameter ||
               scan.has_time_derivative) {
        ScalingDescriptor scaling;
        scaling.h_power = scan.has_cell_diameter ? 1 : 0;
        scaling.parameter_scaled = !scan.parameter_usages.empty();
        scaling.coefficient_scaled = !scan.coefficient_usages.empty();
        d.scaling = scaling;
    }
}

void appendDomainScope(ContributionDescriptor& d,
                       DomainKind domain,
                       int marker,
                       std::string subexpression_id)
{
    const auto exists = std::any_of(
        d.domain_scopes.begin(),
        d.domain_scopes.end(),
        [&](const ContributionDomainScope& scope) {
            return scope.domain == domain &&
                   scope.marker == marker &&
                   scope.subexpression_id == subexpression_id;
        });
    if (!exists) {
        d.domain_scopes.push_back(
            ContributionDomainScope{domain, marker, std::move(subexpression_id)});
    }
}

void attachDomainScopes(ContributionDescriptor& d,
                        const FormExprScanResult& scan)
{
    if (scan.has_cell_integral) {
        appendDomainScope(d, DomainKind::Cell, -1, "cell");
    }
    if (scan.has_boundary_integral) {
        if (scan.boundary_markers.empty()) {
            appendDomainScope(d, DomainKind::Boundary, -1, "boundary");
        } else {
            for (const int marker : scan.boundary_markers) {
                appendDomainScope(d, DomainKind::Boundary, marker, "boundary");
            }
        }
    }
    if (scan.has_interior_face_integral) {
        appendDomainScope(d, DomainKind::InteriorFace, -1, "interior-face");
    }
    if (scan.has_interface_integral) {
        if (scan.interface_markers.empty()) {
            appendDomainScope(d, DomainKind::InterfaceFace, -1, "interface");
        } else {
            for (const int marker : scan.interface_markers) {
                appendDomainScope(d, DomainKind::InterfaceFace, marker, "interface");
            }
        }
    }
    if (d.domain_scopes.empty()) {
        appendDomainScope(d, DomainKind::Cell, -1, "default-cell");
    }

    if (scan.has_interface_integral) {
        d.domain = DomainKind::InterfaceFace;
        d.interface_marker =
            scan.interface_markers.empty() ? -1 : scan.interface_markers.front();
        return;
    }
    if (scan.has_boundary_integral) {
        d.domain = DomainKind::Boundary;
        d.boundary_marker =
            scan.boundary_markers.empty() ? -1 : scan.boundary_markers.front();
        return;
    }
    if (scan.has_interior_face_integral) {
        d.domain = DomainKind::InteriorFace;
        return;
    }
    d.domain = DomainKind::Cell;
}

} // namespace

std::vector<ContributionDescriptor>
lowerFormulation(const FormulationRecord& rec) {
    std::vector<ContributionDescriptor> contributions;

    if (!rec.residual_expr) return contributions;

    FormStructureAnalyzer fsa;

    // Run full analysis on the residual once to get per-field summaries
    // including test-side structure (self_adjoint_pattern, etc.)
    // This provides accurate trait flags for all fields.
    forms::FormExpr residual_expr(std::const_pointer_cast<forms::FormExprNode>(rec.residual_expr));
    const bool residual_has_boundary_terms =
        residual_expr.isValid() && residual_expr.node() &&
        scanFormExpr(*residual_expr.node()).has_boundary_integral;
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
            forms::FormExpr block_expr(std::const_pointer_cast<forms::FormExprNode>(block_node));

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
                d.role = ContributionRole::SourceVector;
                d.traits = OperatorTraitFlags::SourceLike;
                d.confidence = AnalysisConfidence::Medium;
                d.source_block_key = block_key;
                d.source_expression = block_node;
                d.temporal = TemporalDescriptor{
                    0, TemporalContributionKind::TimeIndependentResidual};
                d.balance = BalanceDescriptor{
                    "", BalanceRole::SourceLike, 1, false};

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

                auto scan_src = scanFormExpr(*block_node);
                attachDomainScopes(d, scan_src);
                if (d.domain == DomainKind::Boundary ||
                    d.domain == DomainKind::InterfaceFace) {
                    d.traits = d.traits | OperatorTraitFlags::BoundaryFluxLike;
                }
                attachRuntimeMetadata(d, scan_src);

                d.ensureStableContributionId();
                contributions.push_back(std::move(d));
                continue;
            }

            // Scan for boundary/interface markers on the block node
            auto scan = scanFormExpr(*block_node);
            const bool block_has_boundary_terms = scan.has_boundary_integral;

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

            const bool block_has_stabilization =
                rec.has_stabilization_terms && fs.has_stabilization;
            const bool emit_separate_stabilization_surrogate =
                block_has_stabilization && rec.active_fields.size() > 1;

            // Classify role
            bool is_diagonal = (test_fid == trial_fid);
            if (is_diagonal) {
                if (block_has_stabilization &&
                    !emit_separate_stabilization_surrogate) {
                    d.role = ContributionRole::StabilizationBlock;
                } else {
                    d.role = ContributionRole::DiagonalBlock;
                }
            } else {
                d.role = ContributionRole::OffDiagonalBlock;
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
            if (scan.has_cell_diameter) {
                flags = flags | OperatorTraitFlags::MeshScaleDependentHint;
            }
            if (d.role == ContributionRole::StabilizationBlock) {
                flags = flags | OperatorTraitFlags::StabilizationLike;
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
                d.temporal = TemporalDescriptor{
                    0, TemporalContributionKind::PureAlgebraicConstraint};
            } else if (is_diagonal) {
                d.temporal = TemporalDescriptor{
                    0, TemporalContributionKind::TimeIndependentResidual};
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
                d.role == ContributionRole::OffDiagonalBlock ||
                d.role == ContributionRole::InterfaceCoupling) {
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

            attachDomainScopes(d, scan);
            if (!is_diagonal &&
                d.role == ContributionRole::OffDiagonalBlock &&
                d.domain == DomainKind::InterfaceFace) {
                d.role = ContributionRole::InterfaceCoupling;
            }
            attachRuntimeMetadata(d, scan);

            // Emit nullspace hints for fields through annihilating ops
            if (is_diagonal) {
                appendBoundaryInsensitiveNullspaceHint(
                    d, trial_fid, fs, block_expr, block_has_boundary_terms, fsa);
            }

            d.ensureStableContributionId();
            contributions.push_back(std::move(d));

            if (emit_separate_stabilization_surrogate) {
                ContributionDescriptor stab;
                stab.operator_tag = rec.operator_tag;
                stab.origin = "FormsInstaller(stabilization surrogate, " +
                              contributions.back().block_context + ")";
                stab.domain = contributions.back().domain;
                stab.boundary_marker = contributions.back().boundary_marker;
                stab.interface_scope = contributions.back().interface_scope;
                stab.interface_marker = contributions.back().interface_marker;
                stab.domain_scopes = contributions.back().domain_scopes;
                stab.test_variables = contributions.back().test_variables;
                stab.trial_variables = contributions.back().trial_variables;
                stab.role = ContributionRole::StabilizationBlock;
                stab.traits = OperatorTraitFlags::HasSecondOrder |
                              OperatorTraitFlags::StabilizationLike;
                if (scan.has_cell_diameter) {
                    stab.traits =
                        stab.traits | OperatorTraitFlags::MeshScaleDependentHint;
                }
                stab.confidence = AnalysisConfidence::High;
                stab.temporal = TemporalDescriptor{
                    0, TemporalContributionKind::TimeIndependentResidual};
                stab.consistency_kind = ConsistencyKind::ConsistentPerturbation;
                stab.parameter_usages = contributions.back().parameter_usages;
                stab.coefficient_usages = contributions.back().coefficient_usages;
                stab.scale_usages = contributions.back().scale_usages;
                stab.scaling = contributions.back().scaling;
                stab.source_block_key = contributions.back().source_block_key;
                stab.source_expression = contributions.back().source_expression;
                stab.block_context = contributions.back().block_context;
                stab.ensureStableContributionId();
                contributions.push_back(std::move(stab));
            }
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
            auto full_scan = scanFormExpr(*rec.residual_expr);
            if (full_scan.has_cell_diameter) {
                flags = flags | OperatorTraitFlags::MeshScaleDependentHint;
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
                d.temporal = TemporalDescriptor{
                    0, TemporalContributionKind::TimeIndependentResidual};
            }

            // Transport character
            if (fs.has_absolute_value && fs.has_gradient &&
                !fs.only_through_annihilating_ops) {
                d.transport_character = TransportCharacter::DirectionalFirstOrder;
                d.traits = d.traits | OperatorTraitFlags::HasFirstOrder;
            }

            // Nullspace hints
            appendBoundaryInsensitiveNullspaceHint(
                d, fid, fs, residual_expr, residual_has_boundary_terms, fsa);

            attachDomainScopes(d, full_scan);
            attachRuntimeMetadata(d, full_scan);

            d.ensureStableContributionId();
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

    attachStabilizationSurrogates(contributions);
    return contributions;
}

} // namespace analysis
} // namespace FE
} // namespace svmp

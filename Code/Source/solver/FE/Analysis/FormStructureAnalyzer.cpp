/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/FormStructureAnalyzer.h"
#include "Analysis/FormExprScanner.h"
#include "Forms/FormExpr.h"

#include <algorithm>
#include <optional>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

[[nodiscard]] bool subtreeContainsCellDiameter(const forms::FormExprNode& node)
{
    using FT = forms::FormExprType;
    if (node.type() == FT::CellDiameter) {
        return true;
    }

    for (const auto* child : node.children()) {
        if (child && subtreeContainsCellDiameter(*child)) {
            return true;
        }
    }

    return false;
}

struct FieldRoleSets {
    std::vector<FieldId> test_fields;
    std::vector<FieldId> trial_fields;
};

void appendFieldIfAbsent(std::vector<FieldId>& fields, FieldId field)
{
    if (field == INVALID_FIELD_ID) {
        return;
    }
    if (std::find(fields.begin(), fields.end(), field) == fields.end()) {
        fields.push_back(field);
    }
}

void mergeFieldRoleSets(FieldRoleSets& dst, const FieldRoleSets& src)
{
    for (const auto field : src.test_fields) {
        appendFieldIfAbsent(dst.test_fields, field);
    }
    for (const auto field : src.trial_fields) {
        appendFieldIfAbsent(dst.trial_fields, field);
    }
}

[[nodiscard]] FieldRoleSets collectFieldRoleSets(
    const forms::FormExprNode& node)
{
    using FT = forms::FormExprType;

    FieldRoleSets sets;
    switch (node.type()) {
        case FT::TestFunction:
            if (const auto field = node.fieldId()) {
                appendFieldIfAbsent(sets.test_fields, *field);
            }
            break;
        case FT::DiscreteField:
        case FT::StateField:
        case FT::TrialFunction:
            if (const auto field = node.fieldId()) {
                appendFieldIfAbsent(sets.trial_fields, *field);
            }
            break;
        default:
            break;
    }

    for (const auto* child : node.children()) {
        if (!child) continue;
        mergeFieldRoleSets(sets, collectFieldRoleSets(*child));
    }
    return sets;
}

void appendCouplingIfAbsent(
    std::vector<std::pair<FieldId, FieldId>>& couplings,
    FieldId test_field,
    FieldId trial_field)
{
    if (test_field == INVALID_FIELD_ID || trial_field == INVALID_FIELD_ID ||
        test_field == trial_field) {
        return;
    }
    const auto edge = std::make_pair(test_field, trial_field);
    if (std::find(couplings.begin(), couplings.end(), edge) == couplings.end()) {
        couplings.push_back(edge);
    }
}

void collectObservedFieldCouplings(
    const forms::FormExprNode& node,
    std::vector<std::pair<FieldId, FieldId>>& couplings)
{
    using FT = forms::FormExprType;
    const auto ty = node.type();

    if (ty == FT::Multiply ||
        ty == FT::InnerProduct ||
        ty == FT::DoubleContraction ||
        ty == FT::OuterProduct ||
        ty == FT::CrossProduct) {
        const auto sets = collectFieldRoleSets(node);
        for (const auto test_field : sets.test_fields) {
            for (const auto trial_field : sets.trial_fields) {
                appendCouplingIfAbsent(couplings, test_field, trial_field);
            }
        }
    }

    for (const auto* child : node.children()) {
        if (child) {
            collectObservedFieldCouplings(*child, couplings);
        }
    }
}

} // namespace

// ============================================================================
// FormStructureAnalyzer::analyze
// ============================================================================

FormStructureSummary
FormStructureAnalyzer::analyze(const forms::FormExpr& residual,
                                std::span<const FieldId> fields) const
{
    FormStructureSummary summary;

    if (!residual.isValid() || !residual.node()) {
        return summary;
    }

    const auto& root = *residual.node();

    // Per-field analysis (trial-side)
    for (FieldId fid : fields) {
        auto field_summary = analyzeField(root, fid);
        summary.per_field.push_back(field_summary);

        if (field_summary.has_stabilization) {
            summary.has_stabilization = true;
        }
    }

    // Test-function analysis: walk the DAG to find TestFunction nodes and
    // track which differential operators they appear under. This lets us
    // detect self-adjoint patterns (test and trial under same operators).
    {
        struct TestWalkState {
            bool under_gradient{false};
            bool under_sym_part{false};
        };
        struct TestFieldMetadata {
            FieldId field{INVALID_FIELD_ID};
            bool has_gradient{false};
            bool has_sym_grad{false};
            bool has_absolute_value{false};
        };

        // Walk for TestFunction nodes and record their operator context
        bool unbound_test_has_gradient = false;
        bool unbound_test_has_sym_grad = false;
        bool unbound_test_has_absolute_value = false;
        bool any_bound_test_function = false;
        std::vector<TestFieldMetadata> test_metadata;

        const auto metadataFor = [&](FieldId field) -> TestFieldMetadata& {
            auto it = std::find_if(test_metadata.begin(), test_metadata.end(),
                [&](const auto& meta) { return meta.field == field; });
            if (it == test_metadata.end()) {
                test_metadata.push_back(TestFieldMetadata{field});
                return test_metadata.back();
            }
            return *it;
        };

        const auto walkTest = [&](const auto& self, const forms::FormExprNode& node,
                                   const TestWalkState& state) -> void {
            using FT = forms::FormExprType;
            if (node.type() == FT::TestFunction) {
                if (const auto field = node.fieldId()) {
                    any_bound_test_function = true;
                    auto& meta = metadataFor(*field);
                    if (!state.under_gradient) {
                        meta.has_absolute_value = true;
                    }
                    if (state.under_gradient) {
                        meta.has_gradient = true;
                        if (state.under_sym_part) {
                            meta.has_sym_grad = true;
                        }
                    }
                } else {
                    if (!state.under_gradient) {
                        unbound_test_has_absolute_value = true;
                    }
                    if (state.under_gradient) {
                        unbound_test_has_gradient = true;
                        if (state.under_sym_part) {
                            unbound_test_has_sym_grad = true;
                        }
                    }
                }
                return;
            }
            auto children = node.children();
            if (node.type() == FT::Gradient) {
                for (const auto* child : children) {
                    if (!child) continue;
                    TestWalkState s = state;
                    s.under_gradient = true;
                    self(self, *child, s);
                }
                return;
            }
            if (node.type() == FT::SymmetricPart) {
                for (const auto* child : children) {
                    if (!child) continue;
                    TestWalkState s = state;
                    s.under_sym_part = true;
                    self(self, *child, s);
                }
                return;
            }
            for (const auto* child : children) {
                if (child) self(self, *child, state);
            }
        };
        walkTest(walkTest, root, TestWalkState{});

        // Annotate each field summary with test-side info and self-adjoint check
        for (auto& fs : summary.per_field) {
            const auto meta_it = std::find_if(test_metadata.begin(), test_metadata.end(),
                [&](const auto& meta) { return meta.field == fs.field; });
            const bool use_single_field_unbound_metadata =
                !any_bound_test_function && fields.size() == 1u;

            if (meta_it != test_metadata.end()) {
                fs.test_has_gradient = meta_it->has_gradient;
                fs.test_has_sym_grad = meta_it->has_sym_grad;
                fs.test_has_absolute_value = meta_it->has_absolute_value;
            } else if (use_single_field_unbound_metadata) {
                fs.test_has_gradient = unbound_test_has_gradient;
                fs.test_has_sym_grad = unbound_test_has_sym_grad;
                fs.test_has_absolute_value = unbound_test_has_absolute_value;
            }

            // Self-adjoint pattern: both test and trial under the same operator
            if (fs.has_gradient && fs.test_has_gradient &&
                !fs.has_absolute_value && !fs.test_has_absolute_value) {
                fs.self_adjoint_pattern = true;
            }
            if (fs.has_sym_grad && fs.test_has_sym_grad &&
                !fs.has_absolute_value && !fs.test_has_absolute_value) {
                fs.self_adjoint_pattern = true;
            }
        }
    }

    // Scan for non-FE dependencies
    auto scan = scanFormExpr(root);
    for (const auto& name : scan.boundary_functional_names) {
        summary.boundary_functional_dependencies.push_back(
            VariableKey::named(VariableKind::BoundaryFunctional, name));
    }
    for (const auto& name : scan.auxiliary_state_names) {
        summary.auxiliary_state_dependencies.push_back(
            VariableKey::named(VariableKind::AuxiliaryState, name));
    }

    // Build variable couplings: FE fields ↔ boundary functionals / aux states
    for (FieldId fid : fields) {
        auto fk = VariableKey::field(fid);
        for (const auto& bf : summary.boundary_functional_dependencies) {
            summary.variable_couplings.emplace_back(fk, bf);
        }
        for (const auto& aux : summary.auxiliary_state_dependencies) {
            summary.variable_couplings.emplace_back(fk, aux);
        }
    }

    // Detect only observed mixed couplings. A field list alone is not evidence
    // of matrix-block or residual-block coupling. This pass records pairs only
    // when a field-bound test function and a trial/state field co-occur inside
    // the same product/contraction subtree.
    if (fields.size() > 1) {
        collectObservedFieldCouplings(root, summary.mixed_couplings);
        summary.has_saddle_point_structure = false;
    }

    return summary;
}

// ============================================================================
// FormStructureAnalyzer::analyzeField
// ============================================================================

FieldOperatorSummary
FormStructureAnalyzer::analyzeField(const forms::FormExprNode& root,
                                     FieldId target_field) const
{
    FieldOperatorSummary result;
    result.field = target_field;

    WalkState initial{};
    walkNode(root, target_field, initial, result);

    // Reset vacuously-true flags when no occurrences found
    if (result.occurrence_count == 0) {
        result.only_through_annihilating_ops = false;
        result.only_through_sym_grad = false;
    }

    return result;
}

// ============================================================================
// FormStructureAnalyzer::walkNode — generalized recursive DAG walk
// ============================================================================

void FormStructureAnalyzer::walkNode(const forms::FormExprNode& node,
                                      FieldId target_field,
                                      const WalkState& state,
                                      FieldOperatorSummary& result) const
{
    using FT = forms::FormExprType;
    const auto ty = node.type();

    // ---- Leaf: check if this node references the target field ----
    switch (ty) {
        case FT::DiscreteField:
        case FT::StateField:
        case FT::TrialFunction: {
            auto fid = node.fieldId();
            if (fid.has_value() && *fid == target_field) {
                result.occurrence_count++;

                // Capture field dimension
                const auto* sig = node.spaceSignature();
                if (sig && sig->value_dimension > result.value_dimension) {
                    result.value_dimension = sig->value_dimension;
                }

                // Operator classification
                if (!state.hasAnnihilatingOp() && !state.under_time_derivative) {
                    result.only_through_annihilating_ops = false;
                    result.has_absolute_value = true;
                }

                if (state.under_time_derivative) {
                    result.has_time_derivative = true;
                    if (state.time_derivative_order > result.time_derivative_order) {
                        result.time_derivative_order = state.time_derivative_order;
                    }
                }

                if (state.near_cell_diameter && state.hasAnnihilatingOp()) {
                    result.has_stabilization = true;
                }

                // Track specific operators
                if (state.under_gradient) {
                    result.has_gradient = true;
                    if (state.under_sym_part) {
                        result.has_sym_grad = true;
                    } else {
                        result.has_plain_grad = true;
                        result.only_through_sym_grad = false;
                    }
                } else if (state.under_divergence) {
                    result.has_divergence = true;
                    result.only_through_sym_grad = false;
                } else if (state.under_curl) {
                    result.has_curl = true;
                    result.only_through_sym_grad = false;
                } else if (state.under_hessian) {
                    result.has_hessian = true;
                    result.only_through_sym_grad = false;
                } else if (!state.under_time_derivative) {
                    result.only_through_sym_grad = false;
                }

                // DG / trace
                if (state.under_jump) result.has_jump = true;
                if (state.under_average) result.has_average = true;
                if (state.in_boundary_integral) result.has_trace_terms = true;
            }
            return;
        }

        case FT::TestFunction:
        case FT::PreviousSolutionRef:
            return;

        default:
            break;
    }

    // ---- Differential operators: propagate state ----
    auto children = node.children();

    switch (ty) {
        case FT::Gradient: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState s = state;
                s.under_gradient = true;
                walkNode(*child, target_field, s, result);
            }
            return;
        }
        case FT::Divergence: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState s = state;
                s.under_divergence = true;
                walkNode(*child, target_field, s, result);
            }
            return;
        }
        case FT::Curl: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState s = state;
                s.under_curl = true;
                walkNode(*child, target_field, s, result);
            }
            return;
        }
        case FT::Hessian: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState s = state;
                s.under_hessian = true;
                walkNode(*child, target_field, s, result);
            }
            return;
        }
        case FT::TimeDerivative: {
            int order = 1;
            auto opt = node.timeDerivativeOrder();
            if (opt.has_value()) order = *opt;
            for (const auto* child : children) {
                if (!child) continue;
                WalkState s = state;
                s.under_time_derivative = true;
                s.time_derivative_order = std::max(s.time_derivative_order, order);
                walkNode(*child, target_field, s, result);
            }
            return;
        }
        case FT::SymmetricPart: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState s = state;
                s.under_sym_part = true;
                walkNode(*child, target_field, s, result);
            }
            return;
        }
        case FT::Jump: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState s = state;
                s.under_jump = true;
                walkNode(*child, target_field, s, result);
            }
            return;
        }
        case FT::Average: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState s = state;
                s.under_average = true;
                walkNode(*child, target_field, s, result);
            }
            return;
        }
        case FT::BoundaryIntegral:
        case FT::InteriorFaceIntegral:
        case FT::InterfaceIntegral: {
            for (const auto* child : children) {
                if (!child) continue;
                WalkState s = state;
                s.in_boundary_integral = true;
                walkNode(*child, target_field, s, result);
            }
            return;
        }
        case FT::CellDiameter:
            return; // leaf

        default:
            break;
    }

    // ---- Stabilization context for product nodes ----
    bool subtree_has_cell_diameter = false;
    if (ty == FT::Multiply || ty == FT::InnerProduct || ty == FT::DoubleContraction) {
        for (const auto* child : children) {
            if (child && subtreeContainsCellDiameter(*child)) {
                subtree_has_cell_diameter = true;
                break;
            }
        }
    }

    // ---- Default: recurse into children ----
    for (const auto* child : children) {
        if (!child) continue;
        WalkState s = state;
        if (subtree_has_cell_diameter) {
            s.near_cell_diameter = true;
        }
        walkNode(*child, target_field, s, result);
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp

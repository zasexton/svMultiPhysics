/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_NULLSPACE_ANALYZER_H
#define SVMP_FE_FORMS_NULLSPACE_ANALYZER_H

/**
 * @file NullspaceAnalyzer.h
 * @brief Symbolic nullspace inference from FormExpr residual expressions
 *
 * NullspaceAnalyzer walks the FormExpr DAG for each field and classifies how
 * the field appears in the weak form.  This determines which canonical
 * transformations leave the weak form invariant (i.e., which modes are in the
 * operator's nullspace).
 *
 * Implemented:
 *   - Scalar constant modes: field appears ONLY through Gradient / Divergence /
 *     Curl / Hessian → ScalarConstant family.
 *   - Componentwise vector constant modes: vector field appears only through
 *     gradient-like operators → ComponentwiseConstant family.
 *   - Rigid-body modes: vector field appears ONLY through sym(grad(u)) →
 *     KernelOfSymGrad family (6 modes in 3D, 3 in 2D).
 *   - Absolute-value term detection (mass, Robin) → anchored.
 *   - Stabilization detection (CellDiameter-scaled terms) → near-nullspace.
 *
 * Future:
 *   - DG fields: fields appearing through Jump/Average operators.
 *   - ConstitutiveModel analysis: infer how constitutive outputs depend on
 *     field gradients vs absolute values.
 *   - Per-connected-component scope: replicate candidates for disconnected
 *     mesh regions (requires mesh topology, handled by GaugeRegistry).
 *
 * @see GaugeRegistry for candidate storage and enforcement
 */

#include "Forms/FormExpr.h"
#include "Constraints/GaugeRegistry.h"

#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

/**
 * @brief Classification of how a field appears in a weak-form expression
 */
struct FieldAppearanceClassification {
    FieldId field{INVALID_FIELD_ID};

    /// Field appears only through differential operators that annihilate constants
    /// (Gradient, Divergence, Curl, Hessian)
    bool only_through_annihilating_ops{true};

    /// Field appears through absolute-value terms (direct evaluation, mass-like
    /// inner products, Robin terms) — anchors the constant mode
    bool has_absolute_terms{false};

    /// Field appears through stabilization-like patterns (CellDiameter-scaled
    /// penalty terms) — weakly breaks the nullspace
    bool has_stabilization{false};

    /// Every occurrence of the field goes through sym(grad(...)).
    /// True only when ALL annihilating-op paths are sym_grad.
    /// When true AND only_through_annihilating_ops, the field has rigid-body
    /// mode nullspace (translations + rotations).
    bool only_through_sym_grad{true};

    /// At least one occurrence goes through grad (but NOT sym(grad))
    bool has_plain_grad{false};

    /// Field appears through time derivative — constant modes are annihilated
    /// in transient problems but this is NOT a steady-state nullspace
    bool has_time_derivative{false};

    /// Number of times the field was found in the DAG
    int occurrence_count{0};

    /// Dimension of the field's space signature (0 = unknown/not seen)
    int field_value_dimension{0};
};

/**
 * @brief Analyze FormExpr residual for nullspace modes
 *
 * Usage:
 * @code
 *   NullspaceAnalyzer analyzer;
 *   auto candidates = analyzer.analyze(residual, {field_p, field_u});
 *   for (auto& c : candidates) {
 *       registry.addCandidate(std::move(c));
 *   }
 * @endcode
 */
class NullspaceAnalyzer {
public:
    NullspaceAnalyzer() = default;

    /**
     * @brief Analyze a residual FormExpr for nullspace modes in the given fields
     *
     * @param residual  The weak-form residual expression (may reference multiple fields)
     * @param fields    FieldIds to analyze (test = trial, standard Galerkin)
     * @return          GaugeCandidates for fields with detected nullspace modes
     */
    [[nodiscard]] std::vector<gauge::GaugeCandidate>
    analyze(const FormExpr& residual, std::span<const FieldId> fields) const;

    /**
     * @brief Classify how a single field appears in a FormExpr DAG
     *
     * Lower-level interface for testing and introspection.
     */
    [[nodiscard]] FieldAppearanceClassification
    classifyField(const FormExprNode& root, FieldId target_field) const;

private:
    /**
     * @brief Recursive DAG walk state
     *
     * Tracks which differential operators have been applied between a field
     * leaf and the current position in the walk.
     */
    struct WalkState {
        bool under_gradient{false};
        bool under_divergence{false};
        bool under_curl{false};
        bool under_hessian{false};
        bool under_sym_part{false};
        bool under_time_derivative{false};
        bool near_cell_diameter{false};  ///< stabilization heuristic

        [[nodiscard]] bool hasAnnihilatingOp() const noexcept {
            return under_gradient || under_divergence || under_curl || under_hessian;
        }
    };

    /**
     * @brief Walk the DAG from root, looking for occurrences of target_field
     *
     * Each time the target field is found at a leaf, the current WalkState
     * records which operators were applied on the path from the root.
     */
    void walkNode(const FormExprNode& node,
                  FieldId target_field,
                  const WalkState& state,
                  FieldAppearanceClassification& result) const;
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_NULLSPACE_ANALYZER_H

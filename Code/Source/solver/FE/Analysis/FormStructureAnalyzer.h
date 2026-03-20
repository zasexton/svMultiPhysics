/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_FORM_STRUCTURE_ANALYZER_H
#define SVMP_FE_ANALYSIS_FORM_STRUCTURE_ANALYZER_H

/**
 * @file FormStructureAnalyzer.h
 * @brief Generalized FormExpr DAG analysis producing per-field/per-form summaries
 *
 * FormStructureAnalyzer walks the FormExpr DAG to produce per-field and
 * per-form structural summaries:
 *   - per-field FieldOperatorSummary (gradient, sym_grad, absolute, stabilization, ...)
 *   - per-block FormBlockSummary (mass-like, stiffness-like, ...)
 *   - aggregate FormStructureSummary (saddle-point, stabilization, couplings)
 *
 * KernelAnalyzer consumes FieldOperatorSummary to emit nullspace PropertyClaims.
 *
 * @see KernelAnalyzer for the nullspace detection pass
 * @see FormulationRecord for metadata persistence
 */

#include "Core/Types.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {

namespace forms {
class FormExpr;
class FormExprNode;
} // namespace forms

namespace analysis {

// ============================================================================
// Per-field operator summary
// ============================================================================

/**
 * @brief Summary of how a single field appears in a FormExpr DAG
 *
 * Classifies how a field appears in a weak-form expression,
 * including trace/jump/penalty/degree tracking.
 */
struct FieldOperatorSummary {
    FieldId field{INVALID_FIELD_ID};
    int value_dimension{0};                 ///< From SpaceSignature (0 = unknown)

    // ---- Differential operator usage ----
    bool has_gradient{false};               ///< Under Gradient
    bool has_divergence{false};             ///< Under Divergence
    bool has_curl{false};                   ///< Under Curl
    bool has_hessian{false};                ///< Under Hessian
    bool has_sym_grad{false};               ///< Under SymmetricPart(Gradient(...))
    bool only_through_sym_grad{true};       ///< ALL gradient paths are symmetric
    bool has_plain_grad{false};             ///< At least one non-symmetric gradient path

    // ---- Nullspace-relevant flags ----
    bool only_through_annihilating_ops{true}; ///< Only under Grad/Div/Curl/Hessian
    bool has_absolute_value{false};         ///< Appears without differential operators
    bool has_time_derivative{false};        ///< Under TimeDerivative
    int time_derivative_order{0};           ///< 0, 1, or 2

    // ---- DG / trace / boundary ----
    bool has_trace_terms{false};            ///< In boundary/trace integrals
    bool has_jump{false};                   ///< Under Jump operator
    bool has_average{false};                ///< Under Average operator

    // ---- Stabilization ----
    bool has_stabilization{false};          ///< Near CellDiameter
    bool has_penalty{false};                ///< In penalty-scaled terms

    int occurrence_count{0};                ///< Number of field leaf occurrences

    // ---- Test-side structure (for self-adjoint check) ----
    // These track how the TestFunction in the same form appears.
    // Populated by FormStructureAnalyzer::analyze() when the full
    // form (with both test and trial) is available.
    bool test_has_gradient{false};           ///< TestFunction under Gradient
    bool test_has_sym_grad{false};           ///< TestFunction under sym(grad)
    bool test_has_absolute_value{false};     ///< TestFunction without differential operators

    /// True when trial and test appear through the same operator pattern
    /// (both under gradient, or both under sym_grad). This is a necessary
    /// (but not sufficient) condition for self-adjointness.
    bool self_adjoint_pattern{false};
};

// ============================================================================
// Per-block summary
// ============================================================================

struct FormBlockSummary {
    FieldId test_field{INVALID_FIELD_ID};
    FieldId trial_field{INVALID_FIELD_ID};
    bool is_diagonal_block{false};

    std::vector<FieldOperatorSummary> per_field;

    bool has_skew_symmetric_terms{false};
    bool has_mass_like_terms{false};        ///< Both test and trial without derivatives
    bool has_stiffness_like_terms{false};   ///< Both test and trial under gradient
    std::string trial_degree_classification{"unknown"}; ///< "linear", "nonlinear", "quasilinear"
};

// ============================================================================
// Aggregate summary
// ============================================================================

struct FormStructureSummary {
    std::vector<FieldOperatorSummary> per_field;
    std::vector<FormBlockSummary> per_block;
    std::vector<std::pair<FieldId, FieldId>> mixed_couplings;

    std::vector<VariableKey> boundary_functional_dependencies;
    std::vector<VariableKey> auxiliary_state_dependencies;
    std::vector<VariableKey> global_scalar_dependencies;
    std::vector<std::pair<VariableKey, VariableKey>> variable_couplings;

    bool has_stabilization{false};
    bool has_saddle_point_structure{false};
};

// ============================================================================
// FormStructureAnalyzer
// ============================================================================

/**
 * @brief Generalized FormExpr DAG analyzer
 *
 * Walks the DAG once per field to produce FieldOperatorSummary, then
 * aggregates into FormStructureSummary.
 *
 * Usage:
 * @code
 *   FormStructureAnalyzer analyzer;
 *   auto summary = analyzer.analyze(residual, {field_u, field_p});
 *   // summary.per_field[0] describes how field_u appears
 *   // summary.has_saddle_point_structure tells if mixed indefinite
 * @endcode
 */
class FormStructureAnalyzer {
public:
    FormStructureAnalyzer() = default;

    /**
     * @brief Analyze a residual FormExpr for all given fields
     */
    [[nodiscard]] FormStructureSummary
    analyze(const forms::FormExpr& residual, std::span<const FieldId> fields) const;

    /**
     * @brief Classify how a single field appears in a FormExpr DAG
     */
    [[nodiscard]] FieldOperatorSummary
    analyzeField(const forms::FormExprNode& root, FieldId target_field) const;

private:
    struct WalkState {
        bool under_gradient{false};
        bool under_divergence{false};
        bool under_curl{false};
        bool under_hessian{false};
        bool under_sym_part{false};
        bool under_time_derivative{false};
        bool near_cell_diameter{false};
        bool under_jump{false};
        bool under_average{false};
        bool in_boundary_integral{false};
        int time_derivative_order{0};

        [[nodiscard]] bool hasAnnihilatingOp() const noexcept {
            return under_gradient || under_divergence || under_curl || under_hessian;
        }
    };

    void walkNode(const forms::FormExprNode& node,
                  FieldId target_field,
                  const WalkState& state,
                  FieldOperatorSummary& result) const;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_FORM_STRUCTURE_ANALYZER_H

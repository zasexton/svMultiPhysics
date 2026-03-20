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
 * NullspaceAnalyzer classifies how fields appear in a weak form to determine
 * which canonical transformations leave the weak form invariant (nullspace modes).
 *
 * The heavy DAG walking is delegated to FormStructureAnalyzer (Analysis module).
 * NullspaceAnalyzer consumes the FieldOperatorSummary and maps it to
 * GaugeCandidate objects for the GaugeRegistry enforcement pipeline.
 *
 * NOTE: This class is no longer called in production code. Gauge candidate
 * population is handled by the ContributionDescriptor/NullspaceHint path:
 *   FormsInstaller → FormContributionLowerer → NullspaceHint
 *   → SystemSetup NullspaceHint→GaugeCandidate conversion
 * The FormContributionLowerer uses FormStructureAnalyzer with the same
 * classification logic as this class. NullspaceAnalyzer is retained as a
 * standalone utility for tests and direct analysis use.
 *
 * Implemented:
 *   - Scalar constant modes (ScalarConstant)
 *   - Componentwise vector constant modes (ComponentwiseConstant)
 *   - Rigid-body modes via sym(grad) (KernelOfSymGrad)
 *   - Absolute-value term detection → anchored
 *   - Stabilization detection → near-nullspace (Medium confidence)
 *
 * @see FormStructureAnalyzer for the DAG walker
 * @see FormContributionLowerer for the production nullspace hint emission path
 * @see GaugeRegistry for candidate storage and enforcement
 */

#include "Forms/FormExpr.h"
#include "Constraints/GaugeRegistry.h"
#include "Analysis/FormStructureAnalyzer.h"

#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

/**
 * @brief Classification of how a field appears in a weak-form expression
 *
 * Retained for backward compatibility with existing tests.
 * Internally populated from analysis::FieldOperatorSummary.
 */
struct FieldAppearanceClassification {
    FieldId field{INVALID_FIELD_ID};

    bool only_through_annihilating_ops{true};
    bool has_absolute_terms{false};
    bool has_stabilization{false};
    bool only_through_sym_grad{true};
    bool has_plain_grad{false};
    bool has_time_derivative{false};
    int occurrence_count{0};
    int field_value_dimension{0};
};

/**
 * @brief Analyze FormExpr residual for nullspace modes
 */
class NullspaceAnalyzer {
public:
    NullspaceAnalyzer() = default;

    /**
     * @brief Analyze a residual FormExpr for nullspace modes in the given fields
     *
     * Internally delegates DAG walking to FormStructureAnalyzer, then maps
     * the FieldOperatorSummary to GaugeCandidates via analyzeFromSummary().
     */
    [[nodiscard]] std::vector<gauge::GaugeCandidate>
    analyze(const FormExpr& residual, std::span<const FieldId> fields) const;

    /**
     * @brief Produce GaugeCandidates from a pre-computed FormStructureSummary
     *
     * This is the classification logic extracted from the old analyze() method.
     * Can be called independently when a FormStructureSummary is already available.
     */
    [[nodiscard]] std::vector<gauge::GaugeCandidate>
    analyzeFromSummary(const analysis::FormStructureSummary& summary) const;

    /**
     * @brief Classify how a single field appears in a FormExpr DAG
     *
     * Thin wrapper over FormStructureAnalyzer::analyzeField() that converts
     * FieldOperatorSummary → FieldAppearanceClassification for backward compatibility.
     */
    [[nodiscard]] FieldAppearanceClassification
    classifyField(const FormExprNode& root, FieldId target_field) const;
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_NULLSPACE_ANALYZER_H

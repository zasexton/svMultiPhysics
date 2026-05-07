/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_FORMEXPR_SCANNER_H
#define SVMP_FE_ANALYSIS_FORMEXPR_SCANNER_H

/**
 * @file FormExprScanner.h
 * @brief Lightweight DAG scanner for extracting structural properties from FormExpr
 *
 * Walks a FormExprNode tree once and collects boolean flags and symbol names.
 * Used by FormsInstaller to populate FormulationRecord fields.
 *
 * @see FormulationRecord for the output structure
 * @see FormStructureAnalyzer (Phase 3) for deeper per-field analysis
 */

#include "Forms/FormExpr.h"
#include "Analysis/ConstitutiveLawMetadata.h"
#include "Analysis/FormRuntimeMetadata.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

/**
 * @brief Result of scanning a FormExprNode tree for structural properties
 */
struct FormExprScanResult {
    bool has_time_derivative{false};
    bool has_cell_diameter{false};          ///< CellDiameter present (mesh-scale hint)
    bool has_jump{false};                   ///< Jump operator (DG)
    bool has_average{false};                ///< Average operator (DG)
    bool has_cell_integral{false};
    bool has_boundary_integral{false};
    bool has_interior_face_integral{false};
    bool has_interface_integral{false};

    /// Names of BoundaryIntegralSymbol nodes found
    std::vector<std::string> boundary_functional_names;

    /// Names of AuxiliaryStateSymbol nodes found
    std::vector<std::string> auxiliary_state_names;

    /// Names of AuxiliaryInputSymbol nodes found
    std::vector<std::string> auxiliary_input_names;

    /// Names of AuxiliaryOutputSymbol nodes found
    std::vector<std::string> auxiliary_output_names;

    /// Exact boundary markers found on BoundaryIntegral nodes
    std::vector<int> boundary_markers;

    /// Exact interface markers found on InterfaceIntegral nodes
    std::vector<int> interface_markers;

    /// Constitutive law metadata discovered from constitutive model nodes
    std::vector<ConstitutiveLawMetadata> constitutive_laws;

    /// Runtime parameter terminals referenced by this expression.
    std::vector<FormParameterUsage> parameter_usages;

    /// Callback coefficient terminals referenced by this expression.
    std::vector<FormCoefficientUsage> coefficient_usages;

    /// Recognized multiplicative h/dt/parameter/coefficient scale factors.
    std::vector<FormScaleUsage> scale_usages;

    /// Primitive-DAG admissibility domains inferred from exact algebraic structure.
    std::vector<ExpressionDomainConstraint> expression_domain_constraints;

    /// Scheme-level invariant-domain metadata from explicit producers only.
    std::vector<InvariantDomainDescriptor> invariant_domain_descriptors;

    /// Convenience
    [[nodiscard]] bool has_stabilization() const noexcept { return has_cell_diameter; }
    [[nodiscard]] bool has_interior_face_terms() const noexcept { return has_jump || has_average || has_interior_face_integral; }

    /// Build active_domains list from scan flags
    [[nodiscard]] std::vector<DomainKind> activeDomains() const;
};

/**
 * @brief Scan a FormExprNode tree for structural properties in a single pass
 *
 * @param root  Root of the FormExpr DAG
 * @return      Scan results (boolean flags + symbol names)
 */
[[nodiscard]] FormExprScanResult scanFormExpr(const forms::FormExprNode& root);

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_FORMEXPR_SCANNER_H

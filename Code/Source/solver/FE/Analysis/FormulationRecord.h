/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_FORMULATION_RECORD_H
#define SVMP_FE_ANALYSIS_FORMULATION_RECORD_H

/**
 * @file FormulationRecord.h
 * @brief Metadata snapshot of an installed variational formulation
 *
 * Populated by FormsInstaller when a formulation is installed.
 * Retained as a source artifact for FormContributionLowerer and as a
 * fallback input for analyzer passes when ContributionDescriptors are
 * not available. The primary analysis path consumes ContributionDescriptors
 * (lowered from this record by FormContributionLowerer in Phase 11).
 *
 * @see FormsInstaller::installFormulation() for the producer
 * @see FormContributionLowerer for the lowering to ContributionDescriptor
 * @see FormStructureAnalyzer for the fallback DAG analysis
 */

#include "Core/Types.h"
#include "Analysis/ConstitutiveLawMetadata.h"
#include "Analysis/FormRuntimeMetadata.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

namespace forms {
class FormExprNode;
} // namespace forms

namespace analysis {

struct AuxiliaryOutputConsumerRecord {
    std::uint32_t output_id{0u};
    std::string qualified_output_name{};
    std::string operator_tag{};
    DomainKind domain_kind{DomainKind::Cell};
    FieldId reference_field{INVALID_FIELD_ID};
    FieldId test_field{INVALID_FIELD_ID};
    FieldId trial_field{INVALID_FIELD_ID};
};

/**
 * @brief Structured metadata for an installed variational formulation
 *
 * Captures the structure of a weak-form residual as installed by
 * installFormulation().  This record preserves enough information for
 * downstream analysis passes without re-walking the FormExpr DAG.
 */
struct FormulationRecord {
    /// Operator tag (e.g. "equations", "NavierStokesVMS")
    std::string operator_tag;

    /// FE fields appearing in the residual
    std::vector<FieldId> active_fields;

    /// Generic variables (FE fields + coupled non-FE symbols)
    std::vector<VariableKey> active_variables;

    /// Shared pointer to the residual FormExprNode root.
    /// Kept alive so downstream analyzers can re-walk the DAG if needed.
    std::shared_ptr<const forms::FormExprNode> residual_expr;

    /// Whether AffineAnalysis decomposition succeeded (linear PDE)
    bool affine_split_succeeded{false};

    /// Whether test ≠ trial spaces exist (cross-coupling blocks)
    bool is_mixed{false};

    /// DG jump/average terms present
    bool has_interior_face_terms{false};

    /// Any field under TimeDerivative
    bool has_time_derivative{false};

    /// CellDiameter-scaled terms detected (SUPG/PSPG/GLS)
    bool has_stabilization_terms{false};

    /// Domains on which the formulation contributes integrals
    std::vector<DomainKind> active_domains;

    /// (test_field, trial_field) pairs for FE Jacobian blocks
    std::vector<std::pair<FieldId, FieldId>> block_couplings;

    /// Generic couplings including aux state / boundary functionals
    std::vector<std::pair<VariableKey, VariableKey>> variable_couplings;

    /// Boundary functional symbols referenced in the residual
    std::vector<VariableKey> boundary_functional_dependencies;

    /// Auxiliary state symbols referenced in the residual
    std::vector<VariableKey> auxiliary_state_dependencies;

    /// Generalized auxiliary input dependencies (from AuxiliaryInputSymbol terminals)
    std::vector<VariableKey> auxiliary_input_dependencies;

    /// Auxiliary output dependencies (from AuxiliaryOutputSymbol terminals)
    std::vector<VariableKey> auxiliary_output_dependencies;

    /// Constitutive laws intentionally used while authoring this residual.
    std::vector<ConstitutiveLawMetadata> constitutive_laws;

    /// Runtime parameters discovered in the residual DAG.
    std::vector<FormParameterUsage> parameter_usages;

    /// Callback coefficients discovered in the residual DAG.
    std::vector<FormCoefficientUsage> coefficient_usages;

    /// Recognized scale factors discovered in the residual DAG.
    std::vector<FormScaleUsage> scale_usages;

    /// Expression well-definedness constraints inferred from primitive DAG structure.
    std::vector<ExpressionDomainConstraint> expression_domain_constraints;

    /// Scheme-level invariant-domain metadata from explicit/theorem producers.
    std::vector<InvariantDomainDescriptor> invariant_domain_descriptors;

    /// Resolved auxiliary-output consumers keyed by stable output id.
    std::vector<AuxiliaryOutputConsumerRecord> auxiliary_output_consumers;

    /// Per-block residual FormExprNode handles.
    /// Key: (test_field, trial_field). For single-field formulations, contains
    /// one entry {(field, field)} → residual root. For multi-field, populated
    /// when per-block expressions are available from the form compiler.
    std::vector<std::pair<std::pair<FieldId, FieldId>,
                          std::shared_ptr<const forms::FormExprNode>>> block_residual_exprs;

    // ---- Mixed-form provenance (Phase 4) ----

    /// Field names from the mixed expression, keyed by FieldId.
    /// Populated by installFormulation() for multi-field formulations.
    /// Used by diagnostics to produce messages like "block (velocity, pressure)"
    /// instead of "block (FieldId=0, FieldId=1)".
    std::vector<std::pair<FieldId, std::string>> field_names;

    /// Test function names in expression order (e.g., {"v", "q"})
    std::vector<std::string> test_function_names;

    /// Trial function names in expression order (e.g., {"u", "p"})
    std::vector<std::string> trial_function_names;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_FORMULATION_RECORD_H

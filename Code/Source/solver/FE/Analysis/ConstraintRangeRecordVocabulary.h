/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */

#ifndef SVMP_FE_ANALYSIS_CONSTRAINT_RANGE_RECORD_VOCABULARY_H
#define SVMP_FE_ANALYSIS_CONSTRAINT_RANGE_RECORD_VOCABULARY_H

#include <array>
#include <string_view>

namespace svmp::FE::analysis {

/// Labels and optional qualification text for a constraint-range observation.
/// The solver supplies the measured values, cache decisions and gate results.
/// A formulation may bind its own static record schema to the method.
struct ConstraintRangeRecordVocabulary {
    std::string_view sample_heading{"NewtonSolver: constraint range diagnostic=constraint_range"};
    std::string_view range_gate_heading{"NewtonSolver: constraint-range distance gate diagnostic=constraint_range_distance_gate"};
    std::string_view constant_gate_heading{"NewtonSolver: constant-field KKT distance gate diagnostic=constant_field_kkt_distance_gate"};
    std::string_view initializer_heading{"NewtonSolver: compatible constraint-field initializer diagnostic=constraint_field_initializer"};
    const char* assembly_source{"constraint_range"};
    const char* initializer_phase{"constraint_field_initializer"};
    std::string_view range{"constraint_range"};
    std::string_view constant{"constant_field"};
    std::string_view field{"constraint_field"};
    std::string_view multiplier{"constraint_multiplier"};
    std::string_view normalization{"constraint_plus_potential_norms"};
    std::string_view constant_fit{"closed_form_one_dimensional_constant_trace"};
    std::string_view load{"declared_constraint_independent_load"};
    std::string_view scope{" scope=declared_first_variations"};
    std::string_view exclusions{""};
    std::string_view sample_claims{""};
    std::string_view initializer_scope{" force_projection_applied=0 production_operator_changed=0"};
    std::array<std::string_view, 6> work_norms{
        "constraint_work_norm", "first_potential_work_norm", "second_potential_work_norm",
        "potential_work_norm", "load_norm", "balance_norm"};
    std::string_view (*reason_name)(std::string_view) noexcept{nullptr};

    [[nodiscard]] std::string_view reason(std::string_view numerical_reason) const noexcept
    {
        return reason_name ? reason_name(numerical_reason) : numerical_reason;
    }
};

} // namespace svmp::FE::analysis

#endif // SVMP_FE_ANALYSIS_CONSTRAINT_RANGE_RECORD_VOCABULARY_H

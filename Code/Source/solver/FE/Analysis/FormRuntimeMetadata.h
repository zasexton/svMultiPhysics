/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_FORM_RUNTIME_METADATA_H
#define SVMP_FE_ANALYSIS_FORM_RUNTIME_METADATA_H

/**
 * @file FormRuntimeMetadata.h
 * @brief Physics-agnostic DAG metadata used to produce runtime analysis summaries.
 */

#include "Analysis/ProblemAnalysisTypes.h"
#include "Core/Types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

enum class FormCoefficientRank : std::uint8_t {
    Scalar,
    Vector,
    Rank2Tensor,
    Rank3Tensor,
    Rank4Tensor,
    Unknown,
};

enum class ExpressionDomainConstraintKind : std::uint8_t {
    NonzeroDenominator,
    NonnegativeRadicand,
    PositiveJacobianLikeDeterminant,
    InvertibleMatrixExpression,
    BoundedLogArgument,
    BoundedPowerBase,
    UserDeclaredAdmissibilityCondition,
};

enum class InvariantSetKind : std::uint8_t {
    Unknown,
    Interval,
    PositivityCone,
    ConvexSet,
    UserDeclared,
};

struct FormParameterUsage {
    std::string name;
    std::optional<std::uint32_t> slot;
    DomainKind domain{DomainKind::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
    bool real_valued{true};
};

struct FormCoefficientUsage {
    std::string name;
    FormCoefficientRank rank{FormCoefficientRank::Unknown};
    bool time_dependent{false};
    DomainKind domain{DomainKind::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
};

struct FormScaleUsage {
    std::string scale_id;
    int h_power{0};
    int dt_power{0};
    std::vector<std::string> parameter_names;
    std::vector<std::uint32_t> parameter_slots;
    std::vector<std::string> coefficient_names;
    DomainKind domain{DomainKind::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
    bool exact_for_analysis{false};
};

struct ExpressionDomainConstraint {
    ExpressionDomainConstraintKind kind{
        ExpressionDomainConstraintKind::UserDeclaredAdmissibilityCondition};
    std::vector<VariableKey> variables;
    std::optional<Real> lower_bound;
    std::optional<Real> upper_bound;
    bool has_excluded_value{false};
    Real excluded_value{};
    std::string expression_id;
    DomainKind domain{DomainKind::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
    EvidenceLevel evidence_level{EvidenceLevel::DAGPatternHint};
};

struct InvariantDomainDescriptor {
    InvariantSetKind invariant_set{InvariantSetKind::Unknown};
    std::string invariant_set_id;
    std::vector<VariableKey> variables;
    DomainKind domain{DomainKind::Cell};
    std::optional<Real> lower_bound;
    std::optional<Real> upper_bound;
    std::optional<Real> excluded_value;
    std::optional<Real> cfl_estimate;
    std::optional<Real> accepted_cfl_bound;
    std::optional<Real> wave_speed_bound;
    std::string time_step_scope;
    std::string mesh_size_scope;
    std::string theorem_id;
    bool limiter_evidence_present{false};
    bool cfl_condition_satisfied{false};
    bool ssp_time_discretization_evidence_present{false};
    bool source_admissibility_evidence_present{false};
    bool low_order_invariant_domain_evidence_present{false};
    bool convex_limiting_evidence_present{false};
    bool spatial_monotonicity_evidence_present{false};
    bool mass_positivity_evidence_present{false};
    FieldId sampled_field{INVALID_FIELD_ID};
    int sampled_component{-1};
    Real bound_tolerance{static_cast<Real>(1.0e-12)};
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_FORM_RUNTIME_METADATA_H

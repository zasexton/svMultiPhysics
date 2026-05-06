/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/AnalysisSummaryTypes.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace svmp {
namespace FE {
namespace analysis {

// ============================================================================
// String conversion
// ============================================================================

const char* toString(PropertyKind k) noexcept {
    switch (k) {
        case PropertyKind::Nullspace:              return "Nullspace";
        case PropertyKind::OverConstraint:         return "OverConstraint";
        case PropertyKind::UnderConstraint:        return "UnderConstraint";
        case PropertyKind::MixedSaddlePoint:       return "MixedSaddlePoint";
        case PropertyKind::CompatibilityCondition: return "CompatibilityCondition";
        case PropertyKind::OperatorSymmetry:       return "OperatorSymmetry";
        case PropertyKind::OperatorDefiniteness:   return "OperatorDefiniteness";
        case PropertyKind::Stabilization:          return "Stabilization";
        case PropertyKind::TopologyScopedKernel:   return "TopologyScopedKernel";
        case PropertyKind::ConstraintRedundancy:   return "ConstraintRedundancy";
        case PropertyKind::CoupledSystemStructure: return "CoupledSystemStructure";
        case PropertyKind::InterfaceCondition:     return "InterfaceCondition";
        case PropertyKind::InfSupCondition:        return "InfSupCondition";
        case PropertyKind::ConservationStructure:  return "ConservationStructure";
        case PropertyKind::DifferentialAlgebraicStructure: return "DifferentialAlgebraicStructure";
        case PropertyKind::SpaceCompatibility:     return "SpaceCompatibility";
        case PropertyKind::OperatorTransportCharacter: return "OperatorTransportCharacter";
        case PropertyKind::DiscreteMaximumPrinciple: return "DiscreteMaximumPrinciple";
        case PropertyKind::ZMatrixStructure:       return "ZMatrixStructure";
        case PropertyKind::MMatrixStructure:       return "MMatrixStructure";
        case PropertyKind::MatrixMonotonicityRisk: return "MatrixMonotonicityRisk";
        case PropertyKind::CompatibleComplexStructure: return "CompatibleComplexStructure";
        case PropertyKind::EnergyStability:        return "EnergyStability";
        case PropertyKind::EntropyStability:       return "EntropyStability";
        case PropertyKind::TemporalStability:      return "TemporalStability";
        case PropertyKind::WeakBoundaryCoercivity: return "WeakBoundaryCoercivity";
        case PropertyKind::MeshGeometryValidity:   return "MeshGeometryValidity";
        case PropertyKind::CoefficientPositivity:  return "CoefficientPositivity";
        case PropertyKind::NonlinearTangentStructure: return "NonlinearTangentStructure";
        case PropertyKind::LockingRisk:            return "LockingRisk";
        case PropertyKind::SpectralCorrectness:    return "SpectralCorrectness";
        case PropertyKind::ErrorEstimatorEligibility: return "ErrorEstimatorEligibility";
        case PropertyKind::SolverCompatibility:    return "SolverCompatibility";
        case PropertyKind::QuadratureAdequacy:     return "QuadratureAdequacy";
        case PropertyKind::BoundaryComplementingCondition: return "BoundaryComplementingCondition";
        case PropertyKind::IndefiniteOperatorResolution: return "IndefiniteOperatorResolution";
        case PropertyKind::MinimumResidualStability: return "MinimumResidualStability";
        case PropertyKind::InvariantDomainPreservation: return "InvariantDomainPreservation";
        case PropertyKind::EquilibriumPreservation: return "EquilibriumPreservation";
        case PropertyKind::GeometricConservation:  return "GeometricConservation";
        case PropertyKind::TransferOperatorCompatibility: return "TransferOperatorCompatibility";
        case PropertyKind::AdjointConsistency:     return "AdjointConsistency";
        case PropertyKind::ParameterRobustness:    return "ParameterRobustness";
        case PropertyKind::InitialDataCompatibility: return "InitialDataCompatibility";
    }
    return "Unknown";
}

const char* toString(PropertyStatus s) noexcept {
    switch (s) {
        case PropertyStatus::Exact:     return "Exact";
        case PropertyStatus::Likely:    return "Likely";
        case PropertyStatus::Violated:  return "Violated";
        case PropertyStatus::Preserved: return "Preserved";
        case PropertyStatus::Unknown:   return "Unknown";
    }
    return "Unknown";
}

const char* toString(AnalysisConfidence c) noexcept {
    switch (c) {
        case AnalysisConfidence::High:   return "High";
        case AnalysisConfidence::Medium: return "Medium";
        case AnalysisConfidence::Low:    return "Low";
    }
    return "Unknown";
}

const char* toString(IssueSeverity s) noexcept {
    switch (s) {
        case IssueSeverity::Error:   return "ERROR";
        case IssueSeverity::Warning: return "WARNING";
        case IssueSeverity::Info:    return "INFO";
    }
    return "UNKNOWN";
}

const char* toString(VariableKind k) noexcept {
    switch (k) {
        case VariableKind::FieldComponent:     return "FieldComponent";
        case VariableKind::AuxiliaryState:     return "AuxiliaryState";
        case VariableKind::AuxiliaryInput:     return "AuxiliaryInput";
        case VariableKind::AuxiliaryOutput:    return "AuxiliaryOutput";
        case VariableKind::BoundaryFunctional: return "BoundaryFunctional";
        case VariableKind::GlobalScalar:       return "GlobalScalar";
    }
    return "Unknown";
}

const char* toString(DomainKind k) noexcept {
    switch (k) {
        case DomainKind::Cell:            return "Cell";
        case DomainKind::Boundary:        return "Boundary";
        case DomainKind::InteriorFace:    return "InteriorFace";
        case DomainKind::InterfaceFace:   return "InterfaceFace";
        case DomainKind::Global:          return "Global";
        case DomainKind::CoupledBoundary: return "CoupledBoundary";
        case DomainKind::AuxiliaryCoupling: return "AuxiliaryCoupling";
    }
    return "Unknown";
}

// Phase 21 toString helpers

const char* toString(InfSupClass c) noexcept {
    switch (c) {
        case InfSupClass::Required:              return "Required";
        case InfSupClass::StructurallySupported: return "StructurallySupported";
        case InfSupClass::NumericallySupported:  return "NumericallySupported";
        case InfSupClass::StabilizedSurrogate:   return "StabilizedSurrogate";
        case InfSupClass::LikelyViolated:        return "LikelyViolated";
        case InfSupClass::Unknown:               return "Unknown";
    }
    return "Unknown";
}

const char* toString(ConservationClass c) noexcept {
    switch (c) {
        case ConservationClass::LocalClosureExpected:  return "LocalClosureExpected";
        case ConservationClass::GlobalClosureExpected:  return "GlobalClosureExpected";
        case ConservationClass::ExchangeBalanced:       return "ExchangeBalanced";
        case ConservationClass::ClosureBroken:          return "ClosureBroken";
        case ConservationClass::Unknown:                return "Unknown";
    }
    return "Unknown";
}

const char* toString(DAEClass c) noexcept {
    switch (c) {
        case DAEClass::PureODELike:      return "PureODELike";
        case DAEClass::AlgebraicSystem:  return "AlgebraicSystem";
        case DAEClass::Index1DAELike:    return "Index1DAELike";
        case DAEClass::HigherIndexRisk:  return "HigherIndexRisk";
        case DAEClass::Unknown:          return "Unknown";
    }
    return "Unknown";
}

const char* toString(SpaceCompatibilityClass c) noexcept {
    switch (c) {
        case SpaceCompatibilityClass::Compatible:       return "Compatible";
        case SpaceCompatibilityClass::WeaklyCompatible: return "WeaklyCompatible";
        case SpaceCompatibilityClass::Incompatible:     return "Incompatible";
        case SpaceCompatibilityClass::Unknown:          return "Unknown";
    }
    return "Unknown";
}

const char* toString(TransportCharacterClass c) noexcept {
    switch (c) {
        case TransportCharacterClass::None:                    return "None";
        case TransportCharacterClass::DiffusionLike:           return "DiffusionLike";
        case TransportCharacterClass::DirectionalFirstOrderLike: return "DirectionalFirstOrderLike";
        case TransportCharacterClass::NonNormalRisk:            return "NonNormalRisk";
        case TransportCharacterClass::TransportDominatedRisk:   return "TransportDominatedRisk";
        case TransportCharacterClass::Unknown:                  return "Unknown";
    }
    return "Unknown";
}

const char* toString(ApplicabilityClass c) noexcept {
    switch (c) {
        case ApplicabilityClass::Applicable:    return "Applicable";
        case ApplicabilityClass::NotApplicable: return "NotApplicable";
        case ApplicabilityClass::Unknown:       return "Unknown";
    }
    return "Unknown";
}

const char* toString(CertificationClass c) noexcept {
    switch (c) {
        case CertificationClass::Certified:    return "Certified";
        case CertificationClass::Violated:     return "Violated";
        case CertificationClass::NotCertified: return "NotCertified";
        case CertificationClass::Unknown:      return "Unknown";
    }
    return "Unknown";
}

const char* toString(MatrixSignStructureClass c) noexcept {
    switch (c) {
        case MatrixSignStructureClass::ZMatrix:             return "ZMatrix";
        case MatrixSignStructureClass::NotZMatrix:          return "NotZMatrix";
        case MatrixSignStructureClass::MMatrixCertified:    return "MMatrixCertified";
        case MatrixSignStructureClass::MMatrixNotCertified: return "MMatrixNotCertified";
        case MatrixSignStructureClass::Unknown:             return "Unknown";
    }
    return "Unknown";
}

const char* toString(OperatorSymmetryClass c) noexcept {
    switch (c) {
        case OperatorSymmetryClass::Symmetric:    return "Symmetric";
        case OperatorSymmetryClass::Skew:         return "Skew";
        case OperatorSymmetryClass::Nonsymmetric: return "Nonsymmetric";
        case OperatorSymmetryClass::Unknown:      return "Unknown";
    }
    return "Unknown";
}

const char* toString(TemporalStabilityClass c) noexcept {
    switch (c) {
        case TemporalStabilityClass::AStable:              return "AStable";
        case TemporalStabilityClass::LStable:              return "LStable";
        case TemporalStabilityClass::BStable:              return "BStable";
        case TemporalStabilityClass::SSP:                  return "SSP";
        case TemporalStabilityClass::ConditionallyStable:  return "ConditionallyStable";
        case TemporalStabilityClass::Unknown:              return "Unknown";
    }
    return "Unknown";
}

const char* toString(CoercivityClass c) noexcept {
    switch (c) {
        case CoercivityClass::Coercive:     return "Coercive";
        case CoercivityClass::Semicoercive: return "Semicoercive";
        case CoercivityClass::Indefinite:   return "Indefinite";
        case CoercivityClass::NotCoercive:  return "NotCoercive";
        case CoercivityClass::Unknown:      return "Unknown";
    }
    return "Unknown";
}

const char* toString(NullspaceHandlingClass c) noexcept {
    switch (c) {
        case NullspaceHandlingClass::NotApplicable:         return "NotApplicable";
        case NullspaceHandlingClass::AnchoredByConstraints: return "AnchoredByConstraints";
        case NullspaceHandlingClass::ProjectedOut:          return "ProjectedOut";
        case NullspaceHandlingClass::Retained:              return "Retained";
        case NullspaceHandlingClass::Uncontrolled:          return "Uncontrolled";
        case NullspaceHandlingClass::Unknown:               return "Unknown";
    }
    return "Unknown";
}

const char* toString(TemporalStateKind k) noexcept {
    switch (k) {
        case TemporalStateKind::Algebraic: return "Algebraic";
        case TemporalStateKind::Dynamic:   return "Dynamic";
        case TemporalStateKind::Mixed:     return "Mixed";
        case TemporalStateKind::Unknown:   return "Unknown";
    }
    return "Unknown";
}

const char* toString(SpaceFamily f) noexcept {
    switch (f) {
        case SpaceFamily::H1:      return "H1";
        case SpaceFamily::HDiv:    return "HDiv";
        case SpaceFamily::HCurl:   return "HCurl";
        case SpaceFamily::L2:      return "L2";
        case SpaceFamily::Custom:  return "Custom";
        case SpaceFamily::Unknown: return "Unknown";
    }
    return "Unknown";
}

// ============================================================================
// AnalysisRequestPlan
// ============================================================================

bool AnalysisRequestPlan::has(AnalysisSummaryKind kind) const noexcept {
    return std::any_of(summary_requests.begin(), summary_requests.end(),
                       [kind](const AnalysisSummaryRequest& request) {
                           return request.summary_kind == kind;
                       });
}

std::vector<const AnalysisSummaryRequest*>
AnalysisRequestPlan::requestsOfKind(AnalysisSummaryKind kind) const {
    std::vector<const AnalysisSummaryRequest*> result;
    for (const auto& request : summary_requests) {
        if (request.summary_kind == kind) {
            result.push_back(&request);
        }
    }
    return result;
}

bool AnalysisRequestPlan::hasSourceAnalyzer(const std::string& analyzer) const noexcept {
    for (const auto& request : summary_requests) {
        if (std::find(request.source_analyzers.begin(),
                      request.source_analyzers.end(),
                      analyzer) != request.source_analyzers.end()) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// ProblemAnalysisReport — queries
// ============================================================================

std::size_t ProblemAnalysisReport::countByKind(PropertyKind kind) const noexcept {
    return static_cast<std::size_t>(
        std::count_if(claims.begin(), claims.end(),
                      [kind](const PropertyClaim& c) { return c.kind == kind; }));
}

std::size_t ProblemAnalysisReport::countByStatus(PropertyStatus status) const noexcept {
    return static_cast<std::size_t>(
        std::count_if(claims.begin(), claims.end(),
                      [status](const PropertyClaim& c) { return c.status == status; }));
}

std::size_t ProblemAnalysisReport::countBySeverity(IssueSeverity severity) const noexcept {
    return static_cast<std::size_t>(
        std::count_if(issues.begin(), issues.end(),
                      [severity](const AnalysisIssue& i) { return i.severity == severity; }));
}

std::vector<const PropertyClaim*> ProblemAnalysisReport::claimsForField(FieldId field) const {
    std::vector<const PropertyClaim*> result;
    for (const auto& c : claims) {
        if (c.field == field) {
            result.push_back(&c);
        }
    }
    return result;
}

std::vector<const PropertyClaim*> ProblemAnalysisReport::claimsForVariable(const VariableKey& var) const {
    std::vector<const PropertyClaim*> result;
    for (const auto& c : claims) {
        // Check the variables list first
        for (const auto& v : c.variables) {
            if (v == var) {
                result.push_back(&c);
                break;
            }
        }
        // Also match via the legacy field/component shorthand
        if (result.empty() || result.back() != &c) {
            if (var.kind == VariableKind::FieldComponent
                && c.field == var.field_id
                && (var.component == -1 || c.component == -1 || c.component == var.component)) {
                result.push_back(&c);
            }
        }
    }
    return result;
}

std::vector<const PropertyClaim*> ProblemAnalysisReport::claimsOfKind(PropertyKind kind) const {
    std::vector<const PropertyClaim*> result;
    for (const auto& c : claims) {
        if (c.kind == kind) {
            result.push_back(&c);
        }
    }
    return result;
}

bool ProblemAnalysisReport::hasErrors() const noexcept {
    return std::any_of(issues.begin(), issues.end(),
                       [](const AnalysisIssue& i) { return i.severity == IssueSeverity::Error; });
}

bool ProblemAnalysisReport::hasWarnings() const noexcept {
    return std::any_of(issues.begin(), issues.end(),
                       [](const AnalysisIssue& i) { return i.severity == IssueSeverity::Warning; });
}

// ============================================================================
// ProblemAnalysisReport — output
// ============================================================================

namespace {

void appendStructuredField(std::ostream& out,
                           bool& any,
                           const char* name,
                           const char* value)
{
    out << (any ? ", " : " {");
    out << name << "=" << value;
    any = true;
}

void appendStructuredField(std::ostream& out,
                           bool& any,
                           const char* name,
                           double value)
{
    out << (any ? ", " : " {");
    out << name << "=" << value;
    any = true;
}

void appendStructuredField(std::ostream& out,
                           bool& any,
                           const char* name,
                           bool value)
{
    appendStructuredField(out, any, name, value ? "true" : "false");
}

void appendStructuredField(std::ostream& out,
                           bool& any,
                           const char* name,
                           const std::string& value)
{
    appendStructuredField(out, any, name, value.c_str());
}

void appendStructuredFields(std::ostream& out, const PropertyClaim& claim)
{
    bool any = false;

    if (claim.inf_sup_class) appendStructuredField(out, any, "inf_sup", toString(*claim.inf_sup_class));
    if (claim.conservation_class) appendStructuredField(out, any, "conservation", toString(*claim.conservation_class));
    if (claim.dae_class) appendStructuredField(out, any, "dae", toString(*claim.dae_class));
    if (claim.space_compatibility_class) appendStructuredField(out, any, "space", toString(*claim.space_compatibility_class));
    if (claim.transport_character_class) appendStructuredField(out, any, "transport", toString(*claim.transport_character_class));
    if (claim.applicability_class) appendStructuredField(out, any, "applicability", toString(*claim.applicability_class));
    if (claim.certification_class) appendStructuredField(out, any, "certification", toString(*claim.certification_class));
    if (claim.matrix_sign_structure_class) appendStructuredField(out, any, "matrix_sign", toString(*claim.matrix_sign_structure_class));
    if (claim.operator_symmetry_class) appendStructuredField(out, any, "symmetry", toString(*claim.operator_symmetry_class));
    if (claim.temporal_stability_class) appendStructuredField(out, any, "temporal", toString(*claim.temporal_stability_class));
    if (claim.coercivity_class) appendStructuredField(out, any, "coercivity", toString(*claim.coercivity_class));
    if (claim.reduced_definiteness_class) appendStructuredField(out, any, "reduced_definiteness", toString(*claim.reduced_definiteness_class));
    if (claim.nullspace_handling_class) appendStructuredField(out, any, "nullspace_handling", toString(*claim.nullspace_handling_class));

    if (claim.inf_sup_estimate) appendStructuredField(out, any, "inf_sup_estimate", *claim.inf_sup_estimate);
    if (claim.peclet_number) appendStructuredField(out, any, "peclet", *claim.peclet_number);
    if (claim.cfl_number) appendStructuredField(out, any, "cfl", *claim.cfl_number);
    if (claim.nonnormality_indicator) appendStructuredField(out, any, "nonnormality", *claim.nonnormality_indicator);
    if (claim.local_balance_residual) appendStructuredField(out, any, "local_balance_residual", *claim.local_balance_residual);
    if (claim.global_balance_residual) appendStructuredField(out, any, "global_balance_residual", *claim.global_balance_residual);
    if (claim.interface_balance_residual) appendStructuredField(out, any, "interface_balance_residual", *claim.interface_balance_residual);
    if (claim.constraint_drift_norm) appendStructuredField(out, any, "constraint_drift", *claim.constraint_drift_norm);
    if (claim.penalty_scale) appendStructuredField(out, any, "penalty_scale", *claim.penalty_scale);
    if (claim.weak_coercivity_lower_bound) appendStructuredField(out, any, "weak_coercivity_lower_bound", *claim.weak_coercivity_lower_bound);
    if (claim.flux_balance_residual) appendStructuredField(out, any, "flux_balance_residual", *claim.flux_balance_residual);

    if (claim.exact_sequence_compatible) appendStructuredField(out, any, "exact_sequence_compatible", *claim.exact_sequence_compatible);
    if (claim.commuting_projection_available) appendStructuredField(out, any, "commuting_projection_available", *claim.commuting_projection_available);
    if (claim.boundary_complementing_condition_satisfied) appendStructuredField(out, any, "boundary_complementing_satisfied", *claim.boundary_complementing_condition_satisfied);
    if (claim.initial_data_compatible) appendStructuredField(out, any, "initial_data_compatible", *claim.initial_data_compatible);
    if (claim.invariant_domain_metadata_present) appendStructuredField(out, any, "invariant_domain_metadata", *claim.invariant_domain_metadata_present);
    if (claim.well_balanced_metadata_present) appendStructuredField(out, any, "well_balanced_metadata", *claim.well_balanced_metadata_present);

    if (claim.tested_block_id) appendStructuredField(out, any, "tested_block", *claim.tested_block_id);
    if (claim.estimate_scope) appendStructuredField(out, any, "estimate_scope", *claim.estimate_scope);
    if (claim.coefficient_id) appendStructuredField(out, any, "coefficient", *claim.coefficient_id);
    if (claim.equilibrium_id) appendStructuredField(out, any, "equilibrium", *claim.equilibrium_id);
    if (claim.invariant_set_id) appendStructuredField(out, any, "invariant_set", *claim.invariant_set_id);

    if (any) {
        out << "}";
    }
}

std::string lowerAscii(const char* text)
{
    std::string value = text != nullptr ? text : "";
    for (char& ch : value) {
        ch = static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

void appendUniqueString(std::vector<std::string>& values, std::string value)
{
    if (value.empty()) {
        return;
    }
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(std::move(value));
    }
}

std::string joinStrings(const std::vector<std::string>& values,
                        const char* separator = ", ")
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0u) {
            oss << separator;
        }
        oss << values[i];
    }
    return oss.str();
}

std::string variableLabel(const VariableKey& variable)
{
    std::ostringstream oss;
    if (variable.kind == VariableKind::FieldComponent) {
        oss << "field=" << variable.field_id;
        if (variable.component >= 0) {
            oss << ":component=" << variable.component;
        }
        return oss.str();
    }

    oss << toString(variable.kind);
    if (!variable.name.empty()) {
        oss << "=" << variable.name;
    }
    return oss.str();
}

std::string variablesLabel(const std::vector<VariableKey>& variables)
{
    std::vector<std::string> labels;
    labels.reserve(variables.size());
    for (const auto& variable : variables) {
        labels.push_back(variableLabel(variable));
    }
    return joinStrings(labels, "+");
}

std::string claimOrigin(const PropertyClaim& claim)
{
    if (!claim.claim_origin.empty()) {
        return claim.claim_origin;
    }
    for (const auto& evidence : claim.evidence) {
        if (!evidence.source.empty()) {
            return evidence.source;
        }
    }
    return std::string(toString(claim.kind));
}

void appendClaimScope(std::ostream& out, const PropertyClaim& claim)
{
    if (claim.field != INVALID_FIELD_ID) {
        out << " field=" << claim.field;
    }
    if (claim.component >= 0) {
        out << " component=" << claim.component;
    }
    if (!claim.variables.empty()) {
        out << " variables=" << variablesLabel(claim.variables);
    }
    if (claim.region >= 0) {
        out << " region=" << claim.region;
    }
    out << " domain=" << toString(claim.domain);
}

void appendClaimScalars(std::ostream& out, const PropertyClaim& claim)
{
    if (claim.applicability_class) {
        out << " applicability=" << toString(*claim.applicability_class);
    }
    if (claim.certification_class) {
        out << " certification=" << toString(*claim.certification_class);
    }
    if (claim.matrix_sign_structure_class) {
        out << " matrix_sign=" << toString(*claim.matrix_sign_structure_class);
    }
    if (claim.inf_sup_estimate) {
        out << " inf_sup=" << *claim.inf_sup_estimate;
    }
    if (claim.peclet_number) {
        out << " peclet=" << *claim.peclet_number;
    }
    if (claim.cfl_number) {
        out << " cfl=" << *claim.cfl_number;
    }
    if (claim.nonnormality_indicator) {
        out << " nonnormality=" << *claim.nonnormality_indicator;
    }
    if (claim.local_balance_residual) {
        out << " local_balance=" << *claim.local_balance_residual;
    }
    if (claim.global_balance_residual) {
        out << " global_balance=" << *claim.global_balance_residual;
    }
    if (claim.interface_balance_residual) {
        out << " interface_balance=" << *claim.interface_balance_residual;
    }
    if (claim.constraint_drift_norm) {
        out << " constraint_drift=" << *claim.constraint_drift_norm;
    }
    if (claim.penalty_scale) {
        out << " penalty_scale=" << *claim.penalty_scale;
    }
    if (claim.weak_coercivity_lower_bound) {
        out << " weak_coercivity_lower_bound="
            << *claim.weak_coercivity_lower_bound;
    }
    if (claim.flux_balance_residual) {
        out << " flux_balance=" << *claim.flux_balance_residual;
    }
    if (claim.tested_block_id) {
        out << " block=" << *claim.tested_block_id;
    }
    if (claim.estimate_scope) {
        out << " scope=" << *claim.estimate_scope;
    }
    if (claim.invariant_set_id) {
        out << " invariant_set=" << *claim.invariant_set_id;
    }
    if (claim.equilibrium_id) {
        out << " equilibrium=" << *claim.equilibrium_id;
    }
}

std::string requestLabel(const AnalysisSummaryRequest& request)
{
    std::ostringstream oss;
    oss << toString(request.summary_kind) << "(domain=" << toString(request.domain);
    if (!request.request_id.empty()) {
        oss << ",id=" << request.request_id;
    }
    if (!request.variables.empty()) {
        oss << ",variables=" << variablesLabel(request.variables);
    }
    if (!request.block_id.empty()) {
        oss << ",block=" << request.block_id;
    }
    if (!request.contribution_id.empty()) {
        oss << ",contribution=" << request.contribution_id;
    }
    if (!request.scope_id.empty()) {
        oss << ",scope=" << request.scope_id;
    }
    if (request.already_available) {
        oss << ",available=true";
    }
    oss << ")";
    return oss.str();
}

std::string blockLabel(const OperatorBlockId& block)
{
    std::ostringstream oss;
    if (!block.operator_tag.empty()) {
        oss << block.operator_tag;
    } else {
        oss << toString(block.domain);
    }
    if (!block.test_variables.empty()) {
        oss << ":test=" << variablesLabel(block.test_variables);
    }
    if (!block.trial_variables.empty()) {
        oss << ":trial=" << variablesLabel(block.trial_variables);
    }
    if (block.marker >= 0) {
        oss << ":marker=" << block.marker;
    }
    return oss.str();
}

void traceMatrixEntry(std::ostream& out,
                      const char* prefix,
                      const MatrixEntrySample& sample)
{
    out << "[FE/Analysis][trace] " << prefix
        << " worst_entry row=" << sample.row
        << " col=" << sample.col
        << " value=" << std::setprecision(17) << sample.value
        << " rank=" << sample.owning_rank;
    if (!sample.note.empty()) {
        out << " note=" << sample.note;
    }
    out << "\n";
}

void traceMatrixSummary(std::ostream& out,
                        const char* prefix,
                        const DiscreteMatrixSummary& summary)
{
    out << "[FE/Analysis][trace] " << prefix
        << " row_summary block=" << blockLabel(summary.block)
        << " rows=" << summary.rows
        << " cols=" << summary.cols
        << " positive_offdiag=" << summary.positive_offdiag_count
        << " row_sum_violations=" << summary.row_sum_violation_count
        << " nonfinite_entries=" << summary.nonfinite_entry_count
        << " nonfinite_row_sums=" << summary.nonfinite_row_sum_count
        << " min_row_sum=" << std::setprecision(17) << summary.min_row_sum
        << " max_row_sum=" << summary.max_row_sum
        << " sign_tolerance=" << summary.sign_tolerance
        << "\n";
    for (const auto& sample : summary.worst_entries) {
        traceMatrixEntry(out, prefix, sample);
    }
}

} // namespace

void ProblemAnalysisReport::print(std::ostream& out) const {
    out << "========================================================================\n";
    out << "  Problem Analysis Report\n";
    out << "========================================================================\n";
    out << "\n";
    out << "  " << summary() << "\n\n";

    if (claims.empty()) {
        out << "  No property claims.\n\n";
    } else {
        // Group by PropertyKind
        static constexpr PropertyKind all_kinds[] = {
            PropertyKind::Nullspace,
            PropertyKind::UnderConstraint,
            PropertyKind::OverConstraint,
            PropertyKind::MixedSaddlePoint,
            PropertyKind::CompatibilityCondition,
            PropertyKind::OperatorSymmetry,
            PropertyKind::OperatorDefiniteness,
            PropertyKind::Stabilization,
            PropertyKind::TopologyScopedKernel,
            PropertyKind::ConstraintRedundancy,
            PropertyKind::CoupledSystemStructure,
            PropertyKind::InterfaceCondition,
            PropertyKind::InfSupCondition,
            PropertyKind::ConservationStructure,
            PropertyKind::DifferentialAlgebraicStructure,
            PropertyKind::SpaceCompatibility,
            PropertyKind::OperatorTransportCharacter,
            PropertyKind::DiscreteMaximumPrinciple,
            PropertyKind::ZMatrixStructure,
            PropertyKind::MMatrixStructure,
            PropertyKind::MatrixMonotonicityRisk,
            PropertyKind::CompatibleComplexStructure,
            PropertyKind::EnergyStability,
            PropertyKind::EntropyStability,
            PropertyKind::TemporalStability,
            PropertyKind::WeakBoundaryCoercivity,
            PropertyKind::MeshGeometryValidity,
            PropertyKind::CoefficientPositivity,
            PropertyKind::NonlinearTangentStructure,
            PropertyKind::LockingRisk,
            PropertyKind::SpectralCorrectness,
            PropertyKind::ErrorEstimatorEligibility,
            PropertyKind::SolverCompatibility,
            PropertyKind::QuadratureAdequacy,
            PropertyKind::BoundaryComplementingCondition,
            PropertyKind::IndefiniteOperatorResolution,
            PropertyKind::MinimumResidualStability,
            PropertyKind::InvariantDomainPreservation,
            PropertyKind::EquilibriumPreservation,
            PropertyKind::GeometricConservation,
            PropertyKind::TransferOperatorCompatibility,
            PropertyKind::AdjointConsistency,
            PropertyKind::ParameterRobustness,
            PropertyKind::InitialDataCompatibility,
        };

        for (auto kind : all_kinds) {
            auto kind_claims = claimsOfKind(kind);
            if (kind_claims.empty()) continue;

            out << "  --- " << toString(kind) << " ---\n";
            for (const auto* cp : kind_claims) {
                out << "    [" << toString(cp->status) << "/" << toString(cp->confidence) << "]";
                if (cp->field != INVALID_FIELD_ID) {
                    out << " field=" << cp->field;
                }
                if (cp->component >= 0) {
                    out << " comp=" << cp->component;
                }
                if (cp->region >= 0) {
                    out << " region=" << cp->region;
                }
                if (!cp->description.empty()) {
                    out << "  " << cp->description;
                }
                appendStructuredFields(out, *cp);
                out << "  (" << cp->evidence.size() << " evidence)\n";
            }
            out << "\n";
        }
    }

    if (!issues.empty()) {
        out << "  --- Issues ---\n";
        for (const auto& issue : issues) {
            out << "    [" << toString(issue.severity) << "] " << issue.message << "\n";
        }
        out << "\n";
    }

    if (!request_plan.empty()) {
        out << "  --- Requested Numeric Summaries ---\n";
        for (const auto& request : request_plan.summary_requests) {
            out << "    [" << toString(request.summary_kind) << "]";
            out << " domain=" << toString(request.domain);
            if (!request.request_id.empty()) {
                out << " id=" << request.request_id;
            }
            if (!request.block_id.empty()) {
                out << " block=" << request.block_id;
            }
            if (!request.contribution_id.empty()) {
                out << " contribution=" << request.contribution_id;
            }
            if (!request.scope_id.empty()) {
                out << " scope=" << request.scope_id;
            }
            out << " confidence=" << toString(request.confidence);
            if (request.already_available) {
                out << " available=true";
            }
            out << " (" << request.reasons.size() << " reason";
            if (request.reasons.size() != 1u) {
                out << "s";
            }
            out << ")\n";
            for (const auto& reason : request.reasons) {
                out << "      - " << reason << "\n";
            }
        }
        out << "\n";
    }

    out << "========================================================================\n";
}

void ProblemAnalysisReport::printApplicationLog(std::ostream& out) const {
    const auto shouldEmitClaim = [](const PropertyClaim& claim) {
        const bool applicable =
            claim.applicability_class &&
            *claim.applicability_class == ApplicabilityClass::Applicable;
        return applicable || claim.status == PropertyStatus::Violated;
    };

    std::vector<std::string> analyzers;
    for (const auto& claim : claims) {
        if (shouldEmitClaim(claim)) {
            appendUniqueString(analyzers, claimOrigin(claim));
        }
    }

    out << "[FE/Analysis] Applicable analyzers: ";
    if (analyzers.empty()) {
        out << "none";
    } else {
        out << joinStrings(analyzers);
    }
    out << "\n";

    out << "[FE/Analysis] Requested summaries: ";
    if (request_plan.empty()) {
        out << "none";
    } else {
        std::vector<std::string> requests;
        requests.reserve(request_plan.summary_requests.size());
        for (const auto& request : request_plan.summary_requests) {
            requests.push_back(requestLabel(request));
        }
        out << joinStrings(requests);
    }
    out << "\n";

    if (!request_plan.empty()) {
        std::vector<std::string> available_kinds;
        std::vector<std::string> missing_kinds;
        for (const auto& request : request_plan.summary_requests) {
            if (request.already_available) {
                appendUniqueString(available_kinds, toString(request.summary_kind));
            } else {
                appendUniqueString(missing_kinds, toString(request.summary_kind));
            }
        }
        out << "[FE/Analysis] Summary availability: available="
            << (available_kinds.empty() ? "none" : joinStrings(available_kinds))
            << " missing="
            << (missing_kinds.empty() ? "none" : joinStrings(missing_kinds))
            << "\n";
    }

    bool emitted_claim = false;
    for (const auto& claim : claims) {
        if (!shouldEmitClaim(claim)) {
            continue;
        }

        emitted_claim = true;
        out << "[FE/Analysis] " << claimOrigin(claim)
            << ": kind=" << toString(claim.kind)
            << " status=" << lowerAscii(toString(claim.status));
        appendClaimScope(out, claim);
        appendClaimScalars(out, claim);
        if (!claim.description.empty()) {
            out << " reason=" << claim.description;
        }
        out << "\n";
    }

    if (!emitted_claim) {
        out << "[FE/Analysis] No applicable or violated claims reported.\n";
    }

    for (const auto& issue : issues) {
        out << "[FE/Analysis] Issue severity="
            << lowerAscii(toString(issue.severity))
            << " message=" << issue.message;
        if (!issue.related_claim_indices.empty()) {
            out << " related_claims=";
            for (std::size_t i = 0; i < issue.related_claim_indices.size(); ++i) {
                if (i != 0u) {
                    out << ",";
                }
                out << issue.related_claim_indices[i];
            }
        }
        out << "\n";
    }
}

void ProblemAnalysisReport::printTraceLog(
    std::ostream& out,
    const AnalysisSummarySet* summaries) const
{
    out << "[FE/Analysis][trace] claim_count=" << claims.size()
        << " issue_count=" << issues.size()
        << " requested_summary_count=" << request_plan.size() << "\n";

    for (std::size_t i = 0; i < claims.size(); ++i) {
        const auto& claim = claims[i];
        out << "[FE/Analysis][trace] claim index=" << i
            << " analyzer=" << claimOrigin(claim)
            << " kind=" << toString(claim.kind)
            << " status=" << lowerAscii(toString(claim.status));
        appendClaimScope(out, claim);
        appendClaimScalars(out, claim);
        out << "\n";

        for (std::size_t j = 0; j < claim.evidence.size(); ++j) {
            const auto& evidence = claim.evidence[j];
            out << "[FE/Analysis][trace] evidence claim=" << i
                << " entry=" << j
                << " source=" << evidence.source
                << " confidence=" << toString(evidence.confidence);
            if (evidence.boundary_marker >= 0) {
                out << " marker=" << evidence.boundary_marker;
            }
            if (!evidence.description.empty()) {
                out << " description=" << evidence.description;
            }
            out << "\n";
        }
    }

    if (summaries == nullptr) {
        return;
    }

    for (const auto& summary : summaries->discrete_matrices) {
        traceMatrixSummary(out, "DiscreteMatrix", summary);
    }

    for (const auto& summary : summaries->reduced_matrices) {
        out << "[FE/Analysis][trace] ReducedMatrix constraint_summary"
            << " block=" << blockLabel(summary.free_free_matrix.block)
            << " reduction=" << static_cast<int>(summary.reduction_kind)
            << " free_dofs=" << summary.free_dof_count
            << " constrained_dofs=" << summary.constrained_dof_count
            << " retained_multiplier_dofs="
            << summary.retained_multiplier_dof_count
            << " affine_terms_accounted_for="
            << (summary.affine_terms_accounted_for ? "true" : "false")
            << " exact_for_analysis="
            << (summary.reduction_exact_for_analysis ? "true" : "false")
            << "\n";
        traceMatrixSummary(out, "ReducedMatrix/free_free",
                           summary.free_free_matrix);
    }

    for (const auto& summary : summaries->inf_sup_estimates) {
        out << "[FE/Analysis][trace] InfSupEstimate coupling_summary"
            << " block=" << blockLabel(summary.block)
            << " estimate=" << std::setprecision(17) << summary.estimate_value
            << " tolerance=" << summary.estimate_tolerance
            << " test_rows=" << summary.test_rows
            << " test_cols=" << summary.test_cols
            << " estimator_metadata="
            << (summary.estimator_metadata_present ? "true" : "false")
            << " norm_metadata="
            << (summary.norm_metadata_present ? "true" : "false")
            << " scope=" << summary.estimate_scope
            << "\n";
    }

    for (const auto& summary : summaries->schur_complements) {
        out << "[FE/Analysis][trace] SchurComplement block_summary"
            << " id=" << summary.schur_id
            << " block=" << blockLabel(summary.block)
            << " available=" << (summary.schur_available ? "true" : "false")
            << " exact_for_analysis="
            << (summary.reduction_exact_for_analysis ? "true" : "false")
            << " primal_invertible="
            << (summary.primal_block_invertible_evidence_present ? "true" : "false")
            << " inf_sup="
            << (summary.inf_sup_evidence_present ? "true" : "false")
            << " nullspace_metadata="
            << (summary.nullspace_handling_evidence_present ? "true" : "false")
            << " condition_estimate_present="
            << (summary.condition_estimate_present ? "true" : "false");
        if (summary.condition_estimate_present) {
            out << " condition_estimate="
                << std::setprecision(17) << summary.condition_estimate;
        }
        out << "\n";
    }

    for (const auto& summary : summaries->stabilization_adequacy) {
        out << "[FE/Analysis][trace] StabilizationAdequacy parameter_summary"
            << " id=" << summary.stabilization_id
            << " method=" << summary.method_family
            << " block=" << blockLabel(summary.block)
            << " parameter_formula="
            << (summary.parameter_formula_metadata_present ? "true" : "false")
            << " residual_consistency="
            << (summary.residual_consistency_evidence_present ? "true" : "false")
            << " method_scope="
            << (summary.method_scope_metadata_present ? "true" : "false")
            << " violation_count=" << summary.violation_count
            << "\n";
    }

    for (const auto& summary : summaries->local_stencils) {
        out << "[FE/Analysis][trace] LocalStencil row_summary"
            << " block=" << blockLabel(summary.block)
            << " element=" << summary.element
            << " positive_offdiag=" << summary.positive_offdiag_count
            << " negative_offdiag=" << summary.negative_offdiag_count
            << " near_zero_offdiag=" << summary.near_zero_offdiag_count
            << " sign_tolerance=" << std::setprecision(17)
            << summary.sign_tolerance
            << "\n";
        for (const auto& sample : summary.worst_local_entries) {
            traceMatrixEntry(out, "LocalStencil", sample);
        }
    }

    for (const auto& summary : summaries->mesh_geometry_quality) {
        out << "[FE/Analysis][trace] MeshGeometry element_summary"
            << " mesh_revision=" << summary.mesh_revision
            << " domain=" << toString(summary.domain)
            << " min_jacobian=" << std::setprecision(17)
            << summary.min_jacobian
            << " max_jacobian=" << summary.max_jacobian
            << " inverted_elements=" << summary.inverted_element_count
            << " poor_quality_elements=" << summary.poor_quality_element_count
            << "\n";
        for (ElementId element : summary.worst_elements) {
            out << "[FE/Analysis][trace] MeshGeometry worst_element id="
                << element << "\n";
        }
    }

    for (const auto& summary : summaries->coefficient_properties) {
        out << "[FE/Analysis][trace] CoefficientProperties coefficient_summary"
            << " coefficient=" << summary.coefficient
            << " block=" << blockLabel(summary.block)
            << " tensor_rank=" << static_cast<int>(summary.tensor_rank)
            << " symmetry=" << static_cast<int>(summary.symmetry)
            << " positivity=" << static_cast<int>(summary.positivity)
            << " min=" << std::setprecision(17) << summary.min_eigenvalue
            << " max=" << summary.max_eigenvalue
            << " coverage_scope=" << summary.coverage_scope
            << " quadrature_coverage="
            << (summary.quadrature_point_coverage_complete ? "true" : "false")
            << "\n";
    }

    for (const auto& summary : summaries->parameter_scales) {
        out << "[FE/Analysis][trace] ParameterScale scale_summary"
            << " id=" << summary.nondimensional_parameter_id
            << " role=" << static_cast<int>(summary.role)
            << " block=" << blockLabel(summary.block)
            << " min=" << std::setprecision(17) << summary.min_scale_value
            << " max=" << summary.max_scale_value
            << " required_lower_bound_present="
            << (summary.required_lower_bound_present ? "true" : "false")
            << " theorem=" << summary.scale_theorem_id
            << "\n";
    }

    for (const auto& summary : summaries->boundary_symbols) {
        out << "[FE/Analysis][trace] BoundarySymbol symbol_summary"
            << " block=" << blockLabel(summary.block)
            << " principal_order=" << summary.principal_operator_order
            << " boundary_order=" << summary.boundary_operator_order
            << " bc_count=" << summary.boundary_condition_count
            << " required_bc_count="
            << summary.required_boundary_condition_count
            << " missing_symbols=" << summary.missing_symbol_count
            << " component_coverage="
            << (summary.component_coverage_complete ? "true" : "false")
            << " dof_coverage="
            << (summary.dof_coverage_complete ? "true" : "false")
            << " scope=" << summary.evidence_scope
            << "\n";
    }

    for (const auto& summary : summaries->temporal_stability) {
        out << "[FE/Analysis][trace] TemporalStability stability_summary"
            << " scheme=" << summary.time_scheme
            << " block=" << blockLabel(summary.block)
            << " cfl_present="
            << (summary.cfl_estimate_present ? "true" : "false")
            << " stability_metadata="
            << (summary.stability_metadata_present ? "true" : "false")
            << " operator_scope="
            << (summary.operator_scope_metadata_present ? "true" : "false")
            << "\n";
    }

    for (const auto& summary : summaries->initial_compatibility) {
        out << "[FE/Analysis][trace] InitialCompatibility constraint_summary"
            << " initial_constraint_residual="
            << std::setprecision(17) << summary.initial_constraint_residual
            << " initial_boundary_residual="
            << summary.initial_boundary_residual
            << " invariant_domain_initial_violations="
            << summary.invariant_domain_initial_violation_count
            << " tolerance=" << summary.residual_tolerance
            << "\n";
    }

    for (const auto& summary : summaries->dae_structure_evidence) {
        out << "[FE/Analysis][trace] DAEStructure dae_summary"
            << " system=" << summary.system_id
            << " variables=" << summary.variables.size()
            << " form_class=" << static_cast<int>(summary.dae_form_class)
            << " mass_rank="
            << (summary.mass_matrix_rank_metadata_present ? "true" : "false")
            << " algebraic_rank="
            << (summary.algebraic_jacobian_rank_metadata_present ? "true" : "false")
            << " hidden_constraints="
            << summary.hidden_constraint_count
            << "\n";
    }

    for (const auto& summary : summaries->flux_balances) {
        out << "[FE/Analysis][trace] FluxBalance balance_summary"
            << " group=" << summary.balance_group
            << " block=" << blockLabel(summary.block)
            << " local=" << std::setprecision(17)
            << summary.local_residual_norm
            << " global=" << summary.global_residual_norm
            << " interface=" << summary.interface_pair_residual_norm
            << " symbolic="
            << (summary.symbolic_balance_evidence_present ? "true" : "false")
            << "\n";
    }

    for (const auto& summary : summaries->coupled_system_stability) {
        out << "[FE/Analysis][trace] CoupledSystemStability coupling_summary"
            << " group=" << summary.coupling_group
            << " variables=" << summary.variables.size()
            << " monolithic="
            << (summary.monolithic_coupling ? "true" : "false")
            << " partitioned="
            << (summary.partitioned_coupling ? "true" : "false")
            << " exchange_residual_present="
            << (summary.exchange_residual_present ? "true" : "false")
            << "\n";
    }
}

std::string ProblemAnalysisReport::summary() const {
    auto n_exact   = countByStatus(PropertyStatus::Exact);
    auto n_likely  = countByStatus(PropertyStatus::Likely);
    auto n_unknown = countByStatus(PropertyStatus::Unknown);
    auto n_errors  = countBySeverity(IssueSeverity::Error);
    auto n_warns   = countBySeverity(IssueSeverity::Warning);

    std::ostringstream oss;
    oss << claims.size() << " claims (";
    oss << n_exact << " exact, " << n_likely << " likely, " << n_unknown << " unknown";
    oss << "), " << issues.size() << " issues (";
    oss << n_warns << " warnings, " << n_errors << " errors)";
    if (!request_plan.empty()) {
        oss << ", " << request_plan.size() << " requested numeric summaries";
    }
    return oss.str();
}

} // namespace analysis
} // namespace FE
} // namespace svmp

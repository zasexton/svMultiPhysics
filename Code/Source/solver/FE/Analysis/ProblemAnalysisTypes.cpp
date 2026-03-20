/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ProblemAnalysisTypes.h"

#include <algorithm>
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
    }
    return "Unknown";
}

// Phase 21 toString helpers

const char* toString(InfSupClass c) noexcept {
    switch (c) {
        case InfSupClass::Required:              return "Required";
        case InfSupClass::StructurallySupported: return "StructurallySupported";
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

    out << "========================================================================\n";
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
    return oss.str();
}

} // namespace analysis
} // namespace FE
} // namespace svmp

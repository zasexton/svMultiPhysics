/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingDiagnostics.h"

#include "Core/FEException.h"

#include <sstream>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

bool CouplingValidationResult::ok() const noexcept
{
    for (const auto& diagnostic : diagnostics) {
        if (diagnostic.severity == CouplingDiagnosticSeverity::Error) {
            return false;
        }
    }
    return true;
}

void CouplingValidationResult::add(CouplingDiagnostic diagnostic)
{
    diagnostics.push_back(std::move(diagnostic));
}

void CouplingValidationResult::addError(std::string message)
{
    diagnostics.push_back(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .message = std::move(message),
    });
}

void CouplingValidationResult::addWarning(std::string message)
{
    diagnostics.push_back(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Warning,
        .message = std::move(message),
    });
}

void CouplingValidationResult::append(const CouplingValidationResult& other)
{
    diagnostics.insert(diagnostics.end(), other.diagnostics.begin(), other.diagnostics.end());
}

const char* toString(CouplingDiagnosticSeverity severity) noexcept
{
    switch (severity) {
    case CouplingDiagnosticSeverity::Info:
        return "info";
    case CouplingDiagnosticSeverity::Warning:
        return "warning";
    case CouplingDiagnosticSeverity::Error:
        return "error";
    }
    return "unknown";
}

std::string formatDiagnostic(const CouplingDiagnostic& diagnostic)
{
    std::ostringstream os;
    os << toString(diagnostic.severity) << ": " << diagnostic.message;
    if (!diagnostic.contract_name.empty()) {
        os << " contract='" << diagnostic.contract_name << "'";
    }
    if (!diagnostic.participant_name.empty()) {
        os << " participant='" << diagnostic.participant_name << "'";
    }
    if (!diagnostic.field_name.empty()) {
        os << " field='" << diagnostic.field_name << "'";
    }
    if (!diagnostic.region_name.empty()) {
        os << " region='" << diagnostic.region_name << "'";
    }
    if (!diagnostic.endpoint_name.empty()) {
        os << " endpoint='" << diagnostic.endpoint_name << "'";
    }
    return os.str();
}

std::string formatDiagnostics(const CouplingValidationResult& result)
{
    std::ostringstream os;
    for (std::size_t i = 0; i < result.diagnostics.size(); ++i) {
        if (i != 0u) {
            os << '\n';
        }
        os << formatDiagnostic(result.diagnostics[i]);
    }
    return os.str();
}

void throwIfInvalid(const CouplingValidationResult& result)
{
    if (!result.ok()) {
        FE_THROW(InvalidArgumentException, formatDiagnostics(result));
    }
}

} // namespace coupling
} // namespace FE
} // namespace svmp

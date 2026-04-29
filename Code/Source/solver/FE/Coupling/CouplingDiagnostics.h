/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGDIAGNOSTICS_H
#define SVMP_FE_COUPLING_COUPLINGDIAGNOSTICS_H

/**
 * @file CouplingDiagnostics.h
 * @brief Setup-time diagnostics for physics-agnostic coupling infrastructure.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

enum class CouplingDiagnosticSeverity : std::uint8_t {
    Info,
    Warning,
    Error,
};

struct CouplingDiagnostic {
    CouplingDiagnosticSeverity severity{CouplingDiagnosticSeverity::Error};
    std::string contract_name;
    std::string participant_name;
    std::string field_name;
    std::string region_name;
    std::string endpoint_name;
    std::string message;
};

struct CouplingValidationResult {
    std::vector<CouplingDiagnostic> diagnostics;

    [[nodiscard]] bool ok() const noexcept;
    void add(CouplingDiagnostic diagnostic);
    void addError(std::string message);
    void addWarning(std::string message);
    void append(const CouplingValidationResult& other);
};

[[nodiscard]] const char* toString(CouplingDiagnosticSeverity severity) noexcept;
[[nodiscard]] std::string formatDiagnostic(const CouplingDiagnostic& diagnostic);
[[nodiscard]] std::string formatDiagnostics(const CouplingValidationResult& result);

void throwIfInvalid(const CouplingValidationResult& result);

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGDIAGNOSTICS_H

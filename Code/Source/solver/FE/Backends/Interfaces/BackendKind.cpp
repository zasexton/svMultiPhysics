/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Interfaces/BackendKind.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace svmp {
namespace FE {
namespace backends {

std::string_view backendKindToString(BackendKind kind) noexcept
{
    switch (kind) {
        case BackendKind::Eigen: return "eigen";
        case BackendKind::FSILS: return "fsils";
        case BackendKind::PETSc: return "petsc";
        case BackendKind::Trilinos: return "trilinos";
        default: return "unknown";
    }
}

std::optional<BackendKind> backendKindFromString(std::string_view name) noexcept
{
    std::string lowered{name};
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (lowered == "eigen") return BackendKind::Eigen;
    if (lowered == "fsils") return BackendKind::FSILS;
    if (lowered == "petsc") return BackendKind::PETSc;
    if (lowered == "trilinos") return BackendKind::Trilinos;
    return std::nullopt;
}

} // namespace backends
} // namespace FE
} // namespace svmp


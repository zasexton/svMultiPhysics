/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_BACKEND_KIND_H
#define SVMP_FE_BACKENDS_BACKEND_KIND_H

#include <cstdint>
#include <optional>
#include <string_view>

namespace svmp {
namespace FE {
namespace backends {

/**
 * @brief Backend selection for FE linear algebra / solvers
 *
 * This is intentionally FE-scoped (separate from the legacy solver's linear
 * algebra selection) and is used by `backends::BackendFactory`.
 */
enum class BackendKind : std::uint8_t {
    Eigen,
    FSILS,
    PETSc,
    Trilinos
};

[[nodiscard]] std::string_view backendKindToString(BackendKind kind) noexcept;

[[nodiscard]] std::optional<BackendKind> backendKindFromString(std::string_view name) noexcept;

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_BACKEND_KIND_H


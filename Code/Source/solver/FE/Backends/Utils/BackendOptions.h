/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_BACKEND_OPTIONS_H
#define SVMP_FE_BACKENDS_BACKEND_OPTIONS_H

#include "Core/Types.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

enum class SolverMethod : std::uint8_t {
    Direct,
    CG,
    BiCGSTAB,
    GMRES,
    FGMRES,
    BlockSchur
};

enum class PreconditionerType : std::uint8_t {
    None,
    Diagonal,
    ILU,
    AMG,
    RowColumnScaling,
    FieldSplit
};

enum class FieldSplitKind : std::uint8_t {
    Additive,
    Multiplicative,
    Schur
};

struct FieldSplitOptions {
    FieldSplitKind kind{FieldSplitKind::Additive};
    std::vector<std::string> split_names{}; // Optional, defaults to field0/field1/...
};

struct SolverOptions {
    SolverMethod method{SolverMethod::Direct};
    PreconditionerType preconditioner{PreconditionerType::None};

    Real rel_tol{1e-10};
    Real abs_tol{0.0};
    int max_iter{1000};
    bool use_initial_guess{false};

    FieldSplitOptions fieldsplit{};

    // Backend-specific pass-through (optional).
    // PETSc: used with KSPSetOptionsPrefix()/KSPSetFromOptions().
    // Example (CLI): -my_solver_ksp_type gmres
    // Example (prefix): petsc_options_prefix="my_solver_"
    std::string petsc_options_prefix{};

    // Trilinos: optional Teuchos::ParameterList XML file for solver configuration.
    std::string trilinos_xml_file{};

    // FSILS: best-effort row/column scaling toggle.
    bool fsils_use_rcs{false};

    // Backend-specific pass-through key/value list (optional).
    // - PETSc: key maps to an option name (with or without '-' prefix).
    // - Trilinos: key maps to a Teuchos::ParameterList entry.
    std::vector<std::pair<std::string, std::string>> passthrough{};
};

struct SolverReport {
    bool converged{false};
    int iterations{0};
    Real initial_residual_norm{0.0};
    Real final_residual_norm{0.0};
    Real relative_residual{0.0};
    std::string message{};
};

[[nodiscard]] std::string_view solverMethodToString(SolverMethod m) noexcept;
[[nodiscard]] std::string_view preconditionerToString(PreconditionerType pc) noexcept;
[[nodiscard]] std::string_view fieldSplitKindToString(FieldSplitKind kind) noexcept;

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_BACKEND_OPTIONS_H

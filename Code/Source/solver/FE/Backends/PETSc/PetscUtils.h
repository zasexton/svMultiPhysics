/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_PETSC_UTILS_H
#define SVMP_FE_BACKENDS_PETSC_UTILS_H

#include "Core/FEException.h"

#if defined(FE_HAS_PETSC)
#include <petscsys.h>

#include <string>

namespace svmp {
namespace FE {
namespace backends {
namespace petsc {

[[nodiscard]] inline std::string errorMessage(PetscErrorCode ierr)
{
    const char* msg = nullptr;
    PetscErrorMessage(ierr, &msg, nullptr);
    if (msg) {
        return std::string(msg);
    }
    return "unknown PETSc error";
}

inline void check(PetscErrorCode ierr, const char* expr, const char* file, int line)
{
    if (ierr == 0) {
        return;
    }
    FE_THROW(FEException,
             std::string("PETSc call failed: ") + expr + " (" + errorMessage(ierr) + ") at " + file + ":" +
                 std::to_string(line));
}

} // namespace petsc
} // namespace backends
} // namespace FE
} // namespace svmp

#define FE_PETSC_CALL(expr) ::svmp::FE::backends::petsc::check((expr), #expr, __FILE__, __LINE__)

#endif // FE_HAS_PETSC

#endif // SVMP_FE_BACKENDS_PETSC_UTILS_H


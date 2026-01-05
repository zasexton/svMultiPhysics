/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_LINEAR_SOLVER_H
#define SVMP_FE_BACKENDS_LINEAR_SOLVER_H

#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Backends/Utils/BackendOptions.h"

namespace svmp {
namespace FE {
namespace backends {

class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    [[nodiscard]] virtual BackendKind backendKind() const noexcept = 0;

    virtual void setOptions(const SolverOptions& options) = 0;
    [[nodiscard]] virtual const SolverOptions& getOptions() const noexcept = 0;

    [[nodiscard]] virtual SolverReport solve(const GenericMatrix& A,
                                             GenericVector& x,
                                             const GenericVector& b) = 0;
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_LINEAR_SOLVER_H


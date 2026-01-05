/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_TRILINOS_LINEAR_SOLVER_H
#define SVMP_FE_BACKENDS_TRILINOS_LINEAR_SOLVER_H

#include "Backends/Interfaces/LinearSolver.h"

#if defined(FE_HAS_TRILINOS)

namespace svmp {
namespace FE {
namespace backends {

class TrilinosLinearSolver final : public LinearSolver {
public:
    explicit TrilinosLinearSolver(const SolverOptions& options);

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::Trilinos; }

    void setOptions(const SolverOptions& options) override;
    [[nodiscard]] const SolverOptions& getOptions() const noexcept override { return options_; }

    [[nodiscard]] SolverReport solve(const GenericMatrix& A,
                                     GenericVector& x,
                                     const GenericVector& b) override;

private:
    SolverOptions options_{};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_TRILINOS

#endif // SVMP_FE_BACKENDS_TRILINOS_LINEAR_SOLVER_H


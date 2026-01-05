/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_PETSC_LINEAR_SOLVER_H
#define SVMP_FE_BACKENDS_PETSC_LINEAR_SOLVER_H

#include "Backends/Interfaces/LinearSolver.h"

#if defined(FE_HAS_PETSC)

#include "Backends/PETSc/PetscUtils.h"

#include <petscksp.h>

namespace svmp {
namespace FE {
namespace backends {

class PetscLinearSolver final : public LinearSolver {
public:
    explicit PetscLinearSolver(const SolverOptions& options);
    ~PetscLinearSolver() override;

    PetscLinearSolver(PetscLinearSolver&& other) noexcept;
    PetscLinearSolver& operator=(PetscLinearSolver&& other) noexcept;

    PetscLinearSolver(const PetscLinearSolver&) = delete;
    PetscLinearSolver& operator=(const PetscLinearSolver&) = delete;

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::PETSc; }

    void setOptions(const SolverOptions& options) override;
    [[nodiscard]] const SolverOptions& getOptions() const noexcept override { return options_; }

    [[nodiscard]] SolverReport solve(const GenericMatrix& A,
                                     GenericVector& x,
                                     const GenericVector& b) override;

private:
    void ensureKspCreated();
    void applyBaseOptionsToKsp();
    void applyKspFromOptions();

    SolverOptions options_{};

    KSP ksp_{nullptr};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_PETSC

#endif // SVMP_FE_BACKENDS_PETSC_LINEAR_SOLVER_H

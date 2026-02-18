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
#include "Core/Types.h"

#include <span>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

/// Represents a symmetric rank-1 perturbation: J += sigma * v * v^T
/// where v is stored as a sparse vector of (global_dof_index, value) pairs.
struct RankOneUpdate {
    Real sigma{0.0};
    std::vector<std::pair<GlobalIndex, Real>> v;
};

class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    [[nodiscard]] virtual BackendKind backendKind() const noexcept = 0;

    virtual void setOptions(const SolverOptions& options) = 0;
    [[nodiscard]] virtual const SolverOptions& getOptions() const noexcept = 0;

    [[nodiscard]] virtual SolverReport solve(const GenericMatrix& A,
                                             GenericVector& x,
                                             const GenericVector& b) = 0;

    /// Provide rank-1 updates (J += sigma * v * v^T) for preconditioner correction.
    /// Backends that support native handling (e.g. FSILS face mechanism) will use these
    /// for both the matrix-free mat-vec and the preconditioner correction.
    virtual void setRankOneUpdates(std::span<const RankOneUpdate> /*updates*/) {}

    /// Provide an "effective" time step size for the current nonlinear stage (e.g. α_f*dt for
    /// generalized-α). Backends may use this to scale equation rows internally for improved
    /// conditioning and to match legacy solver conventions for coupled boundary-condition
    /// linearization.
    ///
    /// Default is a no-op.
    virtual void setEffectiveTimeStep(double /*dt_eff*/) {}

    /// Provide the set of Dirichlet-constrained DOFs for the current solve.
    ///
    /// Some backends (notably FSILS) use this information to apply the same
    /// boundary-condition handling as the legacy solver (e.g., via FSILS faces)
    /// to improve robustness of specialized solvers like the Navier-Stokes
    /// Block-Schur method.
    ///
    /// DOF indices are in the FE system's global numbering.
    ///
    /// Default is a no-op.
    virtual void setDirichletDofs(std::span<const GlobalIndex> /*dofs*/) {}

    /// Returns true if this backend handles rank-1 updates natively (mat-vec + preconditioner).
    [[nodiscard]] virtual bool supportsNativeRankOneUpdates() const noexcept { return false; }
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_LINEAR_SOLVER_H

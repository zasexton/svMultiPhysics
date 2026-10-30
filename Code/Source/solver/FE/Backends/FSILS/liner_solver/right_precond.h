/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef FSI_LINEAR_SOLVER_RIGHT_PRECOND_H
#define FSI_LINEAR_SOLVER_RIGHT_PRECOND_H

#include "Array.h"

#include <functional>

namespace fe_fsi_linear_solver {

class FSILS_lhsType;

/// Optional right preconditioner M for the vector GMRES kernel.
///
/// GMRES then builds its Krylov space for A M^{-1} and returns x = M^{-1} V y.
/// For a right preconditioner the Arnoldi residual equals the residual of the
/// (scaled) original system, so the FSILS stopping tests are unchanged.
class FSILS_rightPreconditioner {
public:
  virtual ~FSILS_rightPreconditioner() = default;

  /// out = M^{-1} in on owned nodes (internal ordering, `dof x nNo` arrays).
  /// Ghost entries of `out` are set to zero; callers synchronize ghosts.
  virtual void apply(const Array<double>& in, Array<double>& out) const = 0;
};

/// Hook used by fsils_solve() to obtain a right preconditioner for GMRES.
///
/// `prepare` is called after the scaling preconditioner has modified Val and
/// the right-hand side; it builds the preconditioner from that operator and
/// returns nullptr to run the unpreconditioned kernel.
///
/// `finish` is called once per fsils_solve() with the GMRES iterations, the
/// residual reduction and the convergence flag.
struct FSILS_rightPreconditionerHook {
  std::function<const FSILS_rightPreconditioner*(const FSILS_lhsType& lhs,
                                                 int dof,
                                                 const Array<double>& Val)>
      prepare{};
  std::function<void(int iterations, double residual_reduction, bool converged)> finish{};

  [[nodiscard]] bool active() const noexcept { return static_cast<bool>(prepare); }
};

} // namespace fe_fsi_linear_solver

#endif

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
/// the right-hand side.  `row_scale`/`col_scale` are the applied diagonal
/// scalings (nullptr when none).  `force_refresh` requests a fresh
/// preconditioner (used to repeat a solve that failed with a reused one).
/// The callee sets `fresh` to true when it built the preconditioner from this
/// operator, and returns nullptr to run the unpreconditioned kernel.
///
/// `stale_iteration_cap` returns the iteration budget of a solve that uses a
/// reused preconditioner for a target residual reduction (0: no budget); a
/// solve exceeding it is abandoned and repeated with a fresh preconditioner.
///
/// `finish` is called once per fsils_solve() with the GMRES iterations and the
/// residual reduction of the final attempt, the convergence flag, whether the
/// final attempt used a fresh preconditioner and whether it was a repeat.
struct FSILS_rightPreconditionerHook {
  std::function<const FSILS_rightPreconditioner*(const FSILS_lhsType& lhs,
                                                 int dof,
                                                 const Array<double>& Val,
                                                 const Array<double>* row_scale,
                                                 const Array<double>* col_scale,
                                                 bool force_refresh,
                                                 bool& fresh)>
      prepare{};
  std::function<int(double target_reduction)> stale_iteration_cap{};
  std::function<void(int iterations, double residual_reduction, bool converged, bool fresh, bool retried)>
      finish{};

  [[nodiscard]] bool active() const noexcept { return static_cast<bool>(prepare); }
};

} // namespace fe_fsi_linear_solver

#endif

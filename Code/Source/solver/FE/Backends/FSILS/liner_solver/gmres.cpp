/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-------------------------------------------------------------------------
// Generalized minimum residual algorithm.
// Optimized for VMS consistent tangents using CGS2 (Re-orthogonalization),
// Cache-Blocked Fused BLAS updates, zero-allocation tracking, and hypot stability.
//-------------------------------------------------------------------------

#include "gmres.h"

#include "fsils_api.hpp"

#include "add_bc_mul.h"
#include "bcast.h"
#include "dot.h"
#include "norm.h"
#include "omp_la.h"
#include "spar_mul.h"

#include "Array3.h"

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace gmres {

/// @brief Pre-calculates the boundary condition normalization factor.
/// Removed O(N) heap allocation, using fast local accumulations.
void bc_pre(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
            const int mynNo, const int nNo)
{
  int nsd = dof - 1;
  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto &face = lhs.face[faIn];

    if (face.coupledFlag) {
      if (face.sharedFlag) {
        double local_nS = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          if (Ac < mynNo) {
            for (int i = 0; i < nsd; i++) {
              local_nS += face.valM(i,a) * face.valM(i,a);
            }
          }
        }

        double global_nS = 0.0;
        if (lhs.commu.nTasks > 1) {
          MPI_Allreduce(&local_nS, &global_nS, 1, cm_mod::mpreal, MPI_SUM, lhs.commu.comm);
        } else {
          global_nS = local_nS;
        }
        face.nS = global_nS;

      } else {
        face.nS = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          for (int i = 0; i < nsd; i++) {
            face.nS += face.valM(i,a) * face.valM(i,a);
          }
        }
      }
    }
  }
}

/// @brief Solver the system Val * X = R.
///
/// Reproduces the Fortran 'GMRES' subroutine.
void gmres(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
           const Array<double>& Val, const Array<double>& R, Array<double>& X)
{
  using namespace fe_fsi_linear_solver;
  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;

  ls.ws.ensure_gmres_v(dof, nNo, ls.sD);
  auto& h = ls.ws.h;
  auto& u = ls.ws.u3;
  auto& unCondU = ls.ws.unCondU;
  auto& y = ls.ws.y;
  auto& c = ls.ws.c;
  auto& s = ls.ws.s;
  auto& err = ls.ws.err;

  double time = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;
  double eps = 0.0;
  int last_i = 0;
  X = 0.0;

  // Pre-allocate workspace for Hessenberg column (avoids per-iteration heap allocation)
  std::vector<double> h_col(ls.sD + 1);

  for (int l = 0; l < ls.mItr; l++) {
    auto u_slice = u.rslice(0);

    if (l == 0) {
      u.set_slice(0, R);
    } else {
      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, X, u_slice);
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, X, u_slice);
      ls.itr = ls.itr + 1;

      // Zero-allocation residual: u_slice = R - u_slice
      omp_la::omp_axpby_v(dof, nNo, u_slice, R, -1.0, u_slice);
    }

    for (auto& face : lhs.face) {
      if (face.coupledFlag) {
        auto unCondU_ref = u.rslice(0);
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU_ref, u_slice);
        break;
      }
    }

    err[0] = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u_slice);

    if (l == 0) {
      eps = err[0];
      ls.iNorm = eps;
      ls.fNorm = eps;
      eps = std::max(ls.absTol, ls.relTol*eps);
    }

    ls.dB = ls.fNorm;

    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true;
      break;
    }

    omp_la::omp_mul_v(dof, nNo, 1.0 / err[0], u_slice);

    for (int i = 0; i < ls.sD; i++) {
      last_i = i;
      auto u_slice_prev = u.rslice(i);
      auto u_slice_next = u.rslice(i+1);

      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, u_slice_prev, u_slice_next);
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, u_slice_prev, u_slice_next);
      ls.itr = ls.itr + 1;

      for (auto& face : lhs.face) {
        if (face.coupledFlag) {
          auto unCondU_ref = u.rslice(i+1);
          add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU_ref, u_slice_next);
          break;
        }
      }

      // ----------------------------------------------------------------------
      // CGS2: Classical Gram-Schmidt with Re-orthogonalization
      // ----------------------------------------------------------------------
      for (int j = 0; j <= i; j++) {
        h_col[j] = dot::fsils_nc_dot_v(dof, mynNo, u.rslice(j), u_slice_next);
      }

      bcast::fsils_bcast_v(i+1, h_col, lhs.commu);

      double proj_sq_norm = 0.0;
      for (int j = 0; j <= i; j++) {
        h(j,i) = h_col[j];
        proj_sq_norm += h_col[j] * h_col[j];
      }

      // Cache-Fused Pass 1
      for (int j = 0; j <= i; ) {
        if (j + 1 <= i) {
          omp_la::omp_axpbypgz_v(dof, nNo, u_slice_next, u_slice_next, -h(j,i), u.rslice(j), -h(j+1,i), u.rslice(j+1));
          j += 2;
        } else {
          omp_la::omp_sum_v(dof, nNo, -h(j,i), u_slice_next, u.rslice(j));
          j++;
        }
      }

      double new_norm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u_slice_next);

      // CGS2: Iterated Gram-Schmidt Correction
      // Pythagorean identity: ||v||^2 = sum(h^2) + ||v_new||^2
      // Check: ||v_new||^2 < sum(h^2) ≡ ||v_new|| < (1/√2)||v||
      if (new_norm * new_norm < proj_sq_norm && new_norm > std::numeric_limits<double>::epsilon()) {
        for (int j = 0; j <= i; j++) h_col[j] = dot::fsils_nc_dot_v(dof, mynNo, u.rslice(j), u_slice_next);
        bcast::fsils_bcast_v(i+1, h_col, lhs.commu);

        for (int j = 0; j <= i; ) {
          if (j + 1 <= i) {
            omp_la::omp_axpbypgz_v(dof, nNo, u_slice_next, u_slice_next, -h_col[j], u.rslice(j), -h_col[j+1], u.rslice(j+1));
            h(j,i) += h_col[j];
            h(j+1,i) += h_col[j+1];
            j += 2;
          } else {
            omp_la::omp_sum_v(dof, nNo, -h_col[j], u_slice_next, u.rslice(j));
            h(j,i) += h_col[j];
            j++;
          }
        }
        new_norm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u_slice_next);
      }

      h(i+1,i) = new_norm;

      // Happy Breakdown Protection & Safe Givens Rotation
      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (h(i+1,i) > breakdown_tol) {
        omp_la::omp_mul_v(dof, nNo, 1.0/h(i+1,i), u_slice_next);
      } else {
        h(i+1,i) = 0.0;
        breakdown = true;
      }

      for (int j = 0; j <= i-1; j++) {
        double tmp_h = c(j)*h(j,i) + s(j)*h(j+1,i);
        h(j+1,i) = -s(j)*h(j,i) + c(j)*h(j+1,i);
        h(j,i) = tmp_h;
      }

      double tmp_hypot = std::hypot(h(i,i), h(i+1,i));
      if (tmp_hypot == 0.0) {
        c(i) = 1.0; s(i) = 0.0;
      } else {
        c(i) = h(i,i) / tmp_hypot; s(i) = h(i+1,i) / tmp_hypot;
      }

      h(i,i) = tmp_hypot;
      h(i+1,i) = 0.0;

      err(i+1) = -s(i)*err(i);
      err(i) = c(i)*err(i);

      if (std::abs(err(i+1)) < eps || breakdown) {
        ls.suc = true;
        break;
      }
    }

    if (last_i >= ls.sD) last_i = ls.sD - 1;

    for (int i = 0; i <= last_i; i++) y(i) = err(i);

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) y(j) -= h(j,k)*y(k);
      y(j) /= h(j,j);
    }

    // Cache-Fused Solution Reconstruction
    for (int j = 0; j <= last_i; ) {
      if (j + 1 <= last_i) {
        omp_la::omp_axpbypgz_v(dof, nNo, X, X, y(j), u.rslice(j), y(j+1), u.rslice(j+1));
        j += 2;
      } else {
        omp_la::omp_sum_v(dof, nNo, y(j), X, u.rslice(j));
        j++;
      }
    }

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) break;
  }

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - time + ls.callD;
  ls.dB  = 10.0 * std::log(ls.fNorm / ls.dB);
}

//---------
// gmres_s
//---------
// Reproduces the Fortran 'GMRESS' subroutine.
//
void gmres_s(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
             const Vector<double>& Val, Vector<double>& R)
{
  using namespace fe_fsi_linear_solver;
  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;

  ls.ws.ensure_gmres_s(nNo, ls.sD);
  auto& h = ls.ws.h;
  auto& u = ls.ws.u2;
  auto& X = ls.ws.Xs;
  auto& y = ls.ws.y;
  auto& c = ls.ws.c;
  auto& s = ls.ws.s;
  auto& err = ls.ws.err;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;

  double eps = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  ls.iNorm = eps;
  ls.fNorm = eps;
  eps = std::max(ls.absTol, ls.relTol*eps);
  ls.itr = 0;
  int last_i = 0;
  X = 0.0;

  if (ls.iNorm <= ls.absTol) {
    ls.callD = std::numeric_limits<double>::epsilon();
    ls.dB = 0.0;
    return;
  }

  // Pre-allocate workspace for Hessenberg column
  std::vector<double> h_local(ls.sD + 1);

  for (int l = 0; l < ls.mItr; l++) {
    ls.dB = ls.fNorm;
    ls.itr++;

    auto u_col_curr = u.col(0);

    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, Val, X, u_col_curr);
    omp_la::omp_axpby_s(nNo, u_col_curr, R, -1.0, u_col_curr);

    err[0] = norm::fsi_ls_norms(mynNo, lhs.commu, u_col_curr);
    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true; break;
    }

    omp_la::omp_mul_s(nNo, 1.0 / err[0], u_col_curr);
    u.set_col(0, u_col_curr);

    for (int i = 0; i < ls.sD; i++) {
      ls.itr++;
      last_i = i;

      auto u_col_prev = u.col(i);
      auto u_col_next = u.col(i+1);
      spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, Val, u_col_prev, u_col_next);

      // --- CGS Step 1 ---
      for (int j = 0; j <= i; j++) {
        h_local[j] = dot::fsils_nc_dot_s(mynNo, u.col(j), u_col_next);
      }

      bcast::fsils_bcast_v(i+1, h_local, lhs.commu);

      double proj_sq_norm = 0.0;
      for (int j = 0; j <= i; j++) {
        h(j,i) = h_local[j];
        proj_sq_norm += h_local[j] * h_local[j];
      }

      // Cache-Fused Pass 1
      for (int j = 0; j <= i; ) {
        if (j + 1 <= i) {
          omp_la::omp_axpbypgz_s(nNo, u_col_next, u_col_next, -h(j,i), u.col(j), -h(j+1,i), u.col(j+1));
          j += 2;
        } else {
          omp_la::omp_sum_s(nNo, -h(j,i), u_col_next, u.col(j));
          j++;
        }
      }

      double new_norm = norm::fsi_ls_norms(mynNo, lhs.commu, u_col_next);

      // CGS2: Iterated Gram-Schmidt Correction (Pythagorean check)
      if (new_norm * new_norm < proj_sq_norm && new_norm > std::numeric_limits<double>::epsilon()) {
        for (int j = 0; j <= i; j++) h_local[j] = dot::fsils_nc_dot_s(mynNo, u.col(j), u_col_next);
        bcast::fsils_bcast_v(i+1, h_local, lhs.commu);

        for (int j = 0; j <= i; ) {
          if (j + 1 <= i) {
            omp_la::omp_axpbypgz_s(nNo, u_col_next, u_col_next, -h_local[j], u.col(j), -h_local[j+1], u.col(j+1));
            h(j,i) += h_local[j];
            h(j+1,i) += h_local[j+1];
            j += 2;
          } else {
            omp_la::omp_sum_s(nNo, -h_local[j], u_col_next, u.col(j));
            h(j,i) += h_local[j];
            j++;
          }
        }
        new_norm = norm::fsi_ls_norms(mynNo, lhs.commu, u_col_next);
      }

      h(i+1,i) = new_norm;

      // Happy Breakdown Check
      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (h(i+1,i) > breakdown_tol) {
        omp_la::omp_mul_s(nNo, 1.0/h(i+1,i), u_col_next);
      } else {
        h(i+1,i) = 0.0;
        breakdown = true;
      }
      u.set_col(i+1, u_col_next);

      for (int j = 0; j <= i-1; j++) {
        double tmp_h = c(j)*h(j,i) + s(j)*h(j+1,i);
        h(j+1,i) = -s(j)*h(j,i) + c(j)*h(j+1,i);
        h(j,i) = tmp_h;
      }

      double tmp_hypot = std::hypot(h(i,i), h(i+1,i));
      if (tmp_hypot == 0.0) {
        c(i) = 1.0; s(i) = 0.0;
      } else {
        c(i) = h(i,i) / tmp_hypot; s(i) = h(i+1,i) / tmp_hypot;
      }

      h(i,i) = tmp_hypot;
      h(i+1,i) = 0.0;
      err(i+1) = -s(i)*err(i);
      err(i) = c(i)*err(i);

      if (std::abs(err(i+1)) < eps || breakdown) {
        ls.suc = true;
        break;
      }
    }

    if (last_i >= ls.sD) last_i = ls.sD - 1;

    for (int i = 0; i <= last_i; i++) y(i) = err(i);

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) y(j) -= h(j,k)*y(k);
      y(j) /= h(j,j);
    }

    // Cache-Fused Reconstruction
    for (int j = 0; j <= last_i; ) {
      if (j + 1 <= last_i) {
        omp_la::omp_axpbypgz_s(nNo, X, X, y(j), u.col(j), y(j+1), u.col(j+1));
        j += 2;
      } else {
        omp_la::omp_sum_s(nNo, y(j), X, u.col(j));
        j++;
      }
    }

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) break;
  }

  R = X;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB  = 10.0 * std::log(ls.fNorm / ls.dB);
}

//---------
// gmres_v
//---------
// Generalized minimum residual algorithm implemented for vector problems.
//
// Reproduces the Fortran 'GMRESV' subroutine.
//
void gmres_v(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
             const Array<double>& Val, Array<double>& R)
{
  using namespace fe_fsi_linear_solver;
  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;

  ls.ws.ensure_gmres_v(dof, nNo, ls.sD);
  auto& h = ls.ws.h;
  auto& u = ls.ws.u3;
  auto& X = ls.ws.X2;
  auto& unCondU = ls.ws.unCondU;
  auto& y = ls.ws.y;
  auto& c = ls.ws.c;
  auto& s = ls.ws.s;
  auto& err = ls.ws.err;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;

  double eps = norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
  ls.iNorm = eps;
  ls.fNorm = eps;
  eps = std::max(ls.absTol, ls.relTol*eps);
  ls.itr = 0;
  int last_i = 0;
  X = 0.0;

  bc_pre(lhs, ls, dof, mynNo, nNo);

  if (ls.iNorm <= ls.absTol) {
    ls.callD = std::numeric_limits<double>::epsilon();
    ls.dB = 0.0;
    return;
  }

  // Pre-allocate workspace for Hessenberg column
  std::vector<double> h_col(ls.sD + 1);

  for (int l = 0; l < ls.mItr; l++) {
    ls.dB = ls.fNorm;
    ls.itr++;

    auto u_slice = u.rslice(0);
    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, X, u_slice);
    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, X, u_slice);

    omp_la::omp_axpby_v(dof, nNo, u_slice, R, -1.0, u_slice);

    for (auto& face : lhs.face) {
      if (face.coupledFlag) {
        auto unCondU_ref = u.rslice(0);
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU_ref, u_slice);
        break;
      }
    }

    err[0] = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u.rslice(0));
    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true; break;
    }

    omp_la::omp_mul_v(dof, nNo, 1.0 / err[0], u_slice);

    for (int i = 0; i < ls.sD; i++) {
      ls.itr++;
      last_i = i;
      auto u_slice_prev = u.rslice(i);
      auto u_slice_next = u.rslice(i+1);

      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, u_slice_prev, u_slice_next);
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, u_slice_prev, u_slice_next);

      for (auto& face : lhs.face) {
        if (face.coupledFlag) {
          auto unCondU_ref = u.rslice(i+1);
          add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU_ref, u_slice_next);
          break;
        }
      }

      // --- CGS Step 1 ---
      for (int j = 0; j <= i; j++) {
        h_col[j] = dot::fsils_nc_dot_v(dof, mynNo, u.rslice(j), u_slice_next);
      }

      bcast::fsils_bcast_v(i+1, h_col, lhs.commu);

      double proj_sq_norm = 0.0;
      for (int j = 0; j <= i; j++) {
        h(j,i) = h_col[j];
        proj_sq_norm += h_col[j] * h_col[j];
      }

      // Cache-Fused Pass 1
      for (int j = 0; j <= i; ) {
        if (j + 1 <= i) {
          omp_la::omp_axpbypgz_v(dof, nNo, u_slice_next, u_slice_next, -h(j,i), u.rslice(j), -h(j+1,i), u.rslice(j+1));
          j += 2;
        } else {
          omp_la::omp_sum_v(dof, nNo, -h(j,i), u_slice_next, u.rslice(j));
          j++;
        }
      }

      double new_norm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u_slice_next);

      // CGS2: Iterated Gram-Schmidt Correction (Pythagorean check)
      if (new_norm * new_norm < proj_sq_norm && new_norm > std::numeric_limits<double>::epsilon()) {
        for (int j = 0; j <= i; j++) h_col[j] = dot::fsils_nc_dot_v(dof, mynNo, u.rslice(j), u_slice_next);
        bcast::fsils_bcast_v(i+1, h_col, lhs.commu);

        for (int j = 0; j <= i; ) {
          if (j + 1 <= i) {
            omp_la::omp_axpbypgz_v(dof, nNo, u_slice_next, u_slice_next, -h_col[j], u.rslice(j), -h_col[j+1], u.rslice(j+1));
            h(j,i) += h_col[j];
            h(j+1,i) += h_col[j+1];
            j += 2;
          } else {
            omp_la::omp_sum_v(dof, nNo, -h_col[j], u_slice_next, u.rslice(j));
            h(j,i) += h_col[j];
            j++;
          }
        }
        new_norm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u_slice_next);
      }

      h(i+1,i) = new_norm;

      // Happy Breakdown Protection
      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (h(i+1,i) > breakdown_tol) {
        omp_la::omp_mul_v(dof, nNo, 1.0/h(i+1,i), u_slice_next);
      } else {
        h(i+1,i) = 0.0;
        breakdown = true;
      }

      for (int j = 0; j <= i-1; j++) {
        double tmp_h = c(j)*h(j,i) + s(j)*h(j+1,i);
        h(j+1,i) = -s(j)*h(j,i) + c(j)*h(j+1,i);
        h(j,i) = tmp_h;
      }

      double tmp_hypot = std::hypot(h(i,i), h(i+1,i));
      if (tmp_hypot == 0.0) {
        c(i) = 1.0; s(i) = 0.0;
      } else {
        c(i) = h(i,i) / tmp_hypot; s(i) = h(i+1,i) / tmp_hypot;
      }

      h(i,i) = tmp_hypot;
      h(i+1,i) = 0.0;

      err(i+1) = -s(i)*err(i);
      err(i) = c(i)*err(i);

      if (std::abs(err(i+1)) < eps || breakdown) {
        ls.suc = true;
        break;
      }
    }

    if (last_i >= ls.sD) last_i = ls.sD - 1;

    for (int i = 0; i <= last_i; i++) y(i) = err(i);

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) y(j) -= h(j,k)*y(k);
      y(j) /= h(j,j);
    }

    // Cache-Fused Reconstruction
    for (int j = 0; j <= last_i; ) {
      if (j + 1 <= last_i) {
        omp_la::omp_axpbypgz_v(dof, nNo, X, X, y(j), u.rslice(j), y(j+1), u.rslice(j+1));
        j += 2;
      } else {
        omp_la::omp_sum_v(dof, nNo, y(j), X, u.rslice(j));
        j++;
      }
    }

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) break;
  }

  R = X;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB  = 10.0 * std::log(ls.fNorm / ls.dB);
}

}; // namespace gmres

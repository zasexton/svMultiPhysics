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

#include "bicgs.h"

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
#include <algorithm>

namespace bicgs {

/// @brief Biconjugate-gradient stabilized algorithm for vector systems.
void bicgsv(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
    const Array<double>& K, Array<double>& R)
{
  #define n_debug_bicgsv
  #ifdef debug_bicgsv
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  using namespace fe_fsi_linear_solver;

  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;
  #ifdef debug_bicgsv
  dmsg << "ls.mItr: " << ls.mItr;
  dmsg << "dof: " << dof;
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  #endif

  ls.ws.ensure_bicgs_v(dof, nNo);
  auto& P = ls.ws.bicgs_P;
  auto& Rh = ls.ws.bicgs_Rh;
  auto& X = ls.ws.bicgs_X;
  auto& V = ls.ws.bicgs_V;
  auto& S = ls.ws.bicgs_S;
  auto& T = ls.ws.bicgs_T;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;
  double err = norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
  double errO = err;
  ls.iNorm = err;
  double eps = std::max(ls.absTol,ls.relTol*err);
  double rho = err*err;
  double beta = rho;
  X = 0.0;
  P = R;
  Rh = R;
  int i_itr = 1;
  #ifdef debug_bicgsv
  dmsg;
  dmsg << "err: " << err;
  dmsg << "eps: " << eps;
  #endif

  for (int i = 0; i < ls.mItr; i++) {
    #ifdef debug_bicgsv
    dmsg;
    dmsg << "----- i " << i+1 << " -----";
    dmsg << "err: " << err;
    dmsg << "eps: " << eps;
    #endif
    if (err < eps) {
      ls.suc = true;
      break;
    }

    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, K, P, V);
    double denom_alpha = dot::fsils_dot_v(dof, mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    double alpha = rho / denom_alpha;
    omp_la::omp_axpby_v(dof, nNo, S, R, -alpha, V);

    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, K, S, T);
    double t_sq = dot::fsils_dot_v(dof, mynNo, lhs.commu, T, T);
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) break;
    double omega = dot::fsils_dot_v(dof, mynNo, lhs.commu, T, S) / t_sq;

    omp_la::omp_axpbypgz_v(dof, nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_v(dof, nNo, R, S, -omega, T);

    errO = err;
    err =  norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
    double rhoO  = rho;
    rho = dot::fsils_dot_v(dof, mynNo, lhs.commu, R, Rh);
    double denom_beta = rhoO * omega;
    if (std::abs(denom_beta) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    beta = rho*alpha / denom_beta;

    #ifdef debug_bicgsv
    dmsg << "alpha: " << alpha;
    dmsg << "omega: " << omega;
    dmsg << "rho: " << rho;
    dmsg << "beta: " << beta;
    #endif

    // P = R + beta*(P - omega*V)
    omp_la::omp_sum_v(dof, nNo, -omega, P, V);
    omp_la::omp_axpby_v(dof, nNo, P, R, beta, P);
    i_itr += 1;
  }

  R = X;
  ls.itr = i_itr - 1;
  ls.fNorm = err;
  ls.callD =  fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  #ifdef debug_bicgsv
  dmsg << "ls.itr: " << ls.itr;
  #endif

  if (errO < std::numeric_limits<double>::epsilon()) {
     ls.dB = 0.0;
  } else {
     ls.dB = 10.0 * std::log(err / errO);
  }

}

//--------
// bicgss
//--------
//
void bicgss(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const Vector<double>& K, Vector<double>& R)
{
  #define n_debug_bicgss
  #ifdef debug_bicgss
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  using namespace fe_fsi_linear_solver;

  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;
  #ifdef debug_bicgss
  dmsg << "ls.mItr: " << ls.mItr;
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  #endif

  ls.ws.ensure_bicgs_s(nNo);
  auto& P = ls.ws.bicgs_Ps;
  auto& Rh = ls.ws.bicgs_Rhs;
  auto& X = ls.ws.bicgs_Xs;
  auto& V = ls.ws.bicgs_Vs;
  auto& S = ls.ws.bicgs_Ss;
  auto& T = ls.ws.bicgs_Ts;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;
  double err = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  double errO = err;
  ls.iNorm = err;
  double eps = std::max(ls.absTol,ls.relTol*err);
  double rho = err*err;
  double beta = rho;
  X = 0.0;
  P = R;
  Rh = R;
  int i_itr = 1;
  #ifdef debug_bicgss
  dmsg;
  dmsg << "err: " << err;
  dmsg << "eps: " << eps;
  #endif

  for (int i = 0; i < ls.mItr; i++) {
    #ifdef debug_bicgss
    dmsg;
    dmsg << "----- i " << i+1 << " -----";
    dmsg << "err: " << err;
    dmsg << "eps: " << eps;
    #endif
    if (err < eps) {
      ls.suc = true;
      break;
    }

    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, K, P, V);
    double denom_alpha = dot::fsils_dot_s(mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    double alpha = rho / denom_alpha;
    omp_la::omp_axpby_s(nNo, S, R, -alpha, V);

    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, K, S, T);
    double t_sq = dot::fsils_dot_s(mynNo, lhs.commu, T, T);
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) break;
    double omega = dot::fsils_dot_s(mynNo, lhs.commu, T, S) / t_sq;

    omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, omega, S);
    omp_la::omp_axpby_s(nNo, R, S, -omega, T);

    errO = err;
    err =  norm::fsi_ls_norms(mynNo, lhs.commu, R);
    double rhoO  = rho;
    rho = dot::fsils_dot_s(mynNo, lhs.commu, R, Rh);
    double denom_beta = rhoO * omega;
    if (std::abs(denom_beta) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    beta = rho*alpha / denom_beta;

    #ifdef debug_bicgss
    dmsg << "alpha: " << alpha;
    dmsg << "omega: " << omega;
    dmsg << "rho: " << rho;
    dmsg << "beta: " << beta;
    #endif

    // P = R + beta*(P - omega*V)
    omp_la::omp_sum_s(nNo, -omega, P, V);
    omp_la::omp_axpby_s(nNo, P, R, beta, P);
    i_itr += 1;
  }

  R = X;
  ls.itr = i_itr - 1;
  ls.fNorm = err;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  #ifdef debug_bicgss
  dmsg << "ls.itr: " << ls.itr;
  #endif

  if (errO < std::numeric_limits<double>::epsilon()) {
     ls.dB = 0.0;
  } else {
     ls.dB = 10.0 * std::log(err / errO);
  }
}

//--------
// schur
//--------
/// @brief VMS-compatible BiCGStab Schur complement solver for (L - D*H*G) P = R.
/// Replaces the symmetric CG Schur solver. Handles exact non-symmetric D != -G^T
/// and utilizes an algebraic diagonal SIMPLEC scaling based on the PSPG L block.
void schur(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int nsd,
           const Array<double>& D, const Array<double>& G, const Vector<double>& L, Vector<double>& R)
{
  using namespace fe_fsi_linear_solver;
  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;

  ls.ws.ensure_bicgs_s(nNo);
  auto& P  = ls.ws.bicgs_Ps;
  auto& Rh = ls.ws.bicgs_Rhs;
  auto& X  = ls.ws.bicgs_Xs;
  auto& V  = ls.ws.bicgs_Vs;
  auto& S  = ls.ws.bicgs_Ss;
  auto& T  = ls.ws.bicgs_Ts;

  Array<double> GP(nsd, nNo);
  Vector<double> SP(nNo), DGP(nNo);

  // 1. SIMPLEC Preconditioner: M_inv = 1 / diag(L)
  Vector<double> M_inv(nNo);
  for (fsils_int i = 0; i < nNo; ++i) {
    double diag_val = L(lhs.diagPtr(i));
    // Protect against division by zero for standard Galerkin nodes where L_ii = 0
    M_inv(i) = (std::abs(diag_val) > 1e-12) ? 1.0 / diag_val : 1.0;
  }

  // 2. Precondition the initial right-hand side: R = M_inv * R
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; i++) {
    R(i) = R(i) * M_inv(i);
  }

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;

  double err = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  double errO = err;
  ls.iNorm = err;
  double eps = std::max(ls.absTol, ls.relTol * err);
  double rho = err * err;
  double beta = rho;

  X = 0.0;
  P = R;
  Rh = R;
  int i_itr = 1;

  // 3. Define the preconditioned Schur Operator Q = M_inv * (L - D*H*G) * in_vec
  auto apply_schur_operator = [&](const Vector<double>& in_vec, Vector<double>& out_vec) {
    // GP = G * in_vec
    spar_mul::fsils_spar_mul_sv(lhs, lhs.rowPtr, lhs.colPtr, nsd, G, in_vec, GP);

    // Apply Boundary Sherman-Morrison Preconditioner H
    for (auto& face : lhs.face) {
      if (face.coupledFlag) {
        Array<double> unCondU = GP;
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, nsd, unCondU, GP);
        break;
      }
    }

    // DGP = D * GP (using exact analytical D, NOT G^T)
    spar_mul::fsils_spar_mul_vs(lhs, lhs.rowPtr, lhs.colPtr, nsd, D, GP, DGP);

    // SP = L * in_vec (PSPG Laplacian)
    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, L, in_vec, SP);

    // out_vec = M_inv * (SP - DGP)
    #pragma omp parallel for schedule(static)
    for (fsils_int i = 0; i < nNo; i++) {
      out_vec(i) = M_inv(i) * (SP(i) - DGP(i));
    }
  };

  // 4. Left-Preconditioned BiCGStab Loop
  for (int i = 0; i < ls.mItr; i++) {
    if (err < eps) {
      ls.suc = true;
      break;
    }

    apply_schur_operator(P, V);

    double denom_alpha = dot::fsils_dot_s(mynNo, lhs.commu, Rh, V);
    if (std::abs(denom_alpha) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    double alpha = rho / denom_alpha;

    // S = R - alpha * V
    omp_la::omp_axpby_s(nNo, S, R, -alpha, V);

    apply_schur_operator(S, T);

    double t_sq = dot::fsils_dot_s(mynNo, lhs.commu, T, T);
    if (std::sqrt(t_sq) < std::numeric_limits<double>::epsilon() * ls.iNorm) break;
    double omega = dot::fsils_dot_s(mynNo, lhs.commu, T, S) / t_sq;

    // X = X + alpha * P + omega * S
    omp_la::omp_axpbypgz_s(nNo, X, X, alpha, P, omega, S);

    // R = S - omega * T
    omp_la::omp_axpby_s(nNo, R, S, -omega, T);

    errO = err;
    err = norm::fsi_ls_norms(mynNo, lhs.commu, R);

    double rhoO = rho;
    rho = dot::fsils_dot_s(mynNo, lhs.commu, R, Rh);

    double denom_beta = rhoO * omega;
    if (std::abs(denom_beta) < std::numeric_limits<double>::epsilon() * (std::abs(rho) + std::numeric_limits<double>::epsilon())) break;
    beta = (rho * alpha) / denom_beta;

    // P = R + beta * (P - omega * V)
    omp_la::omp_sum_s(nNo, -omega, P, V);
    omp_la::omp_axpby_s(nNo, P, R, beta, P);

    i_itr += 1;
  }

  R = X;
  ls.itr = i_itr - 1;
  ls.fNorm = err;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;

  if (errO < std::numeric_limits<double>::epsilon()) {
    ls.dB = 0.0;
  } else {
    ls.dB = 10.0 * std::log(err / errO);
  }
}

} // namespace bicgs

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

#include "cgrad.h"
#include "DebugMsg.h"

#include "fsils_api.hpp"
#include "add_bc_mul.h"
#include "distributed_mpi_ops.h"
#include "distributed_sparse_operator.h"
#include "dot.h"
#include "omp_la.h"
#include "norm.h"

#include <math.h>
#include <limits>

namespace cgrad {

namespace dso = fe_fsi_linear_solver::distributed_sparse_operator;

/// @brief Conjugate-gradient algorithm for scaler, vector and Schur
/// complement cases.
///
/// Reproduces 'SUBROUTINE CGRAD_SCHUR(lhs, ls, dof, D, G, L, R)'
//
void schur(const fe_fsi_linear_solver::distributed_solver_bundles::ScalarConstraintSchurSystem& system,
    FSILS_subLsType& ls, Vector<double>& R)
{
  auto& lhs = *system.lhs;
  const int dof = system.momentum_components;
  const auto& D = system.D;
  const auto& G = system.G;
  const auto& L = system.L;
  #define n_debug_schur
  #ifdef debug_schur
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;
  const fe_fsi_linear_solver::HaloExchange halo(lhs);

  Vector<double> X(nNo), P(nNo), SP(nNo), DGP(nNo); 
  Array<double> GP(dof,nNo), unCondU(dof,nNo);

  double time = fe_fsi_linear_solver::fsils_cpu_t();
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;
  ls.suc = false;
  ls.iNorm = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  double eps = pow(std::max(ls.absTol,ls.relTol*ls.iNorm),2.0);
  double errO = ls.iNorm*ls.iNorm;
  double err = errO;
  #ifdef debug_schur
  dmsg << "dof: " << dof;
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  dmsg << "ls.iNorm: " << ls.iNorm;
  dmsg << "eps: " << eps;
  dmsg << "errO: " << errO;
  #endif

  X = 0.0;
  P = R;
  int last_i = 0;

  for (int i = 0; i < ls.mItr; i++) {
    #ifdef debug_schur
    dmsg;
    dmsg << "----- i " << i+1 << " -----";
    dmsg << "err: " << err;
    auto istr = "_" + std::to_string(i+1);
    #endif
    last_i = i;

    if (err < eps) {
      ls.suc = true;
      break;
    }

    errO = err;

    halo.sync_owned_to_ghost_scalar(P);

    // GP = G * P
    G.apply(
        dso::ghost_synced_input(P),
        dso::owned_only_output(dof, GP));

    for (auto& face : lhs.face) {
      if (face.coupledFlag) {
        auto unCondU = GP;
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU, GP);
        break;
      }
    }

    halo.sync_owned_to_ghost_vector(dof, GP);

    // DGP = K * GP
    D.apply(
        dso::ghost_synced_input(dof, GP),
        dso::owned_only_output(DGP));

    // SP = L * P
    L.apply(
        dso::ghost_synced_input(P),
        dso::owned_only_output(SP));

    // SP = SP - DGP
    omp_la::omp_sum_s(nNo, -1.0, SP, DGP);

    double alpha = errO / dot::fsils_dot_s(mynNo, lhs.commu, P, SP);

    // X = X + alpha * P
    omp_la::omp_sum_s(nNo, alpha, X, P);

    // R = R - alpha * SP
    omp_la::omp_sum_s(nNo, -alpha, R, SP);

    err = norm::fsi_ls_norms(mynNo, lhs.commu, R);
    err = err * err;
    #ifdef debug_schur
    dmsg << "err: " << err;
    dmsg << "errO/err: " << errO/err;
    dmsg << "err/errO: " << err/errO;
    #endif

    // Standard CG update: beta = err/errO, P = R + beta*P
    if (std::abs(errO) <= std::numeric_limits<double>::epsilon()) {
      break;
    }
    double beta = err / errO;
    omp_la::omp_mul_s(nNo, beta, P);
    omp_la::omp_sum_s(nNo, 1.0, P, R);
  }

  R = X;
  ls.fNorm = sqrt(err);
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - time + ls.callD;
  ls.itr = ls.itr + last_i;
  #ifdef debug_schur
  dmsg << "errO: " << errO;
  dmsg << "ls.fNorm: " << ls.fNorm;
  dmsg << "ls.itr: " << ls.itr;
  #endif

  if (errO < std::numeric_limits<double>::epsilon()) {
    ls.dB = 0.0;
  } else {
    ls.dB = 5.0 * log(err/errO);
  }
  ls.stats.record_call(ls.itr - itr_before,
                       /*restart_cycles=*/1,
                       fe_fsi_linear_solver::fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       /*setup_seconds=*/0.0,
                       ls.callD - callD_before);
}

void schur(FSILS_lhsType& lhs, FSILS_subLsType& ls, const int dof, const Array<double>& D,
    const Array<double>& G, const Vector<double>& L, Vector<double>& R)
{
  schur(fe_fsi_linear_solver::distributed_solver_bundles::make_scalar_constraint_schur_system(
            lhs, dof, D, G, L),
        ls, R);
}

//---------
// cgrad_v
//---------
//
void cgrad_v(const fe_fsi_linear_solver::distributed_solver_bundles::VectorLinearSystem& system,
    FSILS_subLsType& ls, Array<double>& R)
{
  #define n_debug_cgrad_v 
  #ifdef debug_cgrad_v
  DebugMsg dmsg(__func__,  system.lhs->commu.task);
  dmsg.banner();
  double time = fe_fsi_linear_solver::fsils_cpu_t();
  #endif

  auto& lhs = *system.lhs;
  const int dof = system.components;
  const auto& A = system.A;
  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;
  const fe_fsi_linear_solver::HaloExchange halo(lhs);
  #ifdef debug_cgrad_v
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  dmsg << "ls.mItr: " << ls.mItr;
  #endif

  ls.ws.ensure_cg_v(dof, nNo);
  auto& P = ls.ws.cg_P;
  auto& KP = ls.ws.cg_KP;
  auto& X = ls.ws.cg_X;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;
  ls.suc = false;
  ls.iNorm = norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
  double eps = pow(std::max(ls.absTol, ls.relTol* ls.iNorm), 2.0);

  double errO = ls.iNorm * ls.iNorm;
  double err  = errO;
  X = 0.0;
  P = R;
  int last_i = 0;
  #ifdef debug_cgrad_v
  dmsg << "ls.iNorm: " << ls.iNorm;
  dmsg << "eps: " << eps;
  dmsg << "err: " << eps;
  #endif

  for (int i = 0; i < ls.mItr; i++) {
    #ifdef debug_cgrad_v
    dmsg;
    dmsg << "----- i " << i+1 << " -----";
    dmsg << "err: " << err;
    auto istr = "_" + std::to_string(i+1);
    #endif
    last_i = i;

    if (err < eps) {
      ls.suc = true;
      break;
    }

    errO = err;

    halo.sync_owned_to_ghost_vector(dof, P);

    A.apply(
        dso::ghost_synced_input(dof, P),
        dso::owned_only_output(dof, KP));

    double alpha = errO / dot::fsils_dot_v(dof, mynNo, lhs.commu, P, KP);
    omp_la::omp_sum_v(dof, nNo, alpha, X, P);
    omp_la::omp_sum_v(dof, nNo, -alpha, R, KP);

    err = norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
    err = err * err;

    // Standard CG update: beta = err/errO, P = R + beta*P
    if (std::abs(errO) <= std::numeric_limits<double>::epsilon()) {
      break;
    }
    double beta = err / errO;
    omp_la::omp_mul_v(dof, nNo, beta, P);
    omp_la::omp_sum_v(dof, nNo, 1.0, P, R);
  }

  R = X;
  ls.itr = last_i;
  ls.fNorm = sqrt(err);
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;

  if (errO < std::numeric_limits<double>::epsilon()) {
    ls.dB = 0.0;
  } else {
    ls.dB = 5.0 * log(err/errO);
  }
  ls.stats.record_call(ls.itr - itr_before,
                       /*restart_cycles=*/1,
                       fe_fsi_linear_solver::fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       /*setup_seconds=*/0.0,
                       ls.callD - callD_before);

  #ifdef debug_cgrad_v
  double exec_time = fe_fsi_linear_solver::fsils_cpu_t() - time;
  dmsg << "Execution time: " << exec_time;
  dmsg << "Done";
  #endif
}

void cgrad_v(FSILS_lhsType& lhs, FSILS_subLsType& ls, const int dof, const Array<double>& K, Array<double>& R)
{
  cgrad_v(fe_fsi_linear_solver::distributed_solver_bundles::make_vector_linear_system(lhs, dof, K), ls, R);
}

//---------
// cgrad_s
//---------
//
void cgrad_s(const fe_fsi_linear_solver::distributed_solver_bundles::ScalarLinearSystem& system,
    FSILS_subLsType& ls, Vector<double>& R)
{
  #define n_debug_cgrad_s 
  #ifdef debug_cgrad_s
  DebugMsg dmsg(__func__,  system.lhs->commu.task);
  dmsg.banner();
  double time = fe_fsi_linear_solver::fsils_cpu_t();
  #endif

  auto& lhs = *system.lhs;
  const auto& A = system.A;
  fsils_int nNo = lhs.nNo;
  fsils_int mynNo = lhs.mynNo;
  const fe_fsi_linear_solver::HaloExchange halo(lhs);
  #ifdef debug_cgrad_s
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  dmsg << "ls.mItr: " << ls.mItr;
  #endif

  ls.ws.ensure_cg_s(nNo);
  auto& P = ls.ws.cg_Ps;
  auto& KP = ls.ws.cg_KPs;
  auto& X = ls.ws.cg_Xs;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  const int itr_before = ls.itr;
  const double callD_before = ls.callD;
  const auto collective_before = lhs.commu.collective_stats;
  ls.suc = false;
  ls.iNorm = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  double eps = pow(std::max(ls.absTol, ls.relTol* ls.iNorm), 2.0);
  double errO = ls.iNorm * ls.iNorm;
  double err  = errO;
  X = 0.0;
  P = R;
  int last_i = 0;
  #ifdef debug_cgrad_s
  dmsg << "ls.iNorm: " << ls.iNorm;
  dmsg << "eps: " << eps;
  dmsg << "err: " << eps;
  #endif

  for (int i = 0; i < ls.mItr; i++) {
    #ifdef debug_cgrad_s
    dmsg;
    dmsg << "----- i " << i+1 << " -----";
    dmsg << "err: " << err;
    auto istr = "_" + std::to_string(i+1);
    #endif
    last_i = i;

    if (err < eps) {
      ls.suc = true;
      break;
    }

    errO = err;

    halo.sync_owned_to_ghost_scalar(P);

    A.apply(
        dso::ghost_synced_input(P),
        dso::owned_only_output(KP));

    double alpha = errO / dot::fsils_dot_s(mynNo, lhs.commu, P, KP);
    omp_la::omp_sum_s(nNo, alpha, X, P);
    omp_la::omp_sum_s(nNo, -alpha, R, KP);

    err = norm::fsi_ls_norms(mynNo, lhs.commu, R);
    err = err * err;

    // Standard CG update: beta = err/errO, P = R + beta*P
    if (std::abs(errO) <= std::numeric_limits<double>::epsilon()) {
      break;
    }
    double beta = err / errO;
    omp_la::omp_mul_s(nNo, beta, P);
    omp_la::omp_sum_s(nNo, 1.0, P, R);
  }

  R = X;
  ls.itr = last_i;
  ls.fNorm = sqrt(err);
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;

  if (errO < std::numeric_limits<double>::epsilon()) {
    ls.dB = 0.0;
  } else {
    ls.dB = 5.0 * log(err/errO);
  }
  ls.stats.record_call(ls.itr - itr_before,
                       /*restart_cycles=*/1,
                       fe_fsi_linear_solver::fsils_collective_delta(collective_before, lhs.commu.collective_stats),
                       /*setup_seconds=*/0.0,
                       ls.callD - callD_before);

  #ifdef debug_cgrad_s
  double exec_time = fe_fsi_linear_solver::fsils_cpu_t() - time;
  dmsg << "Execution time: " << exec_time;
  dmsg << "Done";
  #endif
}

void cgrad_s(FSILS_lhsType& lhs, FSILS_subLsType& ls, const Vector<double>& K, Vector<double>& R)
{
  cgrad_s(fe_fsi_linear_solver::distributed_solver_bundles::make_scalar_linear_system(lhs, K), ls, R);
}

};

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

// Fractional-step solver for incompressible NS / FSI equations.
// Form: AU=R, where A = [K G; D L] with exact analytical D (not -G^T).
// Uses asymmetric BiCGStab Schur complement instead of symmetric CG.

#include "ns_solver.h"

#include "fsils_api.hpp"
#include "fils_struct.hpp"

#include "add_bc_mul.h"
#include "bicgs.h"
#include "dot.h"
#include "ge.h"
#include "gmres.h"
#include "norm.h"
#include "spar_mul.h"

#include "Array3.h"

#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace ns_solver {

using fe_fsi_linear_solver::fsils_int;

namespace {

[[nodiscard]] bool fsilsTraceEnabled() noexcept
{
  const char* env = std::getenv("SVMP_FSILS_TRACE");
  if (env == nullptr) {
    env = std::getenv("SVMP_FSILS_NS_TRACE");
  }
  if (env == nullptr) {
    env = std::getenv("SVMP_FSILS_NS_SOLVER_TRACE");
  }
  if (env == nullptr) {
    return false;
  }
  while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
    ++env;
  }
  if (*env == '\0') {
    return false;
  }
  return *env != '0';
}

} // namespace

/// @brief Modifies: lhs.face[].nS
/// VMS FIX: Avoided uninitialized memory by locally accumulating the norm
/// before MPI_Allreduce, instead of scattering into a full-size array.
void bc_pre(fe_fsi_linear_solver::FSILS_lhsType& lhs, const int nsd, const int dof, const fsils_int nNo, const fsils_int mynNo)
{
  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];

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

/// @brief Store sections of the 'Val' into separate arrays: 'mK', 'mG', 'mD', 'mL'
/// VMS FIX: Removed the O(N * dof^2) transposition to Gt. Exact analytical D is preserved.
///
/// Modifies: mK, mG, mD, and mL.
void depart(fe_fsi_linear_solver::FSILS_lhsType& lhs, const int nsd, const int dof, const fsils_int nNo, const fsils_int nnz,
    const Array<double>& Val, Array<double>& mK, Array<double>& mG, Array<double>& mD, Vector<double>& mL)
{
  const int stride = nsd + 1;

  #pragma omp parallel for schedule(static)
  for (fsils_int nz = 0; nz < nnz; nz++) {
    for (int i = 0; i < nsd; i++) {
      for (int j = 0; j < nsd; j++) {
        mK(i*nsd + j, nz) = Val(i*stride + j, nz);
      }
      mG(i, nz) = Val(i*stride + nsd, nz);
      mD(i, nz) = Val(nsd*stride + i, nz);
    }
    mL(nz) = Val(nsd*stride + nsd, nz);
  }
}

/// @brief Fractional-step solver utilizing an exact asymmetric Schur complement.
///
/// Ri (dof, lhs.nNo)
void ns_solver(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_lsType& ls, const int dof, const Array<double>& Val, Array<double>& Ri)
{
  using namespace consts;
  using namespace fe_fsi_linear_solver;

  #define n_debug_ns_solver
  #ifdef debug_ns_solver
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  double time = fe_fsi_linear_solver::fsils_cpu_t();
  #endif

  const fsils_int nNo = lhs.nNo;
  const fsils_int nnz = lhs.nnz;
  const fsils_int mynNo = lhs.mynNo;
  const int nsd = dof - 1;
  const int iB = ls.RI.mItr;
  const int nB = 2*iB;
  constexpr fsils_int BLOCK_SIZE = 256;

  #ifdef debug_ns_solver
  dmsg << "dof: " << dof;
  dmsg << "nsd: " << nsd;
  dmsg << "nNo: " << nNo;
  dmsg << "nnz: " << nnz;
  dmsg << "mynNo: " << mynNo;
  dmsg << "iB: " << iB;
  dmsg << "nB: " << nB;
  Ri.write(msg_prefix+"Ri");
  #endif

  Vector<double> Rc(nNo), Rci(nNo), tmp(nB*nB+nB), tmpG(nB*nB+nB), B(nB), xB(nB), oldxB(nB);
  oldxB = 0.0;
  Array<double> Rm(nsd,nNo), Rmi(nsd,nNo), A(nB,nB), P(nNo,iB), MP(nNo,nB);
  Array3<double> U(nsd,nNo,iB), MU(nsd,nNo,nB);

  #pragma omp parallel for schedule(static)
  for (fsils_int j = 0; j < nNo; j++) {
    for (int i = 0; i < nsd; i++) {
      Rmi(i,j) = Ri(i,j);
    }
    Rci(j) = Ri(dof-1,j);
  }

  Rm = Rmi;
  Rc = Rci;

  double eps = std::sqrt(std::pow(norm::fsi_ls_normv(nsd,mynNo,lhs.commu,Rm),2.0) +
                         std::pow(norm::fsi_ls_norms(mynNo,lhs.commu,Rc),2.0));

  #ifdef debug_ns_solver
  dmsg << "eps (Rm/Rc): " << eps;
  #endif

  ls.RI.iNorm = eps;
  ls.RI.fNorm = eps*eps;

  if (lhs.commu.masF && fsilsTraceEnabled()) {
    fprintf(stderr, "[NS_SOLVER] eps(initial)=%e iNorm=%e fNorm=%e nsd=%d dof=%d nNo=%lld nnz=%lld\n",
            eps, ls.RI.iNorm, ls.RI.fNorm, nsd, dof, (long long)nNo, (long long)nnz);
    fprintf(stderr, "[NS_SOLVER] relTol=%e absTol=%e mItr=%d\n",
            ls.RI.relTol, ls.RI.absTol, ls.RI.mItr);
  }

  // Calling duration
  ls.CG.callD = 0.0;
  ls.GM.callD = 0.0;
  ls.RI.callD = fe_fsi_linear_solver::fsils_cpu_t();

  ls.CG.itr   = 0;
  ls.GM.itr   = 0;
  ls.RI.suc   = false;
  eps = std::max(ls.RI.absTol, ls.RI.relTol*eps);
  #ifdef debug_ns_solver
  dmsg << "eps: " << eps;
  dmsg << "ls.RI.iNorm: " << ls.RI.iNorm;
  dmsg << "ls.RI.fNorm: " << ls.RI.fNorm;
  #endif

  // VMS FIX: Gt block mapping completely removed; exact analytical D is preserved.
  Array<double> mK(nsd*nsd,nnz), mG(nsd,nnz), mD(nsd,nnz);
  Vector<double> mL(nnz);

  depart(lhs, nsd, dof, nNo, nnz, Val, mK, mG, mD, mL);

  // Computes lhs.face[].nS for each face.
  bc_pre(lhs, nsd, dof, nNo, mynNo);

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];
    #ifdef debug_ns_solver
    dmsg << "faIn: " << faIn << "  face.nS: " << face.nS;
    #endif
    if (lhs.commu.masF && fsilsTraceEnabled()) {
      fprintf(stderr, "[NS_SOLVER] face[%d]: nNo=%d nS=%e coupledFlag=%d incFlag=%d sharedFlag=%d res=%e bGrp=%d\n",
              faIn, face.nNo, face.nS, face.coupledFlag?1:0, face.incFlag?1:0, face.sharedFlag?1:0, face.res, face.bGrp);
    }
   }

  #ifdef debug_ns_solver
  dmsg << "Loop i on ls.RI.mItr ... ";
  #endif
  int iBB{0};
  int i_count{0};

  for (int i = 0; i < ls.RI.mItr; i++) {
    #ifdef debug_ns_solver
    auto istr = "_" + std::to_string(i+1);
    dmsg << "---------- i " << i+1 << " ----------";
    #endif

    int iB = 2*i;
    iBB = 2*i + 1;
    ls.RI.dB = ls.RI.fNorm;
    i_count = i;
    #ifdef debug_ns_solver
    dmsg << "iB: " << iB;
    dmsg << "iBB: " << iBB;
    dmsg << "ls.RI.fNorm: " << ls.RI.fNorm;
    #endif

    // Solve for U = inv(mK) * Rm
    auto U_slice = U.rslice(i);
    gmres::gmres(lhs, ls.GM, nsd, mK, Rm, U_slice);

    // P = D*U (using exact analytical mD)
    auto P_col = P.rcol(i);
    spar_mul::fsils_spar_mul_vs(lhs, lhs.rowPtr, lhs.colPtr, nsd, mD, U_slice, P_col);

    // P = Rc - P
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      P_col(n) = Rc(n) - P_col(n);
    }

    // P = [L - D*H*G]^-1 * P
    // VMS FIX: Solved using asymmetric BiCGStab instead of symmetric CGRAD
    bicgs::schur(lhs, ls.CG, nsd, mD, mG, mL, P_col);

    // MU1 = G*P
    #ifdef debug_ns_solver
    dmsg << "i: " << i+1;
    dmsg << "iB: " << iB+1;
    #endif
    P_col = P.rcol(i);
    auto MU_iB = MU.rslice(iB);
    spar_mul::fsils_spar_mul_sv(lhs, lhs.rowPtr, lhs.colPtr, nsd, mG, P_col, MU_iB);

    // MU2 = Rm - G*P
    auto MU_iBB = MU.rslice(iBB);
    #pragma omp parallel for schedule(static)
    for (fsils_int n = 0; n < nNo; n++) {
      for (int d = 0; d < nsd; d++) {
        MU_iBB(d,n) = Rm(d,n) - MU_iB(d,n);
      }
    }

    // U = inv(K) * [Rm - G*P]
    gmres::gmres(lhs, ls.GM, nsd, mK, MU_iBB, U_slice);

    // MU2 = K*U
    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, nsd, mK, U_slice, MU_iBB);

    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, nsd, U_slice, MU_iBB);

    // MP1 = L*P
    auto MP_iB = MP.rcol(iB);
    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, mL, P.rcol(i), MP_iB);

    // MP2 = D*U
    auto MP_iBB = MP.rcol(iBB);
    spar_mul::fsils_spar_mul_vs(lhs, lhs.rowPtr, lhs.colPtr, nsd, mD, U_slice, MP_iBB);

    int c = 0;

    for (int k = iB; k <= iBB; k++) {
      auto MU_k = MU.rslice(k);
      auto MP_k = MP.rcol(k);

      for (int j = 0; j <= k; j++) {
        tmp(c) = dot::fsils_nc_dot_v(nsd, mynNo, MU.rslice(j), MU_k) +
                 dot::fsils_nc_dot_s(mynNo, MP.rcol(j), MP_k);
        c = c + 1;
      }

      tmp(c) = dot::fsils_nc_dot_v(nsd, mynNo, MU_k, Rmi) +
               dot::fsils_nc_dot_s(mynNo, MP_k, Rci);
      c = c + 1;
    }

    if (lhs.commu.nTasks > 1) {
      MPI_Allreduce(tmp.data(), tmpG.data(), c, cm_mod::mpreal, MPI_SUM, lhs.commu.comm);
      tmp = tmpG;
    }

    // Set arrays for Gauss elimination
    c = 0;

    for (int k = iB; k <= iBB; k++) {
      for (int j = 0; j <= k; j++) {
        A(j,k) = tmp(c);
        A(k,j) = tmp(c);
        c = c + 1;
      }

      B(k) = tmp(c);
      c  = c + 1;
    }

    xB = B;

    if (lhs.commu.masF && fsilsTraceEnabled()) {
      fprintf(stderr, "[NS_SOLVER] iter=%d Galerkin system (iBB+1=%d):\n", i_count, iBB+1);
      for (int kk = 0; kk <= iBB; kk++) {
        fprintf(stderr, "[NS_SOLVER]   A[%d,:] =", kk);
        for (int jj = 0; jj <= iBB; jj++) {
          fprintf(stderr, " %e", A(jj,kk));
        }
        fprintf(stderr, "  B[%d]=%e\n", kk, B(kk));
      }
    }

    // Minimize GCR outer residual
    if (ge::ge(nB, iBB+1, A, xB)) {
      oldxB = xB;

    } else {
      xB = oldxB;

      if (i > 0) {
        iB = iB - 2;
        iBB = iBB - 2;
        i_count = i - 1;
      }
      break;
    }

    double sum = 0.0;
    for (int j = 0; j <= iBB; j++) {
      sum += xB(j) * B(j);
    }
    ls.RI.fNorm = std::max(0.0, std::pow(ls.RI.iNorm,2.0) - sum);
    #ifdef debug_ns_solver
    dmsg << "sum: " << sum;
    dmsg << "ls.RI.fNorm: " << ls.RI.fNorm;
    #endif

    if (lhs.commu.masF && fsilsTraceEnabled()) {
      fprintf(stderr, "[NS_SOLVER] iter=%d iNorm=%e iNorm^2=%e sum=%e fNorm=%e\n",
              i_count, ls.RI.iNorm, std::pow(ls.RI.iNorm,2.0), sum, ls.RI.fNorm);
      for (int ii = 0; ii <= iBB; ii++) {
        fprintf(stderr, "[NS_SOLVER]   xB(%d)=%e B(%d)=%e product=%e\n",
                ii, xB(ii), ii, B(ii), xB(ii)*B(ii));
      }
      fprintf(stderr, "[NS_SOLVER]   GM.itr=%d GM.fNorm=%e CG.itr=%d CG.fNorm=%e\n",
              ls.GM.itr, ls.GM.fNorm, ls.CG.itr, ls.CG.fNorm);
    }

    if (ls.RI.fNorm < eps*eps) {
      ls.RI.suc = true;
      break;
    }

    // Cache-blocked residual update: (Rm,Rc) = (Rmi,Rci) - sum_j xB(j)*(MU_j,MP_j)
    #pragma omp parallel for schedule(static)
    for (fsils_int nBlock = 0; nBlock < nNo; nBlock += BLOCK_SIZE) {
      const fsils_int nEnd = std::min(nBlock + BLOCK_SIZE, nNo);

      for (int j = 0; j <= iBB; j++) {
        const double xb_j = xB(j);
        auto MU_j = MU.rslice(j);
        auto MP_j = MP.rcol(j);

        if (j == 0) {
          for (fsils_int n = nBlock; n < nEnd; n++) {
            for (int d = 0; d < nsd; d++) {
              Rm(d,n) = Rmi(d,n) - xb_j * MU_j(d,n);
            }
            Rc(n) = Rci(n) - xb_j * MP_j(n);
          }
        } else if (xb_j != 0.0) {
          for (fsils_int n = nBlock; n < nEnd; n++) {
            for (int d = 0; d < nsd; d++) {
              Rm(d,n) -= xb_j * MU_j(d,n);
            }
            Rc(n) -= xb_j * MP_j(n);
          }
        }
      }
    }

  } // for i = 0; i < ls.RI.mItr

  if (i_count >= ls.RI.mItr) {
    ls.RI.itr = ls.RI.mItr;
  } else {
    ls.RI.itr = i_count;

    // Cache-blocked pressure residual update: Rc = Rci - sum_j xB(j)*MP_j
    #pragma omp parallel for schedule(static)
    for (fsils_int nBlock = 0; nBlock < nNo; nBlock += BLOCK_SIZE) {
      const fsils_int nEnd = std::min(nBlock + BLOCK_SIZE, nNo);

      for (int j = 0; j <= iBB; j++) {
        const double xb_j = xB(j);
        auto MP_j = MP.rcol(j);

        if (j == 0) {
          for (fsils_int n = nBlock; n < nEnd; n++) {
            Rc(n) = Rci(n) - xb_j * MP_j(n);
          }
        } else if (xb_j != 0.0) {
          for (fsils_int n = nBlock; n < nEnd; n++) {
            Rc(n) -= xb_j * MP_j(n);
          }
        }
      }
    }
  }

  ls.Resc = (ls.RI.fNorm > 0.0)
      ? static_cast<int>(100.0 * std::pow(norm::fsi_ls_norms(mynNo, lhs.commu, Rc),2.0) / ls.RI.fNorm)
      : 0;
  ls.Resm = 100 - ls.Resc;

  #ifdef debug_ns_solver
  dmsg << "ls.Resc: " << ls.Resc;
  dmsg << "ls.Resm: " << ls.Resm;
  dmsg << "ls.RI.itr: " << ls.RI.itr;
  #endif

  // Cache-blocked solution reconstruction.
  #pragma omp parallel for schedule(static)
  for (fsils_int nBlock = 0; nBlock < nNo; nBlock += BLOCK_SIZE) {
    const fsils_int nEnd = std::min(nBlock + BLOCK_SIZE, nNo);

    for (int i = 0; i <= ls.RI.itr; i++) {
      auto U_i = U.rslice(i);
      auto P_i = P.rcol(i);

      if (i == 0) {
        const double xb_1 = xB(1);
        const double xb_0 = xB(0);
        for (fsils_int n = nBlock; n < nEnd; n++) {
          for (int d = 0; d < nsd; d++) {
            Rmi(d,n) = xb_1 * U_i(d,n);
          }
          Rci(n) = xb_0 * P_i(n);
        }

      } else {
        const int iB = 2*i;
        const int iBB = 2*i + 1;
        const double xb_iBB = xB(iBB);
        const double xb_iB = xB(iB);
        for (fsils_int n = nBlock; n < nEnd; n++) {
          for (int d = 0; d < nsd; d++) {
            Rmi(d,n) += xb_iBB * U_i(d,n);
          }
          Rci(n) += xb_iB * P_i(n);
        }
      }
    }
  }

  // Set Calling duration.
  ls.RI.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.RI.callD;

  ls.RI.dB = 5.0 * std::log(ls.RI.fNorm / ls.RI.dB);

  if (ls.Resc < 0.0 || ls.Resm < 0.0) {
    ls.Resc = 0;
    ls.Resm = 0;
    ls.RI.dB = 0;
    ls.RI.fNorm = 0.0;
    ls.RI.suc = false;
  }

  ls.RI.fNorm = std::sqrt(ls.RI.fNorm);
  #ifdef debug_ns_solver
  dmsg << "ls.RI.callD: " << ls.RI.callD;
  dmsg << "ls.RI.dB: " << ls.RI.dB;
  dmsg << "ls.RI.fNorm: " << ls.RI.fNorm;
  #endif

  #pragma omp parallel for schedule(static)
  for (fsils_int j = 0; j < nNo; j++) {
    for (int i = 0; i < nsd; i++) {
      Ri(i,j) = Rmi(i,j);
    }
    Ri(dof-1,j) = Rci(j);
  }

  if (lhs.commu.masF) {
    //CALL LOGFILE
  }

  #ifdef debug_ns_solver
  double exec_time = fe_fsi_linear_solver::fsils_cpu_t() - time;
  dmsg << "Execution time: " << exec_time;
  dmsg << "Done";
  #endif
}

} // namespace ns_solver

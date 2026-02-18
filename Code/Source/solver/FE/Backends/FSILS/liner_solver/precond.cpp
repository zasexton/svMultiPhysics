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

// In this routine, the appropriate LS algorithm is called and
// the solution is returned.

#include "precond.h"

#include "fsils_api.hpp"

#include <math.h>

namespace precond {

/// @brief Post-multipling Val by W: Val = Val*W
///
/// Modifies: Val
//
void pos_mul(const Array<int>& rowPtr, const Vector<int>& colPtr, const int nNo, const int nnz, const int dof, Array<double>& Val, const Array<double>& W)
{
  switch (dof) {
    case 1: {
      for (int Ac = 0; Ac < nNo; Ac++) { 
        for (int i = rowPtr(0,Ac); i <= rowPtr(1,Ac); i++) {
          int a = colPtr(i);
          Val(0,i) = Val(0,i)*W(0,a);
        }
      }
    } break; 

    case 2: {
      for (int Ac = 0; Ac < nNo; Ac++) { 
        for (int i = rowPtr(0,Ac); i <= rowPtr(1,Ac); i++) {
          int a = colPtr(i);
          for (int j = 0; j < 3; j += 2) {
            Val(j+0,i) = Val(j+0,i)*W(0,a);
            Val(j+1,i) = Val(j+1,i)*W(1,a);
          }
        }
      }
    } break; 

    case 3: {
      for (int Ac = 0; Ac < nNo; Ac++) { 
        for (int i = rowPtr(0,Ac); i <= rowPtr(1,Ac); i++) {
          int a = colPtr(i);
          for (int j = 0; j < 7; j += 3) {
            Val(j+0,i) = Val(j+0,i)*W(0,a);
            Val(j+1,i) = Val(j+1,i)*W(1,a);
            Val(j+2,i) = Val(j+2,i)*W(2,a);
          }
        }
      }
    } break; 

    case 4: {
      for (int Ac = 0; Ac < nNo; Ac++) { 
        for (int i = rowPtr(0,Ac); i <= rowPtr(1,Ac); i++) {
          int a = colPtr(i);
          for (int j = 0; j < 13; j += 4) {
            Val(j+0,i) = Val(j+0,i)*W(0,a);
            Val(j+1,i) = Val(j+1,i)*W(1,a);
            Val(j+2,i) = Val(j+2,i)*W(2,a);
            Val(j+3,i) = Val(j+3,i)*W(3,a);
          }
        }
      }
    } break; 

    default: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        for (int i = rowPtr(0,Ac); i <= rowPtr(1,Ac); i++) {
          int a = colPtr(i);
          for (int b = 0; b < dof; b++) {
            int j = dof*(dof-1) + b;
            for (int k = b; k <= j; k += dof) {
              Val(k,i) = Val(k,i)*W(b,a);
            }
          }
        }
      }
    } break; 
  }
}

//--------------
// precond_diag 
//--------------
// Jacobi symmetic preconditioner, to precondition both LHS and RHS.
//
// Modifies: Val, R, W
//
// Reproduces Fortran 'PRECONDDIAG'.
//
void precond_diag(fsi_linear_solver::FSILS_lhsType& lhs, const Array<int>& rowPtr, const Vector<int>& colPtr, 
    const Vector<int>& diagPtr, const int dof, Array<double>& Val, Array<double>& R, Array<double>& W)
{
  #define n_debug_precond_diag
  #ifdef debug_precond_diag
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  int nNo = lhs.nNo;
  #ifdef debug_precond_diag
  dmsg << "lhs.nFaces: " << lhs.nFaces;
  dmsg << "nNo: " << nNo;
  dmsg << "dof: " << dof;
  dmsg << "Val.nrows: " << Val.nrows_;
  dmsg << "Val.ncols: " << Val.ncols_;
  dmsg << "W.nrows: " << W.nrows_;
  dmsg << "W.ncols: " << W.ncols_;
  #endif

  // Calculating W: W = diag(K)
  //
  switch (dof) {
    case 1: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        W(0,Ac) = Val(0,diagPtr(Ac));
      }
    } break;

    case 2: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        int d = diagPtr(Ac);
        W(0,Ac) = Val(0,d);
        W(1,Ac) = Val(3,d);
      }
    } break;

    case 3: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        int d = diagPtr(Ac);
        W(0,Ac) = Val(0,d);
        W(1,Ac) = Val(4,d);
        W(2,Ac) = Val(8,d);
      }
    } break;

    case 4: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        int d = diagPtr(Ac);
        W(0,Ac) = Val(0,d);
        W(1,Ac) = Val(5,d);
        W(2,Ac) = Val(10,d);
        W(3,Ac) = Val(15,d);
      }
    } break;

    default: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        int d = diagPtr(Ac);
        for (int i = 0; i < dof; i++) {
          W(i,Ac) = Val(i*dof+i,d);
        }
      }
    } break;
  }

  fsils_commuv(lhs, dof, W);

  // Accounting for Dirichlet BC and inversing W = W^{-1/2}
  //
  for (int Ac = 0; Ac < nNo; Ac++) {
    for (int i = 0; i < dof; i++) {
      if (W(i,Ac) == 0.0) {
        W(i,Ac) = 1.0;
      }
    }
  }

  for (int i = 0; i < W.size(); i++) {
    W(i) = 1.0 / sqrt(fabs(W(i)));
  }

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];
    #ifdef debug_precond_diag
    dmsg << ">>> faIn: " << faIn;
    dmsg << "face.incFlag: " << face.incFlag;
    #endif

    if (!face.incFlag) {
      continue;
    }

    int n = std::min(face.dof,dof);

    if (face.bGrp == fsi_linear_solver::BcType::BC_TYPE_Dir) {
      for (int a = 0; a < face.nNo; a++) {
        int Ac = face.glob(a);
        for (int i = 0; i < n; i++) {
          W(i,Ac) = W(i,Ac) * face.val(i,a);
        }
      }
    }
  }

  // Pre-multipling K with W: K = W*K
  pre_mul(rowPtr, lhs.nNo, lhs.nnz, dof, Val, W);

  // Multipling R with W: R = W*R
  //
  // W ( dof, lhs.nNo )
  //
  // R ( dof, lhs.nNo )
  //
  // ELement-wise multiplication.
  //
  for (int i = 0; i < W.size(); i++) {
    R(i) = W(i) * R(i);
  }

  // Now post-multipling K by W: K = K*W
  pos_mul(rowPtr, colPtr, lhs.nNo, lhs.nnz, dof, Val, W);

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];

    if (face.coupledFlag) {
      for (int a = 0; a < face.nNo; a++) {
        int Ac = face.glob(a);
        for (int i = 0; i < std::min(face.dof,dof); i++) {
          face.valM(i,a) = face.val(i,a) * W(i,Ac);
        }
      }
    }
  }
}

//-------------
// precond_rcs
//-------------
// Row and column preconditioner, to precondition both LHS and RHS.
//
// Reproduces Fortran 'PRECONDRCS'.
//
void precond_rcs(fsi_linear_solver::FSILS_lhsType& lhs, const Array<int>& rowPtr, const Vector<int>& colPtr,
    const Vector<int>& diagPtr, const int dof, Array<double>& Val, Array<double>& R, Array<double>& W1, Array<double>& W2)
{
  const int nNo = lhs.nNo;
  int maxiter = 10;
  double tol = 2.0;
  int iter = 0;
  bool flag = true;
  W1 = 1.0;
  W2 = 1.0;

  //*****************************************************
  // Apply Dirichlet BC
  //*****************************************************
  //
  Array<double> Wr(dof,nNo), Wc(dof,nNo);
  Wr = 1.0;

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];
    if (!face.incFlag) {
      continue;
    }

    int n = std::min(face.dof,dof);

    if (face.bGrp == fsi_linear_solver::BcType::BC_TYPE_Dir) {
      for (int a = 0; a < face.nNo; a++) {
        int Ac = face.glob(a);
        for (int i = 0; i < n; i++) {
          Wr(i,Ac) = Wr(i,Ac) * face.val(i,a);
        }
      }
    }
  }

  fsils_commuv(lhs, dof, Wr);

  // For parallel case, val and Wr can be larger than 1 due to
  // the addition operator in FSILS_COMMUV. Hence need renormalization.
  //
  for (int i = 0; i < Wr.size(); i++) {
    Wr(i) = (Wr(i) > 0.5) ? 1.0 : 0.0;
  }

  // Kill the row and column corresponding to Dirichlet BC
  //
  // Modifies 'Val'.
  //
  pre_mul(rowPtr, lhs.nNo, lhs.nnz, dof, Val, Wr);

  R = Wr * R;

  pos_mul(rowPtr, colPtr, lhs.nNo, lhs.nnz, dof, Val, Wr);

  // Set diagonal term to one
  //
  switch (dof) {
    case 1:
      for (int Ac = 0; Ac < nNo; Ac++) {
        int d = diagPtr(Ac);
        Val(0,d) = Wr(0,Ac) * (Val(0,d) - 1.0) + 1.0;
      }
    break; 

    case 2:
      for (int Ac = 0; Ac < nNo; Ac++) {
        int d = diagPtr(Ac);
        Val(0,d) = Wr(0,Ac)*(Val(0,d)-1.0) + 1.0;
        Val(3,d) = Wr(1,Ac)*(Val(3,d)-1.0) + 1.0;
      }
    break; 

    case 3:
      for (int Ac = 0; Ac < nNo; Ac++) {
        int d = diagPtr(Ac);
        Val(0,d) = Wr(0,Ac)*(Val(0,d)-1.0) + 1.0;
        Val(4,d) = Wr(1,Ac)*(Val(4,d)-1.0) + 1.0;
        Val(8,d) = Wr(2,Ac)*(Val(8,d)-1.0) + 1.0;
      }
    break; 

    case 4:
      for (int Ac = 0; Ac < nNo; Ac++) {
        int d = diagPtr(Ac);
        Val(0 ,d) = Wr(0,Ac)*(Val(0 ,d)-1.0) + 1.0;
        Val(5 ,d) = Wr(1,Ac)*(Val(5 ,d)-1.0) + 1.0;
        Val(10,d) = Wr(2,Ac)*(Val(10,d)-1.0) + 1.0;
        Val(15,d) = Wr(3,Ac)*(Val(15,d)-1.0) + 1.0;
      }
    break; 

    default: 
      for (int Ac = 0; Ac < nNo; Ac++) {
        int d = diagPtr(Ac);
        for (int i = 0; i < dof; i++) {
          Val(i*dof+i,d) = Wr(i,Ac)*(Val(i*dof+i,d) - 1.0) + 1.0;
        }
      }
    break; 
  } 

  //*****************************************************
  // Row and column scaling
  //*****************************************************
  //
  while (flag) {
    Wr = 0.0;
    Wc = 0.0;
    iter = iter + 1;

    if (iter >= maxiter) {
      std::cout << "[precond_rcs] Warning: maximum iteration number reached";
      flag = false; 
    }

    // Max norm along row and column
    //
    switch (dof) {
      case 1:
        for (int Ac = 0; Ac < nNo; Ac++) {
          double mx = 0.0;
          for (int j = rowPtr(0,Ac); j <= rowPtr(1,Ac); j++) {
            double av = fabs(Val(0,j));
            if (av > mx) mx = av;
            int a = colPtr(j);
            if (av > Wc(0,a)) Wc(0,a) = av;
          }
          Wr(0,Ac) = mx;
        }
      break;

      case 2:
        for (int Ac = 0; Ac < nNo; Ac++) {
          double mx0 = 0.0, mx1 = 0.0;
          for (int j = rowPtr(0,Ac); j <= rowPtr(1,Ac); j++) {
            // Row max: rows 0-1 for dof 0, rows 2-3 for dof 1
            double av;
            av = fabs(Val(0,j)); if (av > mx0) mx0 = av;
            av = fabs(Val(1,j)); if (av > mx0) mx0 = av;
            av = fabs(Val(2,j)); if (av > mx1) mx1 = av;
            av = fabs(Val(3,j)); if (av > mx1) mx1 = av;
            // Col max: columns 0,2 for dof 0; columns 1,3 for dof 1
            int a = colPtr(j);
            av = fabs(Val(0,j)); if (av > Wc(0,a)) Wc(0,a) = av;
            av = fabs(Val(2,j)); if (av > Wc(0,a)) Wc(0,a) = av;
            av = fabs(Val(1,j)); if (av > Wc(1,a)) Wc(1,a) = av;
            av = fabs(Val(3,j)); if (av > Wc(1,a)) Wc(1,a) = av;
          }
          Wr(0,Ac) = mx0;
          Wr(1,Ac) = mx1;
        }
      break;

      case 3:
        for (int Ac = 0; Ac < nNo; Ac++) {
          double mx0 = 0.0, mx1 = 0.0, mx2 = 0.0;
          for (int j = rowPtr(0,Ac); j <= rowPtr(1,Ac); j++) {
            double av;
            // Row max: rows 0-2 for dof 0, rows 3-5 for dof 1, rows 6-8 for dof 2
            av = fabs(Val(0,j)); if (av > mx0) mx0 = av;
            av = fabs(Val(1,j)); if (av > mx0) mx0 = av;
            av = fabs(Val(2,j)); if (av > mx0) mx0 = av;
            av = fabs(Val(3,j)); if (av > mx1) mx1 = av;
            av = fabs(Val(4,j)); if (av > mx1) mx1 = av;
            av = fabs(Val(5,j)); if (av > mx1) mx1 = av;
            av = fabs(Val(6,j)); if (av > mx2) mx2 = av;
            av = fabs(Val(7,j)); if (av > mx2) mx2 = av;
            av = fabs(Val(8,j)); if (av > mx2) mx2 = av;
            // Col max: stride-3 columns
            int a = colPtr(j);
            av = fabs(Val(0,j)); if (av > Wc(0,a)) Wc(0,a) = av;
            av = fabs(Val(3,j)); if (av > Wc(0,a)) Wc(0,a) = av;
            av = fabs(Val(6,j)); if (av > Wc(0,a)) Wc(0,a) = av;
            av = fabs(Val(1,j)); if (av > Wc(1,a)) Wc(1,a) = av;
            av = fabs(Val(4,j)); if (av > Wc(1,a)) Wc(1,a) = av;
            av = fabs(Val(7,j)); if (av > Wc(1,a)) Wc(1,a) = av;
            av = fabs(Val(2,j)); if (av > Wc(2,a)) Wc(2,a) = av;
            av = fabs(Val(5,j)); if (av > Wc(2,a)) Wc(2,a) = av;
            av = fabs(Val(8,j)); if (av > Wc(2,a)) Wc(2,a) = av;
          }
          Wr(0,Ac) = mx0;
          Wr(1,Ac) = mx1;
          Wr(2,Ac) = mx2;
        }
      break;

      case 4:
        for (int Ac = 0; Ac < nNo; Ac++) {
          double mx0 = 0.0, mx1 = 0.0, mx2 = 0.0, mx3 = 0.0;
          for (int j = rowPtr(0,Ac); j <= rowPtr(1,Ac); j++) {
            double av;
            // Row max: rows 0-3 for dof 0, 4-7 for dof 1, 8-11 for dof 2, 12-15 for dof 3
            av = fabs(Val(0,j));  if (av > mx0) mx0 = av;
            av = fabs(Val(1,j));  if (av > mx0) mx0 = av;
            av = fabs(Val(2,j));  if (av > mx0) mx0 = av;
            av = fabs(Val(3,j));  if (av > mx0) mx0 = av;
            av = fabs(Val(4,j));  if (av > mx1) mx1 = av;
            av = fabs(Val(5,j));  if (av > mx1) mx1 = av;
            av = fabs(Val(6,j));  if (av > mx1) mx1 = av;
            av = fabs(Val(7,j));  if (av > mx1) mx1 = av;
            av = fabs(Val(8,j));  if (av > mx2) mx2 = av;
            av = fabs(Val(9,j));  if (av > mx2) mx2 = av;
            av = fabs(Val(10,j)); if (av > mx2) mx2 = av;
            av = fabs(Val(11,j)); if (av > mx2) mx2 = av;
            av = fabs(Val(12,j)); if (av > mx3) mx3 = av;
            av = fabs(Val(13,j)); if (av > mx3) mx3 = av;
            av = fabs(Val(14,j)); if (av > mx3) mx3 = av;
            av = fabs(Val(15,j)); if (av > mx3) mx3 = av;
            // Col max: stride-4 columns
            int a = colPtr(j);
            av = fabs(Val(0,j));  if (av > Wc(0,a)) Wc(0,a) = av;
            av = fabs(Val(4,j));  if (av > Wc(0,a)) Wc(0,a) = av;
            av = fabs(Val(8,j));  if (av > Wc(0,a)) Wc(0,a) = av;
            av = fabs(Val(12,j)); if (av > Wc(0,a)) Wc(0,a) = av;
            av = fabs(Val(1,j));  if (av > Wc(1,a)) Wc(1,a) = av;
            av = fabs(Val(5,j));  if (av > Wc(1,a)) Wc(1,a) = av;
            av = fabs(Val(9,j));  if (av > Wc(1,a)) Wc(1,a) = av;
            av = fabs(Val(13,j)); if (av > Wc(1,a)) Wc(1,a) = av;
            av = fabs(Val(2,j));  if (av > Wc(2,a)) Wc(2,a) = av;
            av = fabs(Val(6,j));  if (av > Wc(2,a)) Wc(2,a) = av;
            av = fabs(Val(10,j)); if (av > Wc(2,a)) Wc(2,a) = av;
            av = fabs(Val(14,j)); if (av > Wc(2,a)) Wc(2,a) = av;
            av = fabs(Val(3,j));  if (av > Wc(3,a)) Wc(3,a) = av;
            av = fabs(Val(7,j));  if (av > Wc(3,a)) Wc(3,a) = av;
            av = fabs(Val(11,j)); if (av > Wc(3,a)) Wc(3,a) = av;
            av = fabs(Val(15,j)); if (av > Wc(3,a)) Wc(3,a) = av;
          }
          Wr(0,Ac) = mx0;
          Wr(1,Ac) = mx1;
          Wr(2,Ac) = mx2;
          Wr(3,Ac) = mx3;
        }
      break;

      default:
        for (int Ac = 0; Ac < nNo; Ac++) {
          // Row max: for each dof row i, scan entries [i*dof .. (i+1)*dof-1] across all columns
          for (int i = 0; i < dof; i++) {
            double mx = 0.0;
            int r0 = i * dof;
            int r1 = r0 + dof - 1;
            for (int j = rowPtr(0,Ac); j <= rowPtr(1,Ac); j++) {
              for (int r = r0; r <= r1; r++) {
                double av = fabs(Val(r,j));
                if (av > mx) mx = av;
              }
            }
            Wr(i,Ac) = mx;
          }

          // Col max: for each dof column b, scan entries b, b+dof, b+2*dof, ...
          for (int j = rowPtr(0,Ac); j <= rowPtr(1,Ac); j++) {
            int a = colPtr(j);
            for (int b = 0; b < dof; b++) {
              double mx = 0.0;
              for (int k = b; k < dof*dof; k += dof) {
                double av = fabs(Val(k,j));
                if (av > mx) mx = av;
              }
              if (mx > Wc(b,a)) Wc(b,a) = mx;
            }
          }
        }
      break;
    }

    fsils_commuv(lhs, dof, Wr);
    fsils_commuv(lhs, dof, Wc);

    if ((max(abs(1.0 - Wr)) < tol) && (max(abs(1.0 - Wc)) < tol)) {
      flag = false;
    }

    Wr = 1.0 / sqrt(Wr);
    Wc = 1.0 / sqrt(Wc);

    pre_mul(rowPtr, lhs.nNo, lhs.nnz, dof, Val, Wr);

    pos_mul(rowPtr, colPtr, lhs.nNo, lhs.nnz, dof, Val, Wc);

    W1 = W1 * Wr;
    W2 = W2 * Wc;

    if (lhs.commu.nTasks > 1) {
      int iflag = flag;
      std::vector<int> gflag(lhs.commu.nTasks);
      MPI_Allgather(&iflag, 1, cm_mod::mplog, gflag.data(), 1, cm_mod::mplog, lhs.commu.comm);
      flag = std::find(gflag.begin(), gflag.end(), 1) != gflag.end();
    }
  } // while

  // Multipling R with Wr: R = Wr*R
  R = W1 * R;
}

//---------
// pre_mul
//---------
// Pre-multipling Val with W: Val = W*Val.
//
// Modifies: Val(dof*dof, nnz)
//
// W(dof,nNo)
//
void pre_mul(const Array<int>& rowPtr, const int nNo, const int nnz, const int dof, Array<double>& Val, const Array<double>& W)
{
  switch (dof) {
    case 1: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        int a = rowPtr(0,Ac);
        int b = rowPtr(1,Ac);
        for (int j = a; j <= b; j++) {
          Val(0,j) = Val(0,j)*W(0,Ac);
        }
      }
    } break;
    
    case 2: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        int a = rowPtr(0,Ac);
        int b = rowPtr(1,Ac);
        for (int i = 0; i < 2; i++) {
          for (int j = a; j <= b; j++) {
            Val(i+0,j) = Val(i+0,j)*W(0,Ac);
            Val(i+2,j) = Val(i+2,j)*W(1,Ac);
          }
        }
      }
    } break;

    case 3: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        int a = rowPtr(0,Ac);
        int b = rowPtr(1,Ac);
        for (int i = 0; i < 3; i++) {
          for (int j = a; j <= b; j++) {
            Val(i+0,j) = Val(i+0,j)*W(0,Ac);
            Val(i+3,j) = Val(i+3,j)*W(1,Ac);
            Val(i+6,j) = Val(i+6,j)*W(2,Ac);
          }
        }
      }
    } break;

    case 4: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        int a = rowPtr(0,Ac);
        int b = rowPtr(1,Ac);

        for (int i = 0; i < 4; i++) {
          for (int j = a; j <= b; j++) {
            Val(i+0,j) = Val(i+0,j)*W(0,Ac);
            Val(i+4,j) = Val(i+4,j)*W(1,Ac);
            Val(i+8,j) = Val(i+8,j)*W(2,Ac);
            Val(i+12,j) = Val(i+12,j)*W(3,Ac);
          }
        }
      }
    } break;

    // Fill rows of 'Val' with length 'dof'.
    //
    default: {
      for (int Ac = 0; Ac < nNo; Ac++) {
        int a = rowPtr(0,Ac);
        int b = rowPtr(1,Ac);

        for (int i = 0; i < dof; i++) {
          int j = i*dof;

          for (int m = j; m < j+dof; m++) {
            for (int n = a; n <= b; n++) {
              Val(m,n) = Val(m,n) * W(i,Ac);
            }
          }
        }
      }
    } break;
  }
}

};

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

//--------------------------------------------------------------------
// Product of a sparse matrix and a vector. The matrix might be
// vector in neither, one or both dimensions.
//--------------------------------------------------------------------
//
// Reproduces code in SPARMUL.f.

#include "spar_mul.h"

#include "fsils_api.hpp"

#include <algorithm>
#include <vector>

namespace spar_mul {

using fe_fsi_linear_solver::fsils_int;

//====================================================================
// Templated SpMV implementations — DOF known at compile time enables
// full unrolling and vectorization for the common cases (1–4).
// All impls take a row range [iStart, iEnd) for overlap support.
//====================================================================

/// SS: K(nnz) * U(nNo) -> KU(nNo)
static void fsils_spar_mul_ss_range(fsils_int iStart, fsils_int iEnd,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Vector<double>& K, const Vector<double>& U, Vector<double>& KU)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    double sum = 0.0;
    for (fsils_int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
      sum += K(j) * U(colPtr(j));
    }
    KU(i) = sum;
  }
}

/// SV: K(dof, nnz) * U(nNo) -> KU(dof, nNo)
template <int DOF>
static void fsils_spar_mul_sv_impl(fsils_int iStart, fsils_int iEnd,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Vector<double>& U, Array<double>& KU)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    double sums[DOF] = {};
    for (fsils_int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
      double u_col = U(colPtr(j));
      for (int m = 0; m < DOF; m++) {
        sums[m] += K(m,j) * u_col;
      }
    }
    for (int m = 0; m < DOF; m++) {
      KU(m,i) = sums[m];
    }
  }
}

static void fsils_spar_mul_sv_dyn(fsils_int iStart, fsils_int iEnd, int dof,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Vector<double>& U, Array<double>& KU)
{
  #pragma omp parallel
  {
    std::vector<double> sums(static_cast<size_t>(std::max(dof, 0)), 0.0);

    #pragma omp for schedule(static)
    for (fsils_int i = iStart; i < iEnd; i++) {
      std::fill(sums.begin(), sums.end(), 0.0);

      for (fsils_int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
        const double u_col = U(colPtr(j));
        for (int m = 0; m < dof; m++) {
          sums[static_cast<size_t>(m)] += K(m,j) * u_col;
        }
      }

      for (int m = 0; m < dof; m++) {
        KU(m,i) = sums[static_cast<size_t>(m)];
      }
    }
  }
}

/// VS: K(dof, nnz) * U(dof, nNo) -> KU(nNo) (dot-product per row)
template <int DOF>
static void fsils_spar_mul_vs_impl(fsils_int iStart, fsils_int iEnd,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Array<double>& U, Vector<double>& KU)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    double sum = 0.0;
    for (fsils_int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
      fsils_int col = colPtr(j);
      for (int m = 0; m < DOF; m++) {
        sum += K(m,j) * U(m,col);
      }
    }
    KU(i) = sum;
  }
}

static void fsils_spar_mul_vs_dyn(fsils_int iStart, fsils_int iEnd, int dof,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Array<double>& U, Vector<double>& KU)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    double sum = 0.0;
    for (fsils_int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
      fsils_int col = colPtr(j);
      for (int m = 0; m < dof; m++) {
        sum += K(m,j) * U(m,col);
      }
    }
    KU(i) = sum;
  }
}

/// VV: K(dof*dof, nnz) * U(dof, nNo) -> KU(dof, nNo)
template <int DOF>
static void fsils_spar_mul_vv_impl(fsils_int iStart, fsils_int iEnd,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Array<double>& U, Array<double>& KU)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    double sums[DOF] = {};
    for (fsils_int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
      fsils_int col = colPtr(j);
      for (int l = 0; l < DOF; l++) {
        double s = 0.0;
        for (int k = 0; k < DOF; k++) {
          s += K(l*DOF + k, j) * U(k, col);
        }
        sums[l] += s;
      }
    }
    for (int l = 0; l < DOF; l++) {
      KU(l, i) = sums[l];
    }
  }
}

static void fsils_spar_mul_vv_dyn(fsils_int iStart, fsils_int iEnd, int dof,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Array<double>& U, Array<double>& KU)
{
  #pragma omp parallel
  {
    std::vector<double> sums(static_cast<size_t>(std::max(dof, 0)), 0.0);

    #pragma omp for schedule(static)
    for (fsils_int i = iStart; i < iEnd; i++) {
      std::fill(sums.begin(), sums.end(), 0.0);

      for (fsils_int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
        const fsils_int col = colPtr(j);
        for (int l = 0; l < dof; l++) {
          const int s = l * dof;
          double sum = 0.0;
          for (int k = 0; k < dof; k++) {
            sum += K(s+k,j) * U(k,col);
          }
          sums[static_cast<size_t>(l)] += sum;
        }
      }

      for (int l = 0; l < dof; l++) {
        KU(l,i) = sums[static_cast<size_t>(l)];
      }
    }
  }
}

//====================================================================
// Public API — overlaps SpMV with MPI communication when possible.
//
// Node ordering from fsils_lhs_create:
//   [0, shnNo)       shared with lower-rank procs
//   [shnNo, mynNo)   interior (no communication needed)
//   [mynNo, nNo)      shared with higher-rank procs
//
// Overlap pattern: compute boundary rows -> start async comm ->
//   compute interior rows -> finish comm.
//====================================================================

void fsils_spar_mul_ss(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const Vector<double>& K,
    const Vector<double>& U, Vector<double>& KU)
{
  fsils_int nNo = lhs.nNo;

  if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
    fsils_spar_mul_ss_range(0, lhs.shnNo, rowPtr, colPtr, K, U, KU);
    fsils_spar_mul_ss_range(lhs.mynNo, nNo, rowPtr, colPtr, K, U, KU);
    fsils_commus_begin(lhs, KU);
    fsils_spar_mul_ss_range(lhs.shnNo, lhs.mynNo, rowPtr, colPtr, K, U, KU);
    fsils_commus_end(lhs, KU);
  } else {
    fsils_spar_mul_ss_range(0, nNo, rowPtr, colPtr, K, U, KU);
  }
}

void fsils_spar_mul_sv(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const int dof, const Array<double>& K,
    const Vector<double>& U, Array<double>& KU)
{
  fsils_int nNo = lhs.nNo;

  auto compute_range = [&](fsils_int iStart, fsils_int iEnd) {
    switch (dof) {
      case 1: fsils_spar_mul_sv_impl<1>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      case 2: fsils_spar_mul_sv_impl<2>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      case 3: fsils_spar_mul_sv_impl<3>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      case 4: fsils_spar_mul_sv_impl<4>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      default: fsils_spar_mul_sv_dyn(iStart, iEnd, dof, rowPtr, colPtr, K, U, KU); break;
    }
  };

  if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
    compute_range(0, lhs.shnNo);
    compute_range(lhs.mynNo, nNo);
    fsils_commuv_begin(lhs, dof, KU);
    compute_range(lhs.shnNo, lhs.mynNo);
    fsils_commuv_end(lhs, dof, KU);
  } else {
    compute_range(0, nNo);
  }
}

void fsils_spar_mul_vs(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const int dof, const Array<double>& K,
    const Array<double>& U, Vector<double>& KU)
{
  fsils_int nNo = lhs.nNo;

  auto compute_range = [&](fsils_int iStart, fsils_int iEnd) {
    switch (dof) {
      case 1: fsils_spar_mul_vs_impl<1>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      case 2: fsils_spar_mul_vs_impl<2>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      case 3: fsils_spar_mul_vs_impl<3>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      case 4: fsils_spar_mul_vs_impl<4>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      default: fsils_spar_mul_vs_dyn(iStart, iEnd, dof, rowPtr, colPtr, K, U, KU); break;
    }
  };

  if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
    compute_range(0, lhs.shnNo);
    compute_range(lhs.mynNo, nNo);
    fsils_commus_begin(lhs, KU);
    compute_range(lhs.shnNo, lhs.mynNo);
    fsils_commus_end(lhs, KU);
  } else {
    compute_range(0, nNo);
  }
}

void fsils_spar_mul_vv(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const int dof, const Array<double>& K,
    const Array<double>& U, Array<double>& KU)
{
  fsils_int nNo = lhs.nNo;

  auto compute_range = [&](fsils_int iStart, fsils_int iEnd) {
    switch (dof) {
      case 1: fsils_spar_mul_vv_impl<1>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      case 2: fsils_spar_mul_vv_impl<2>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      case 3: fsils_spar_mul_vv_impl<3>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      case 4: fsils_spar_mul_vv_impl<4>(iStart, iEnd, rowPtr, colPtr, K, U, KU); break;
      default: fsils_spar_mul_vv_dyn(iStart, iEnd, dof, rowPtr, colPtr, K, U, KU); break;
    }
  };

  if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
    compute_range(0, lhs.shnNo);
    compute_range(lhs.mynNo, nNo);
    fsils_commuv_begin(lhs, dof, KU);
    compute_range(lhs.shnNo, lhs.mynNo);
    fsils_commuv_end(lhs, dof, KU);
  } else {
    compute_range(0, nNo);
  }
}

};

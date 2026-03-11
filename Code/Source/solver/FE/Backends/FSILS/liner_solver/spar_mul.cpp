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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spar_mul {

using fe_fsi_linear_solver::fsils_int;

namespace {

[[nodiscard]] bool use_serial_hot_path() noexcept
{
#ifdef _OPENMP
  return omp_get_max_threads() <= 1;
#else
  return true;
#endif
}

} // namespace

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
  const fsils_int* __restrict__ rp = rowPtr.data();   // column-major: rowPtr(r,c) = rp[r + c*2]
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ k_data = K.data();
  const double* __restrict__ u_data = U.data();
  double* __restrict__ ku_data = KU.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    const fsils_int j_start = rp[2*i];
    const fsils_int j_end   = rp[2*i + 1];
    double sum = 0.0;
    for (fsils_int j = j_start; j <= j_end; j++) {
      sum += k_data[j] * u_data[cp[j]];
    }
    ku_data[i] = sum;
  }
}

/// SV: K(dof, nnz) * U(nNo) -> KU(dof, nNo)
template <int DOF>
static void fsils_spar_mul_sv_impl(fsils_int iStart, fsils_int iEnd,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Vector<double>& U, Array<double>& KU)
{
  const fsils_int* __restrict__ rp = rowPtr.data();
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ k_data = K.data();
  const double* __restrict__ u_data = U.data();
  double* __restrict__ ku_data = KU.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    double sums[DOF] = {};
    const fsils_int j_start = rp[2*i];
    const fsils_int j_end   = rp[2*i + 1];
    for (fsils_int j = j_start; j <= j_end; j++) {
      const double u_col = u_data[cp[j]];
      const double* __restrict__ kj = k_data + static_cast<size_t>(j) * DOF;
      for (int m = 0; m < DOF; m++) {
        sums[m] += kj[m] * u_col;
      }
    }
    double* __restrict__ kui = ku_data + static_cast<size_t>(i) * DOF;
    for (int m = 0; m < DOF; m++) {
      kui[m] = sums[m];
    }
  }
}

static void fsils_spar_mul_sv_dyn(fsils_int iStart, fsils_int iEnd, int dof,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Vector<double>& U, Array<double>& KU)
{
  const fsils_int* __restrict__ rp = rowPtr.data();
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ k_data = K.data();
  const double* __restrict__ u_data = U.data();
  double* __restrict__ ku_data = KU.data();

  #pragma omp parallel
  {
    std::vector<double> sums(static_cast<size_t>(std::max(dof, 0)), 0.0);

    #pragma omp for schedule(static)
    for (fsils_int i = iStart; i < iEnd; i++) {
      std::fill(sums.begin(), sums.end(), 0.0);

      const fsils_int j_start = rp[2*i];
      const fsils_int j_end   = rp[2*i + 1];
      for (fsils_int j = j_start; j <= j_end; j++) {
        const double u_col = u_data[cp[j]];
        const double* __restrict__ kj = k_data + static_cast<size_t>(j) * dof;
        for (int m = 0; m < dof; m++) {
          sums[static_cast<size_t>(m)] += kj[m] * u_col;
        }
      }

      double* __restrict__ kui = ku_data + static_cast<size_t>(i) * dof;
      for (int m = 0; m < dof; m++) {
        kui[m] = sums[static_cast<size_t>(m)];
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
  const fsils_int* __restrict__ rp = rowPtr.data();
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ k_data = K.data();
  const double* __restrict__ u_data = U.data();
  double* __restrict__ ku_data = KU.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    const fsils_int j_start = rp[2*i];
    const fsils_int j_end   = rp[2*i + 1];
    double sum = 0.0;
    for (fsils_int j = j_start; j <= j_end; j++) {
      const fsils_int col = cp[j];
      const double* __restrict__ kj = k_data + static_cast<size_t>(j) * DOF;
      const double* __restrict__ uc = u_data + static_cast<size_t>(col) * DOF;
      for (int m = 0; m < DOF; m++) {
        sum += kj[m] * uc[m];
      }
    }
    ku_data[i] = sum;
  }
}

static void fsils_spar_mul_vs_dyn(fsils_int iStart, fsils_int iEnd, int dof,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Array<double>& U, Vector<double>& KU)
{
  const fsils_int* __restrict__ rp = rowPtr.data();
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ k_data = K.data();
  const double* __restrict__ u_data = U.data();
  double* __restrict__ ku_data = KU.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    const fsils_int j_start = rp[2*i];
    const fsils_int j_end   = rp[2*i + 1];
    double sum = 0.0;
    for (fsils_int j = j_start; j <= j_end; j++) {
      const fsils_int col = cp[j];
      const double* __restrict__ kj = k_data + static_cast<size_t>(j) * dof;
      const double* __restrict__ uc = u_data + static_cast<size_t>(col) * dof;
      for (int m = 0; m < dof; m++) {
        sum += kj[m] * uc[m];
      }
    }
    ku_data[i] = sum;
  }
}

/// VV: K(dof*dof, nnz) * U(dof, nNo) -> KU(dof, nNo)
template <int DOF>
static void fsils_spar_mul_vv_impl(fsils_int iStart, fsils_int iEnd,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Array<double>& U, Array<double>& KU)
{
  constexpr int DOF2 = DOF * DOF;
  const fsils_int* __restrict__ rp = rowPtr.data();   // column-major: rowPtr(r,c) = rp[r + c*2]
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ k_data = K.data();
  const double* __restrict__ u_data = U.data();
  double* __restrict__ ku_data = KU.data();

  if (use_serial_hot_path()) {
    for (fsils_int i = iStart; i < iEnd; i++) {
      double sums[DOF] = {};
      const fsils_int j_start = rp[2*i];
      const fsils_int j_end   = rp[2*i + 1];
      for (fsils_int j = j_start; j <= j_end; j++) {
        const fsils_int col = cp[j];
        const double* __restrict__ kj = k_data + static_cast<size_t>(j) * DOF2;
        const double* __restrict__ uc = u_data + static_cast<size_t>(col) * DOF;
        for (int l = 0; l < DOF; l++) {
          double s = 0.0;
          for (int k = 0; k < DOF; k++) {
            s += kj[l * DOF + k] * uc[k];
          }
          sums[l] += s;
        }
      }
      double* __restrict__ kui = ku_data + static_cast<size_t>(i) * DOF;
      for (int l = 0; l < DOF; l++) {
        kui[l] = sums[l];
      }
    }
    return;
  }

  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    double sums[DOF] = {};
    const fsils_int j_start = rp[2*i];      // rowPtr(0, i)
    const fsils_int j_end   = rp[2*i + 1];  // rowPtr(1, i)
    for (fsils_int j = j_start; j <= j_end; j++) {
      const fsils_int col = cp[j];
      const double* __restrict__ kj = k_data + static_cast<size_t>(j) * DOF2;
      const double* __restrict__ uc = u_data + static_cast<size_t>(col) * DOF;
      for (int l = 0; l < DOF; l++) {
        double s = 0.0;
        for (int k = 0; k < DOF; k++) {
          s += kj[l * DOF + k] * uc[k];
        }
        sums[l] += s;
      }
    }
    double* __restrict__ kui = ku_data + static_cast<size_t>(i) * DOF;
    for (int l = 0; l < DOF; l++) {
      kui[l] = sums[l];
    }
  }
}

static void fsils_spar_mul_vv_dyn(fsils_int iStart, fsils_int iEnd, int dof,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Array<double>& U, Array<double>& KU)
{
  if (use_serial_hot_path()) {
    std::vector<double> sums(static_cast<size_t>(std::max(dof, 0)), 0.0);
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
    return;
  }

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

//====================================================================
// Fused SV+SS: G(nsd,nnz)*in_vec -> GP(nsd,nNo) AND L(nnz)*in_vec -> SP(nNo)
// in a single row loop. Saves one full traversal of colPtr + in_vec.
//====================================================================

template <int DOF>
static void fsils_spar_mul_sv_ss_fused_impl(fsils_int iStart, fsils_int iEnd,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& G, const Vector<double>& L,
    const Vector<double>& in_vec, Array<double>& GP, Vector<double>& SP)
{
  const fsils_int* __restrict__ rp = rowPtr.data();
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ g_data = G.data();
  const double* __restrict__ l_data = L.data();
  const double* __restrict__ u_data = in_vec.data();
  double* __restrict__ gp_data = GP.data();
  double* __restrict__ sp_data = SP.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int i = iStart; i < iEnd; i++) {
    double gp_sums[DOF] = {};
    double sp_sum = 0.0;
    const fsils_int j_start = rp[2*i];
    const fsils_int j_end   = rp[2*i + 1];
    for (fsils_int j = j_start; j <= j_end; j++) {
      const double u_col = u_data[cp[j]];
      const double* __restrict__ gj = g_data + static_cast<size_t>(j) * DOF;
      for (int m = 0; m < DOF; m++) {
        gp_sums[m] += gj[m] * u_col;
      }
      sp_sum += l_data[j] * u_col;
    }
    double* __restrict__ gpi = gp_data + static_cast<size_t>(i) * DOF;
    for (int m = 0; m < DOF; m++) {
      gpi[m] = gp_sums[m];
    }
    sp_data[i] = sp_sum;
  }
}

static void fsils_spar_mul_sv_ss_fused_dyn(fsils_int iStart, fsils_int iEnd, int nsd,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& G, const Vector<double>& L,
    const Vector<double>& in_vec, Array<double>& GP, Vector<double>& SP)
{
  const fsils_int* __restrict__ rp = rowPtr.data();
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ g_data = G.data();
  const double* __restrict__ l_data = L.data();
  const double* __restrict__ u_data = in_vec.data();
  double* __restrict__ gp_data = GP.data();
  double* __restrict__ sp_data = SP.data();

  #pragma omp parallel
  {
    std::vector<double> gp_sums(static_cast<size_t>(std::max(nsd, 0)), 0.0);

    #pragma omp for schedule(static)
    for (fsils_int i = iStart; i < iEnd; i++) {
      std::fill(gp_sums.begin(), gp_sums.end(), 0.0);
      double sp_sum = 0.0;
      const fsils_int j_start = rp[2*i];
      const fsils_int j_end   = rp[2*i + 1];
      for (fsils_int j = j_start; j <= j_end; j++) {
        const double u_col = u_data[cp[j]];
        const double* __restrict__ gj = g_data + static_cast<size_t>(j) * nsd;
        for (int m = 0; m < nsd; m++) {
          gp_sums[static_cast<size_t>(m)] += gj[m] * u_col;
        }
        sp_sum += l_data[j] * u_col;
      }
      double* __restrict__ gpi = gp_data + static_cast<size_t>(i) * nsd;
      for (int m = 0; m < nsd; m++) {
        gpi[m] = gp_sums[static_cast<size_t>(m)];
      }
      sp_data[i] = sp_sum;
    }
  }
}

void fsils_spar_mul_sv_ss_fused(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const int nsd, const Array<double>& G,
    const Vector<double>& L, const Vector<double>& in_vec,
    Array<double>& GP, Vector<double>& SP)
{
  fsils_int nNo = lhs.nNo;

  auto compute_range = [&](fsils_int iStart, fsils_int iEnd) {
    switch (nsd) {
      case 2: fsils_spar_mul_sv_ss_fused_impl<2>(iStart, iEnd, rowPtr, colPtr, G, L, in_vec, GP, SP); break;
      case 3: fsils_spar_mul_sv_ss_fused_impl<3>(iStart, iEnd, rowPtr, colPtr, G, L, in_vec, GP, SP); break;
      default: fsils_spar_mul_sv_ss_fused_dyn(iStart, iEnd, nsd, rowPtr, colPtr, G, L, in_vec, GP, SP); break;
    }
  };

  if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
    // Compute boundary rows first
    compute_range(0, lhs.shnNo);
    compute_range(lhs.mynNo, nNo);
    // Overlap GP (vector) comm with interior computation
    fsils_commuv_begin(lhs, nsd, GP);
    compute_range(lhs.shnNo, lhs.mynNo);
    fsils_commuv_end(lhs, nsd, GP);
    // SP (scalar) comm — boundary rows already computed
    fsils_commus_begin(lhs, SP);
    fsils_commus_end(lhs, SP);
  } else {
    compute_range(0, nNo);
  }
}

//====================================================================
// Rectangular block SpMV: K(out_dof * in_dof, nnz) × U(in_dof, nNo) → KU(out_dof, nNo)
// Generalizes SV (in_dof=1), VS (out_dof=1), VV (out_dof=in_dof).
//====================================================================

static void fsils_spar_mul_rect_range(fsils_int iStart, fsils_int iEnd,
    int out_dof, int in_dof,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& K, const Array<double>& U, Array<double>& KU)
{
  const int block_size = out_dof * in_dof;
  const fsils_int* __restrict__ rp = rowPtr.data();
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ k_data = K.data();
  const double* __restrict__ u_data = U.data();
  double* __restrict__ ku_data = KU.data();

  #pragma omp parallel
  {
    std::vector<double> sums(static_cast<size_t>(std::max(out_dof, 0)), 0.0);

    #pragma omp for schedule(static)
    for (fsils_int i = iStart; i < iEnd; i++) {
      std::fill(sums.begin(), sums.end(), 0.0);
      const fsils_int j_start = rp[2*i];
      const fsils_int j_end   = rp[2*i + 1];
      for (fsils_int j = j_start; j <= j_end; j++) {
        const fsils_int col = cp[j];
        const double* __restrict__ kj = k_data + static_cast<size_t>(j) * block_size;
        const double* __restrict__ uc = u_data + static_cast<size_t>(col) * in_dof;
        for (int l = 0; l < out_dof; l++) {
          double s = 0.0;
          for (int k = 0; k < in_dof; k++) {
            s += kj[l * in_dof + k] * uc[k];
          }
          sums[static_cast<size_t>(l)] += s;
        }
      }
      double* __restrict__ kui = ku_data + static_cast<size_t>(i) * out_dof;
      for (int l = 0; l < out_dof; l++) {
        kui[l] = sums[static_cast<size_t>(l)];
      }
    }
  }
}

void fsils_spar_mul_rect(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, int out_dof, int in_dof,
    const Array<double>& K, const Array<double>& U, Array<double>& KU)
{
  // Dispatch to specialized routines for common degenerate cases.
  if (in_dof == 1 && out_dof == 1) {
    // SS: treat K and U as vectors.  Layout is compatible since K(1*1, nnz)
    // is column-major contiguous like Vector<double>(nnz).
    // Wrap as Vectors via const_cast-free copies.
    Vector<double> Kv(K.ncols());
    for (fsils_int n = 0; n < K.ncols(); n++) Kv(n) = K(0, n);
    Vector<double> Uv(U.ncols());
    for (fsils_int n = 0; n < U.ncols(); n++) Uv(n) = U(0, n);
    Vector<double> KUv(KU.ncols());
    fsils_spar_mul_ss(lhs, rowPtr, colPtr, Kv, Uv, KUv);
    for (fsils_int n = 0; n < KU.ncols(); n++) KU(0, n) = KUv(n);
    return;
  }
  if (out_dof == in_dof) {
    fsils_spar_mul_vv(lhs, rowPtr, colPtr, out_dof, K, U, KU);
    return;
  }

  fsils_int nNo = lhs.nNo;

  if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
    fsils_spar_mul_rect_range(0, lhs.shnNo, out_dof, in_dof, rowPtr, colPtr, K, U, KU);
    fsils_spar_mul_rect_range(lhs.mynNo, nNo, out_dof, in_dof, rowPtr, colPtr, K, U, KU);
    fsils_commuv_begin(lhs, out_dof, KU);
    fsils_spar_mul_rect_range(lhs.shnNo, lhs.mynNo, out_dof, in_dof, rowPtr, colPtr, K, U, KU);
    fsils_commuv_end(lhs, out_dof, KU);
  } else {
    fsils_spar_mul_rect_range(0, nNo, out_dof, in_dof, rowPtr, colPtr, K, U, KU);
  }
}

//====================================================================
// Fused rectangular G×P + square L×P SpMV for multi-component Schur.
// G(mom_ncomp * con_ncomp, nnz) × P(con_ncomp, nNo) → GP(mom_ncomp, nNo)
// L(con_ncomp * con_ncomp, nnz) × P(con_ncomp, nNo) → SP(con_ncomp, nNo)
//====================================================================

static void fsils_spar_mul_rect_vv_fused_range(fsils_int iStart, fsils_int iEnd,
    int mom_ncomp, int con_ncomp,
    const Array<fsils_int>& rowPtr, const Vector<fsils_int>& colPtr,
    const Array<double>& G, const Array<double>& L,
    const Array<double>& P, Array<double>& GP, Array<double>& SP)
{
  const int g_block = mom_ncomp * con_ncomp;
  const int l_block = con_ncomp * con_ncomp;
  const fsils_int* __restrict__ rp = rowPtr.data();
  const fsils_int* __restrict__ cp = colPtr.data();
  const double* __restrict__ g_data = G.data();
  const double* __restrict__ l_data = L.data();
  const double* __restrict__ p_data = P.data();
  double* __restrict__ gp_data = GP.data();
  double* __restrict__ sp_data = SP.data();

  #pragma omp parallel
  {
    std::vector<double> g_sums(static_cast<size_t>(mom_ncomp), 0.0);
    std::vector<double> l_sums(static_cast<size_t>(con_ncomp), 0.0);

    #pragma omp for schedule(static)
    for (fsils_int i = iStart; i < iEnd; i++) {
      std::fill(g_sums.begin(), g_sums.end(), 0.0);
      std::fill(l_sums.begin(), l_sums.end(), 0.0);
      const fsils_int j_start = rp[2*i];
      const fsils_int j_end   = rp[2*i + 1];
      for (fsils_int j = j_start; j <= j_end; j++) {
        const fsils_int col = cp[j];
        const double* __restrict__ pc = p_data + static_cast<size_t>(col) * con_ncomp;
        const double* __restrict__ gj = g_data + static_cast<size_t>(j) * g_block;
        const double* __restrict__ lj = l_data + static_cast<size_t>(j) * l_block;
        // G accumulation: GP(l,i) += Σ_k G(l*con_ncomp+k, j) * P(k, col)
        for (int l = 0; l < mom_ncomp; l++) {
          double s = 0.0;
          for (int k = 0; k < con_ncomp; k++) {
            s += gj[l * con_ncomp + k] * pc[k];
          }
          g_sums[static_cast<size_t>(l)] += s;
        }
        // L accumulation: SP(l,i) += Σ_k L(l*con_ncomp+k, j) * P(k, col)
        for (int l = 0; l < con_ncomp; l++) {
          double s = 0.0;
          for (int k = 0; k < con_ncomp; k++) {
            s += lj[l * con_ncomp + k] * pc[k];
          }
          l_sums[static_cast<size_t>(l)] += s;
        }
      }
      double* __restrict__ gpi = gp_data + static_cast<size_t>(i) * mom_ncomp;
      for (int l = 0; l < mom_ncomp; l++) gpi[l] = g_sums[static_cast<size_t>(l)];
      double* __restrict__ spi = sp_data + static_cast<size_t>(i) * con_ncomp;
      for (int l = 0; l < con_ncomp; l++) spi[l] = l_sums[static_cast<size_t>(l)];
    }
  }
}

void fsils_spar_mul_rect_vv_fused(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, int mom_ncomp, int con_ncomp,
    const Array<double>& G, const Array<double>& L,
    const Array<double>& P, Array<double>& GP, Array<double>& SP)
{
  // For con_ncomp == 1, delegate to the optimized fused SV+SS path.
  if (con_ncomp == 1) {
    Vector<double> Lv(L.ncols());
    for (fsils_int n = 0; n < L.ncols(); n++) Lv(n) = L(0, n);
    Vector<double> Pv(P.ncols());
    for (fsils_int n = 0; n < P.ncols(); n++) Pv(n) = P(0, n);
    Vector<double> SPv(SP.ncols());
    fsils_spar_mul_sv_ss_fused(lhs, rowPtr, colPtr, mom_ncomp, G, Lv, Pv, GP, SPv);
    for (fsils_int n = 0; n < SP.ncols(); n++) SP(0, n) = SPv(n);
    return;
  }

  fsils_int nNo = lhs.nNo;

  auto compute_range = [&](fsils_int iStart, fsils_int iEnd) {
    fsils_spar_mul_rect_vv_fused_range(iStart, iEnd, mom_ncomp, con_ncomp,
        rowPtr, colPtr, G, L, P, GP, SP);
  };

  if (lhs.commu.nTasks > 1 && lhs.nReq > 0) {
    compute_range(0, lhs.shnNo);
    compute_range(lhs.mynNo, nNo);
    // Overlap GP comm with interior
    fsils_commuv_begin(lhs, mom_ncomp, GP);
    compute_range(lhs.shnNo, lhs.mynNo);
    fsils_commuv_end(lhs, mom_ncomp, GP);
    // SP comm
    fsils_commuv_begin(lhs, con_ncomp, SP);
    fsils_commuv_end(lhs, con_ncomp, SP);
  } else {
    compute_range(0, nNo);
  }
}

};

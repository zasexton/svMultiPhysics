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

#include "distributed_sparse_operator.h"
#include "fsils_api.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
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
// Fused local kernels used by the distributed sparse-operator layer.
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

//====================================================================
// Rectangular block local SpMV kernels.
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

}  // namespace spar_mul

namespace fe_fsi_linear_solver::distributed_sparse_operator {

namespace {

[[nodiscard]] bool valid_output_state(VectorState state) noexcept
{
  return state == VectorState::owned_only ||
         state == VectorState::ghost_zeroed;
}

[[nodiscard]] std::string invalid_output_state_message(const char* op_name,
                                                       const char* role,
                                                       VectorState actual)
{
  return std::string(op_name) + " requires " + role +
      " state `owned_only` or `ghost_zeroed` but received `" +
      to_string(actual) + "`";
}

[[nodiscard]] std::string state_mismatch_message(const char* op_name,
                                                 const char* role,
                                                 VectorState expected,
                                                 VectorState actual)
{
  return std::string(op_name) + " requires " + role + " state `" +
      to_string(expected) + "` but received `" + to_string(actual) + "`";
}

void require_state(const char* op_name,
                   const char* role,
                   VectorState expected,
                   VectorState actual)
{
  if (actual != expected) {
    throw std::logic_error(state_mismatch_message(op_name, role, expected, actual));
  }
}

void require_output_state(const char* op_name,
                          const char* role,
                          VectorState actual)
{
  if (!valid_output_state(actual)) {
    throw std::logic_error(invalid_output_state_message(op_name, role, actual));
  }
}

void require_components(const char* op_name,
                        const char* role,
                        int expected,
                        int actual)
{
  if (actual != expected) {
    throw std::logic_error(std::string(op_name) + " requires " + role + " components=" +
        std::to_string(expected) + " but received " + std::to_string(actual));
  }
}

void zero_scalar_range(fsils_int i_start, fsils_int i_end, Vector<double>& values)
{
  for (fsils_int i = i_start; i < i_end; ++i) {
    values(i) = 0.0;
  }
}

void zero_vector_range(fsils_int i_start, fsils_int i_end, int components, Array<double>& values)
{
  for (fsils_int i = i_start; i < i_end; ++i) {
    for (int c = 0; c < components; ++c) {
      values(c, i) = 0.0;
    }
  }
}

template <typename ComputeRange>
void apply_scalar_output(const FSILS_lhsType& lhs,
                         ComputeRange&& compute_range,
                         Vector<double>& output)
{
  const fsils_int nNo = lhs.nNo;
  compute_range(0, lhs.mynNo);
  zero_scalar_range(lhs.mynNo, nNo, output);
}

template <typename ComputeRange>
void apply_vector_output(const FSILS_lhsType& lhs,
                         int components,
                         ComputeRange&& compute_range,
                         Array<double>& output)
{
  const fsils_int nNo = lhs.nNo;
  compute_range(0, lhs.mynNo);
  zero_vector_range(lhs.mynNo, nNo, components, output);
}

void apply_sv_range(fsils_int i_start,
                    fsils_int i_end,
                    int dof,
                    const Array<fsils_int>& row_ptr,
                    const Vector<fsils_int>& col_ptr,
                    const Array<double>& values,
                    const Vector<double>& input,
                    Array<double>& output)
{
  switch (dof) {
    case 1: spar_mul::fsils_spar_mul_sv_impl<1>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    case 2: spar_mul::fsils_spar_mul_sv_impl<2>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    case 3: spar_mul::fsils_spar_mul_sv_impl<3>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    case 4: spar_mul::fsils_spar_mul_sv_impl<4>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    default: spar_mul::fsils_spar_mul_sv_dyn(i_start, i_end, dof, row_ptr, col_ptr, values, input, output); break;
  }
}

void apply_vs_range(fsils_int i_start,
                    fsils_int i_end,
                    int dof,
                    const Array<fsils_int>& row_ptr,
                    const Vector<fsils_int>& col_ptr,
                    const Array<double>& values,
                    const Array<double>& input,
                    Vector<double>& output)
{
  switch (dof) {
    case 1: spar_mul::fsils_spar_mul_vs_impl<1>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    case 2: spar_mul::fsils_spar_mul_vs_impl<2>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    case 3: spar_mul::fsils_spar_mul_vs_impl<3>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    case 4: spar_mul::fsils_spar_mul_vs_impl<4>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    default: spar_mul::fsils_spar_mul_vs_dyn(i_start, i_end, dof, row_ptr, col_ptr, values, input, output); break;
  }
}

void apply_vv_range(fsils_int i_start,
                    fsils_int i_end,
                    int dof,
                    const Array<fsils_int>& row_ptr,
                    const Vector<fsils_int>& col_ptr,
                    const Array<double>& values,
                    const Array<double>& input,
                    Array<double>& output)
{
  switch (dof) {
    case 1: spar_mul::fsils_spar_mul_vv_impl<1>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    case 2: spar_mul::fsils_spar_mul_vv_impl<2>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    case 3: spar_mul::fsils_spar_mul_vv_impl<3>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    case 4: spar_mul::fsils_spar_mul_vv_impl<4>(i_start, i_end, row_ptr, col_ptr, values, input, output); break;
    default: spar_mul::fsils_spar_mul_vv_dyn(i_start, i_end, dof, row_ptr, col_ptr, values, input, output); break;
  }
}

void apply_fused_sv_ss_range(fsils_int i_start,
                             fsils_int i_end,
                             int components,
                             const Array<fsils_int>& row_ptr,
                             const Vector<fsils_int>& col_ptr,
                             const Array<double>& momentum_values,
                             const Vector<double>& constraint_values,
                             const Vector<double>& input,
                             Array<double>& momentum_output,
                             Vector<double>& constraint_output)
{
  switch (components) {
    case 2: spar_mul::fsils_spar_mul_sv_ss_fused_impl<2>(i_start, i_end, row_ptr, col_ptr,
            momentum_values, constraint_values, input, momentum_output, constraint_output); break;
    case 3: spar_mul::fsils_spar_mul_sv_ss_fused_impl<3>(i_start, i_end, row_ptr, col_ptr,
            momentum_values, constraint_values, input, momentum_output, constraint_output); break;
    default: spar_mul::fsils_spar_mul_sv_ss_fused_dyn(i_start, i_end, components, row_ptr, col_ptr,
            momentum_values, constraint_values, input, momentum_output, constraint_output); break;
  }
}

}  // namespace

const char* to_string(VectorState state) noexcept
{
  switch (state) {
    case VectorState::owned_only: return "owned_only";
    case VectorState::ghost_zeroed: return "ghost_zeroed";
    case VectorState::ghost_synced: return "ghost_synced";
  }
  return "unknown";
}

ScalarToScalarOperator::ScalarToScalarOperator(const FSILS_lhsType& lhs,
                                               const Array<fsils_int>& row_ptr,
                                               const Vector<fsils_int>& col_ptr,
                                               const Vector<double>& values) noexcept
    : lhs_(&lhs),
      row_ptr_(&row_ptr),
      col_ptr_(&col_ptr),
      values_(&values)
{
}

UnaryOperatorContract ScalarToScalarOperator::contract() const noexcept
{
  return {};
}

void ScalarToScalarOperator::apply(ScalarInput input, ScalarOutput output) const
{
  static constexpr const char* kOperatorName = "ScalarToScalarOperator";
  const auto op_contract = contract();
  require_state(kOperatorName, "input", op_contract.input_state, input.state);
  require_output_state(kOperatorName, "output", output.state);

  auto compute_range = [&](fsils_int i_start, fsils_int i_end) {
    spar_mul::fsils_spar_mul_ss_range(i_start, i_end, *row_ptr_, *col_ptr_,
                                      *values_, input.values, output.values);
  };
  apply_scalar_output(*lhs_, compute_range, output.values);
}

ScalarToVectorOperator::ScalarToVectorOperator(const FSILS_lhsType& lhs,
                                               const Array<fsils_int>& row_ptr,
                                               const Vector<fsils_int>& col_ptr,
                                               int components,
                                               const Array<double>& values) noexcept
    : lhs_(&lhs),
      row_ptr_(&row_ptr),
      col_ptr_(&col_ptr),
      components_(components),
      values_(&values)
{
}

UnaryOperatorContract ScalarToVectorOperator::contract() const noexcept
{
  return {};
}

void ScalarToVectorOperator::apply(ScalarInput input, BlockOutput output) const
{
  static constexpr const char* kOperatorName = "ScalarToVectorOperator";
  const auto op_contract = contract();
  require_state(kOperatorName, "input", op_contract.input_state, input.state);
  require_output_state(kOperatorName, "output", output.state);
  require_components(kOperatorName, "output", components_, output.components);

  auto compute_range = [&](fsils_int i_start, fsils_int i_end) {
    apply_sv_range(i_start, i_end, components_, *row_ptr_, *col_ptr_,
                   *values_, input.values, output.values);
  };
  apply_vector_output(*lhs_, components_, compute_range, output.values);
}

VectorToScalarOperator::VectorToScalarOperator(const FSILS_lhsType& lhs,
                                               const Array<fsils_int>& row_ptr,
                                               const Vector<fsils_int>& col_ptr,
                                               int components,
                                               const Array<double>& values) noexcept
    : lhs_(&lhs),
      row_ptr_(&row_ptr),
      col_ptr_(&col_ptr),
      components_(components),
      values_(&values)
{
}

UnaryOperatorContract VectorToScalarOperator::contract() const noexcept
{
  return {};
}

void VectorToScalarOperator::apply(BlockInput input, ScalarOutput output) const
{
  static constexpr const char* kOperatorName = "VectorToScalarOperator";
  const auto op_contract = contract();
  require_state(kOperatorName, "input", op_contract.input_state, input.state);
  require_output_state(kOperatorName, "output", output.state);
  require_components(kOperatorName, "input", components_, input.components);

  auto compute_range = [&](fsils_int i_start, fsils_int i_end) {
    apply_vs_range(i_start, i_end, components_, *row_ptr_, *col_ptr_,
                   *values_, input.values, output.values);
  };
  apply_scalar_output(*lhs_, compute_range, output.values);
}

VectorToVectorOperator::VectorToVectorOperator(const FSILS_lhsType& lhs,
                                               const Array<fsils_int>& row_ptr,
                                               const Vector<fsils_int>& col_ptr,
                                               int components,
                                               const Array<double>& values) noexcept
    : lhs_(&lhs),
      row_ptr_(&row_ptr),
      col_ptr_(&col_ptr),
      components_(components),
      values_(&values)
{
}

UnaryOperatorContract VectorToVectorOperator::contract() const noexcept
{
  return {};
}

void VectorToVectorOperator::apply(BlockInput input, BlockOutput output) const
{
  static constexpr const char* kOperatorName = "VectorToVectorOperator";
  const auto op_contract = contract();
  require_state(kOperatorName, "input", op_contract.input_state, input.state);
  require_output_state(kOperatorName, "output", output.state);
  require_components(kOperatorName, "input", components_, input.components);
  require_components(kOperatorName, "output", components_, output.components);

  auto compute_range = [&](fsils_int i_start, fsils_int i_end) {
    apply_vv_range(i_start, i_end, components_, *row_ptr_, *col_ptr_,
                   *values_, input.values, output.values);
  };
  apply_vector_output(*lhs_, components_, compute_range, output.values);
}

RectangularOperator::RectangularOperator(const FSILS_lhsType& lhs,
                                         const Array<fsils_int>& row_ptr,
                                         const Vector<fsils_int>& col_ptr,
                                         int out_components,
                                         int in_components,
                                         const Array<double>& values) noexcept
    : lhs_(&lhs),
      row_ptr_(&row_ptr),
      col_ptr_(&col_ptr),
      out_components_(out_components),
      in_components_(in_components),
      values_(&values)
{
}

UnaryOperatorContract RectangularOperator::contract() const noexcept
{
  return {};
}

void RectangularOperator::apply(BlockInput input, BlockOutput output) const
{
  static constexpr const char* kOperatorName = "RectangularOperator";
  const auto op_contract = contract();
  require_state(kOperatorName, "input", op_contract.input_state, input.state);
  require_output_state(kOperatorName, "output", output.state);
  require_components(kOperatorName, "input", in_components_, input.components);
  require_components(kOperatorName, "output", out_components_, output.components);

  auto compute_range = [&](fsils_int i_start, fsils_int i_end) {
    if (out_components_ == in_components_) {
      apply_vv_range(i_start, i_end, out_components_, *row_ptr_, *col_ptr_,
                     *values_, input.values, output.values);
      return;
    }
    spar_mul::fsils_spar_mul_rect_range(i_start, i_end, out_components_, in_components_,
                                        *row_ptr_, *col_ptr_, *values_,
                                        input.values, output.values);
  };
  apply_vector_output(*lhs_, out_components_, compute_range, output.values);
}

FusedScalarConstraintOperator::FusedScalarConstraintOperator(
    const FSILS_lhsType& lhs,
    const Array<fsils_int>& row_ptr,
    const Vector<fsils_int>& col_ptr,
    int momentum_components,
    const Array<double>& momentum_values,
    const Vector<double>& constraint_values) noexcept
    : lhs_(&lhs),
      row_ptr_(&row_ptr),
      col_ptr_(&col_ptr),
      momentum_components_(momentum_components),
      momentum_values_(&momentum_values),
      constraint_values_(&constraint_values)
{
}

FusedOperatorContract FusedScalarConstraintOperator::contract() const noexcept
{
  return {};
}

void FusedScalarConstraintOperator::apply(ScalarInput input,
                                          BlockOutput momentum_output,
                                          ScalarOutput constraint_output) const
{
  static constexpr const char* kOperatorName = "FusedScalarConstraintOperator";
  const auto op_contract = contract();
  require_state(kOperatorName, "input", op_contract.input_state, input.state);
  require_output_state(kOperatorName, "momentum output", momentum_output.state);
  require_output_state(kOperatorName, "constraint output", constraint_output.state);
  require_components(kOperatorName, "momentum output", momentum_components_, momentum_output.components);

  auto compute_range = [&](fsils_int i_start, fsils_int i_end) {
    apply_fused_sv_ss_range(i_start, i_end, momentum_components_, *row_ptr_, *col_ptr_,
                            *momentum_values_, *constraint_values_,
                            input.values, momentum_output.values, constraint_output.values);
  };

  const fsils_int nNo = lhs_->nNo;
  compute_range(0, lhs_->mynNo);
  zero_vector_range(lhs_->mynNo, nNo, momentum_components_, momentum_output.values);
  zero_scalar_range(lhs_->mynNo, nNo, constraint_output.values);
}

FusedRectangularConstraintOperator::FusedRectangularConstraintOperator(
    const FSILS_lhsType& lhs,
    const Array<fsils_int>& row_ptr,
    const Vector<fsils_int>& col_ptr,
    int momentum_components,
    int constraint_components,
    const Array<double>& momentum_values,
    const Array<double>& constraint_values) noexcept
    : lhs_(&lhs),
      row_ptr_(&row_ptr),
      col_ptr_(&col_ptr),
      momentum_components_(momentum_components),
      constraint_components_(constraint_components),
      momentum_values_(&momentum_values),
      constraint_values_(&constraint_values)
{
}

FusedOperatorContract FusedRectangularConstraintOperator::contract() const noexcept
{
  return {};
}

void FusedRectangularConstraintOperator::apply(BlockInput input,
                                               BlockOutput momentum_output,
                                               BlockOutput constraint_output) const
{
  static constexpr const char* kOperatorName = "FusedRectangularConstraintOperator";
  const auto op_contract = contract();
  require_state(kOperatorName, "input", op_contract.input_state, input.state);
  require_output_state(kOperatorName, "momentum output", momentum_output.state);
  require_output_state(kOperatorName, "constraint output", constraint_output.state);
  require_components(kOperatorName, "input", constraint_components_, input.components);
  require_components(kOperatorName, "momentum output", momentum_components_, momentum_output.components);
  require_components(kOperatorName, "constraint output", constraint_components_, constraint_output.components);

  auto compute_range = [&](fsils_int i_start, fsils_int i_end) {
    spar_mul::fsils_spar_mul_rect_vv_fused_range(i_start, i_end,
        momentum_components_, constraint_components_, *row_ptr_, *col_ptr_,
        *momentum_values_, *constraint_values_, input.values,
        momentum_output.values, constraint_output.values);
  };

  const fsils_int nNo = lhs_->nNo;
  compute_range(0, lhs_->mynNo);
  zero_vector_range(lhs_->mynNo, nNo, momentum_components_, momentum_output.values);
  zero_vector_range(lhs_->mynNo, nNo, constraint_components_, constraint_output.values);
}

SparseOperatorBundle::SparseOperatorBundle(const FSILS_lhsType& lhs,
                                           const Array<fsils_int>& row_ptr,
                                           const Vector<fsils_int>& col_ptr) noexcept
    : lhs_(&lhs),
      row_ptr_(&row_ptr),
      col_ptr_(&col_ptr)
{
}

ScalarToScalarOperator SparseOperatorBundle::scalar(const Vector<double>& values) const noexcept
{
  return ScalarToScalarOperator(*lhs_, *row_ptr_, *col_ptr_, values);
}

ScalarToVectorOperator SparseOperatorBundle::scalar_to_vector(int components,
                                                              const Array<double>& values) const noexcept
{
  return ScalarToVectorOperator(*lhs_, *row_ptr_, *col_ptr_, components, values);
}

VectorToScalarOperator SparseOperatorBundle::vector_to_scalar(int components,
                                                              const Array<double>& values) const noexcept
{
  return VectorToScalarOperator(*lhs_, *row_ptr_, *col_ptr_, components, values);
}

VectorToVectorOperator SparseOperatorBundle::vector(int components,
                                                    const Array<double>& values) const noexcept
{
  return VectorToVectorOperator(*lhs_, *row_ptr_, *col_ptr_, components, values);
}

RectangularOperator SparseOperatorBundle::rectangular(int out_components,
                                                      int in_components,
                                                      const Array<double>& values) const noexcept
{
  return RectangularOperator(*lhs_, *row_ptr_, *col_ptr_, out_components, in_components, values);
}

FusedScalarConstraintOperator SparseOperatorBundle::fused_scalar_constraint(
    int momentum_components,
    const Array<double>& momentum_values,
    const Vector<double>& constraint_values) const noexcept
{
  return FusedScalarConstraintOperator(*lhs_, *row_ptr_, *col_ptr_, momentum_components,
                                       momentum_values, constraint_values);
}

FusedRectangularConstraintOperator SparseOperatorBundle::fused_rectangular_constraint(
    int momentum_components,
    int constraint_components,
    const Array<double>& momentum_values,
    const Array<double>& constraint_values) const noexcept
{
  return FusedRectangularConstraintOperator(*lhs_, *row_ptr_, *col_ptr_,
                                            momentum_components, constraint_components,
                                            momentum_values, constraint_values);
}

}  // namespace fe_fsi_linear_solver::distributed_sparse_operator

namespace spar_mul {

void fsils_spar_mul_ss(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const Vector<double>& K,
    const Vector<double>& U, Vector<double>& KU)
{
  const auto ops = fe_fsi_linear_solver::distributed_sparse_operator::SparseOperatorBundle(lhs, rowPtr, colPtr);
  ops.scalar(K).apply(fe_fsi_linear_solver::distributed_sparse_operator::ghost_synced_input(U),
                      fe_fsi_linear_solver::distributed_sparse_operator::owned_only_output(KU));
}

void fsils_spar_mul_sv(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const int dof, const Array<double>& K,
    const Vector<double>& U, Array<double>& KU)
{
  const auto ops = fe_fsi_linear_solver::distributed_sparse_operator::SparseOperatorBundle(lhs, rowPtr, colPtr);
  ops.scalar_to_vector(dof, K).apply(
      fe_fsi_linear_solver::distributed_sparse_operator::ghost_synced_input(U),
      fe_fsi_linear_solver::distributed_sparse_operator::owned_only_output(dof, KU));
}

void fsils_spar_mul_vs(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const int dof, const Array<double>& K,
    const Array<double>& U, Vector<double>& KU)
{
  const auto ops = fe_fsi_linear_solver::distributed_sparse_operator::SparseOperatorBundle(lhs, rowPtr, colPtr);
  ops.vector_to_scalar(dof, K).apply(
      fe_fsi_linear_solver::distributed_sparse_operator::ghost_synced_input(dof, U),
      fe_fsi_linear_solver::distributed_sparse_operator::owned_only_output(KU));
}

void fsils_spar_mul_vv(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const int dof, const Array<double>& K,
    const Array<double>& U, Array<double>& KU)
{
  const auto ops = fe_fsi_linear_solver::distributed_sparse_operator::SparseOperatorBundle(lhs, rowPtr, colPtr);
  ops.vector(dof, K).apply(
      fe_fsi_linear_solver::distributed_sparse_operator::ghost_synced_input(dof, U),
      fe_fsi_linear_solver::distributed_sparse_operator::owned_only_output(dof, KU));
}

void fsils_spar_mul_sv_ss_fused(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, const int nsd, const Array<double>& G,
    const Vector<double>& L, const Vector<double>& in_vec,
    Array<double>& GP, Vector<double>& SP)
{
  const auto ops = fe_fsi_linear_solver::distributed_sparse_operator::SparseOperatorBundle(lhs, rowPtr, colPtr);
  ops.fused_scalar_constraint(nsd, G, L).apply(
      fe_fsi_linear_solver::distributed_sparse_operator::ghost_synced_input(in_vec),
      fe_fsi_linear_solver::distributed_sparse_operator::owned_only_output(nsd, GP),
      fe_fsi_linear_solver::distributed_sparse_operator::owned_only_output(SP));
}

void fsils_spar_mul_rect(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, int out_dof, int in_dof,
    const Array<double>& K, const Array<double>& U, Array<double>& KU)
{
  const auto ops = fe_fsi_linear_solver::distributed_sparse_operator::SparseOperatorBundle(lhs, rowPtr, colPtr);
  ops.rectangular(out_dof, in_dof, K).apply(
      fe_fsi_linear_solver::distributed_sparse_operator::ghost_synced_input(in_dof, U),
      fe_fsi_linear_solver::distributed_sparse_operator::owned_only_output(out_dof, KU));
}

void fsils_spar_mul_rect_vv_fused(FSILS_lhsType& lhs, const Array<fsils_int>& rowPtr,
    const Vector<fsils_int>& colPtr, int mom_ncomp, int con_ncomp,
    const Array<double>& G, const Array<double>& L,
    const Array<double>& P, Array<double>& GP, Array<double>& SP)
{
  const auto ops = fe_fsi_linear_solver::distributed_sparse_operator::SparseOperatorBundle(lhs, rowPtr, colPtr);
  ops.fused_rectangular_constraint(mom_ncomp, con_ncomp, G, L).apply(
      fe_fsi_linear_solver::distributed_sparse_operator::ghost_synced_input(con_ncomp, P),
      fe_fsi_linear_solver::distributed_sparse_operator::owned_only_output(mom_ncomp, GP),
      fe_fsi_linear_solver::distributed_sparse_operator::owned_only_output(con_ncomp, SP));
}

}  // namespace spar_mul

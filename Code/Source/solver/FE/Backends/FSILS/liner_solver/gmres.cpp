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
#include <cstdlib>
#include <limits>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gmres {

namespace {

using fe_fsi_linear_solver::fsils_int;

constexpr fsils_int BLOCK_SIZE = 256;
constexpr int DOT_THREAD_PAD = 8; // doubles (64 bytes) to reduce false sharing

[[nodiscard]] int max_omp_threads() noexcept
{
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

[[nodiscard]] int dot_thread_stride(const int max_vecs) noexcept
{
  return max_vecs + DOT_THREAD_PAD;
}

[[nodiscard]] int omp_thread_num() noexcept
{
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

// =======================================================================
// FUSED CACHE-BLOCKED VECTOR KERNELS (GEMV-like Operations)
// Reduces O(i) DRAM passes over the grid to O(1) during Gram-Schmidt
// by operating on L1-sized tiles and fusing per-basis-vector work.
// =======================================================================

template <int DOF>
void fused_dot_v_impl(const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                      std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;
    alignas(64) double vbuf[DOF * BLOCK_SIZE];

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);
      const fsils_int blk_nodes = k_end - k_start;
      const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

      const double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] = v_block[idx];
      }

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t idx = 0; idx < blk_elems; ++idx) {
          sum += uj_block[idx] * vbuf[idx];
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }
}

void fused_dot_v_dyn(const int dof, const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                     std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj = u.slice_data(j);
        double sum = 0.0;
        for (fsils_int k = k_start; k < k_end; ++k) {
          const size_t base = static_cast<size_t>(k) * static_cast<size_t>(dof);
          double tmp = 0.0;
          for (int d = 0; d < dof; ++d) {
            tmp += uj[base + static_cast<size_t>(d)] * v[base + static_cast<size_t>(d)];
          }
          sum += tmp;
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }
}

void fused_dot_v(const int dof, const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                 std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  switch (dof) {
    case 1: fused_dot_v_impl<1>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
    case 2: fused_dot_v_impl<2>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
    case 3: fused_dot_v_impl<3>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
    case 4: fused_dot_v_impl<4>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
    default: fused_dot_v_dyn(dof, mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads); break;
  }
}

template <int DOF>
double fused_dot_zz_v_impl(const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                           std::vector<double>& h_col, double* thread_buf, const int thread_stride,
                           const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;
    alignas(64) double vbuf[DOF * BLOCK_SIZE];

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);
      const fsils_int blk_nodes = k_end - k_start;
      const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

      const double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] = v_block[idx];
      }

      double zz = 0.0;
      #pragma omp simd reduction(+:zz)
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        zz += vbuf[idx] * vbuf[idx];
      }
      local[num_vecs] += zz;

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t idx = 0; idx < blk_elems; ++idx) {
          sum += uj_block[idx] * vbuf[idx];
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }

  double zz_sum = 0.0;
  for (int t = 0; t < num_threads; ++t) {
    zz_sum += thread_buf[static_cast<size_t>(t) * thread_stride + num_vecs];
  }
  return zz_sum;
}

double fused_dot_zz_v_dyn(const int dof, const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                          std::vector<double>& h_col, double* thread_buf, const int thread_stride,
                          const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj = u.slice_data(j);
        double sum = 0.0;
        for (fsils_int k = k_start; k < k_end; ++k) {
          const size_t base = static_cast<size_t>(k) * static_cast<size_t>(dof);
          double tmp = 0.0;
          for (int d = 0; d < dof; ++d) {
            tmp += uj[base + static_cast<size_t>(d)] * v[base + static_cast<size_t>(d)];
          }
          sum += tmp;
        }
        local[j] += sum;
      }

      double zz = 0.0;
      for (fsils_int k = k_start; k < k_end; ++k) {
        const size_t base = static_cast<size_t>(k) * static_cast<size_t>(dof);
        for (int d = 0; d < dof; ++d) {
          const double val = v[base + static_cast<size_t>(d)];
          zz += val * val;
        }
      }
      local[num_vecs] += zz;
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }

  double zz_sum = 0.0;
  for (int t = 0; t < num_threads; ++t) {
    zz_sum += thread_buf[static_cast<size_t>(t) * thread_stride + num_vecs];
  }
  return zz_sum;
}

double fused_dot_zz_v(const int dof, const fsils_int mynNo, Array3<double>& u, const int i, const Array<double>& u_next,
                      std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  switch (dof) {
    case 1: return fused_dot_zz_v_impl<1>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
    case 2: return fused_dot_zz_v_impl<2>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
    case 3: return fused_dot_zz_v_impl<3>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
    case 4: return fused_dot_zz_v_impl<4>(mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
    default: return fused_dot_zz_v_dyn(dof, mynNo, u, i, u_next, h_col, thread_buf, thread_stride, num_threads);
  }
}

template <int DOF>
double fused_update_norm_v_impl(const fsils_int nNo, const fsils_int mynNo, fe_fsi_linear_solver::FSILS_commuType& commu,
                                Array3<double>& u, const int i, Array<double>& u_next, const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();
  double local_sq = 0.0;

  #pragma omp parallel for reduction(+:local_sq) schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;
    const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

    alignas(64) double vbuf[DOF * BLOCK_SIZE];
    double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      vbuf[idx] = v_block[idx];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] -= hj * uj_block[idx];
      }
    }

    fsils_int owned_nodes = 0;
    if (k_start < mynNo) {
      owned_nodes = std::min(k_end, mynNo) - k_start;
    }

    const size_t owned_elems = static_cast<size_t>(owned_nodes) * static_cast<size_t>(DOF);
    double blk_sq = 0.0;
    #pragma omp simd reduction(+:blk_sq)
    for (size_t idx = 0; idx < owned_elems; ++idx) {
      blk_sq += vbuf[idx] * vbuf[idx];
    }
    local_sq += blk_sq;

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      v_block[idx] = vbuf[idx];
    }
  }

  double global_sq = local_sq;
  if (commu.nTasks != 1) {
    double tmp = 0.0;
    MPI_Allreduce(&local_sq, &tmp, 1, cm_mod::mpreal, MPI_SUM, commu.comm);
    global_sq = tmp;
  }
  return std::sqrt(global_sq);
}

double fused_update_norm_v_dyn(const int dof, const fsils_int nNo, const fsils_int mynNo, fe_fsi_linear_solver::FSILS_commuType& commu,
                               Array3<double>& u, const int i, Array<double>& u_next, const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const size_t block_offset = static_cast<size_t>(k_start) * static_cast<size_t>(dof);
    const size_t blk_elems = static_cast<size_t>(k_end - k_start) * static_cast<size_t>(dof);

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj = u.slice_data(j) + block_offset;
      double* vv = v + block_offset;
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vv[idx] -= hj * uj[idx];
      }
    }
  }

  return norm::fsi_ls_normv(dof, mynNo, commu, u_next);
}

double fused_update_norm_v(const int dof, const fsils_int nNo, const fsils_int mynNo, fe_fsi_linear_solver::FSILS_commuType& commu,
                           Array3<double>& u, const int i, Array<double>& u_next, const std::vector<double>& h_factors)
{
  switch (dof) {
    case 1: return fused_update_norm_v_impl<1>(nNo, mynNo, commu, u, i, u_next, h_factors);
    case 2: return fused_update_norm_v_impl<2>(nNo, mynNo, commu, u, i, u_next, h_factors);
    case 3: return fused_update_norm_v_impl<3>(nNo, mynNo, commu, u, i, u_next, h_factors);
    case 4: return fused_update_norm_v_impl<4>(nNo, mynNo, commu, u, i, u_next, h_factors);
    default: return fused_update_norm_v_dyn(dof, nNo, mynNo, commu, u, i, u_next, h_factors);
  }
}

template <int DOF>
void fused_update_v_inplace_impl(const fsils_int nNo, Array3<double>& u, const int i, Array<double>& u_next,
                                 const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;
    const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

    alignas(64) double vbuf[DOF * BLOCK_SIZE];
    double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      vbuf[idx] = v_block[idx];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] -= hj * uj_block[idx];
      }
    }

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      v_block[idx] = vbuf[idx];
    }
  }
}

void fused_update_v_inplace_dyn(const int dof, const fsils_int nNo, Array3<double>& u, const int i, Array<double>& u_next,
                                const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const size_t block_offset = static_cast<size_t>(k_start) * static_cast<size_t>(dof);
    const size_t blk_elems = static_cast<size_t>(k_end - k_start) * static_cast<size_t>(dof);

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj = u.slice_data(j) + block_offset;
      double* vv = v + block_offset;
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vv[idx] -= hj * uj[idx];
      }
    }
  }
}

void fused_update_v_inplace(const int dof, const fsils_int nNo, Array3<double>& u, const int i, Array<double>& u_next,
                            const std::vector<double>& h_factors)
{
  switch (dof) {
    case 1: fused_update_v_inplace_impl<1>(nNo, u, i, u_next, h_factors); break;
    case 2: fused_update_v_inplace_impl<2>(nNo, u, i, u_next, h_factors); break;
    case 3: fused_update_v_inplace_impl<3>(nNo, u, i, u_next, h_factors); break;
    case 4: fused_update_v_inplace_impl<4>(nNo, u, i, u_next, h_factors); break;
    default: fused_update_v_inplace_dyn(dof, nNo, u, i, u_next, h_factors); break;
  }
}

template <int DOF>
void fused_copy_update_v_inplace_impl(const fsils_int nNo, const double* src, Array3<double>& u, const int i, Array<double>& u_next,
                                      const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;
    const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

    alignas(64) double vbuf[DOF * BLOCK_SIZE];
    const double* src_block = src + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      vbuf[idx] = src_block[idx];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vbuf[idx] -= hj * uj_block[idx];
      }
    }

    double* v_block = v + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      v_block[idx] = vbuf[idx];
    }
  }
}

void fused_copy_update_v_inplace_dyn(const int dof, const fsils_int nNo, const double* src, Array3<double>& u, const int i, Array<double>& u_next,
                                     const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const size_t block_offset = static_cast<size_t>(k_start) * static_cast<size_t>(dof);
    const size_t blk_elems = static_cast<size_t>(k_end - k_start) * static_cast<size_t>(dof);

    const double* src_block = src + block_offset;
    double* vv = v + block_offset;
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      vv[idx] = src_block[idx];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj = u.slice_data(j) + block_offset;
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        vv[idx] -= hj * uj[idx];
      }
    }
  }
}

void fused_copy_update_v_inplace(const int dof, const fsils_int nNo, const double* src, Array3<double>& u, const int i, Array<double>& u_next,
                                 const std::vector<double>& h_factors)
{
  switch (dof) {
    case 1: fused_copy_update_v_inplace_impl<1>(nNo, src, u, i, u_next, h_factors); break;
    case 2: fused_copy_update_v_inplace_impl<2>(nNo, src, u, i, u_next, h_factors); break;
    case 3: fused_copy_update_v_inplace_impl<3>(nNo, src, u, i, u_next, h_factors); break;
    case 4: fused_copy_update_v_inplace_impl<4>(nNo, src, u, i, u_next, h_factors); break;
    default: fused_copy_update_v_inplace_dyn(dof, nNo, src, u, i, u_next, h_factors); break;
  }
}

template <int DOF>
void fused_recon_v_impl(const fsils_int nNo, Array3<double>& u, const int last_i, Array<double>& X, const Vector<double>& y)
{
  const int num_vecs = last_i + 1;
  double* x = X.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;
    const size_t blk_elems = static_cast<size_t>(blk_nodes) * static_cast<size_t>(DOF);

    alignas(64) double xbuf[DOF * BLOCK_SIZE];
    double* x_block = x + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      xbuf[idx] = x_block[idx];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double yj = y(j);
      if (yj == 0.0) {
        continue;
      }

      const double* uj_block = u.slice_data(j) + static_cast<size_t>(k_start) * static_cast<size_t>(DOF);
      #pragma omp simd
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        xbuf[idx] += yj * uj_block[idx];
      }
    }

    #pragma omp simd
    for (size_t idx = 0; idx < blk_elems; ++idx) {
      x_block[idx] = xbuf[idx];
    }
  }
}

void fused_recon_v_dyn(const int dof, const fsils_int nNo, Array3<double>& u, const int last_i, Array<double>& X, const Vector<double>& y)
{
  const int num_vecs = last_i + 1;
  double* x = X.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const size_t block_offset = static_cast<size_t>(k_start) * static_cast<size_t>(dof);
    const size_t blk_elems = static_cast<size_t>(k_end - k_start) * static_cast<size_t>(dof);

    for (int j = 0; j < num_vecs; ++j) {
      const double yj = y(j);
      if (yj == 0.0) {
        continue;
      }

      const double* uj = u.slice_data(j) + block_offset;
      double* xx = x + block_offset;
      for (size_t idx = 0; idx < blk_elems; ++idx) {
        xx[idx] += yj * uj[idx];
      }
    }
  }
}

void fused_recon_v(const int dof, const fsils_int nNo, Array3<double>& u, const int last_i, Array<double>& X, const Vector<double>& y)
{
  switch (dof) {
    case 1: fused_recon_v_impl<1>(nNo, u, last_i, X, y); break;
    case 2: fused_recon_v_impl<2>(nNo, u, last_i, X, y); break;
    case 3: fused_recon_v_impl<3>(nNo, u, last_i, X, y); break;
    case 4: fused_recon_v_impl<4>(nNo, u, last_i, X, y); break;
    default: fused_recon_v_dyn(dof, nNo, u, last_i, X, y); break;
  }
}

void fused_dot_s(const fsils_int mynNo, Array<double>& u, const int i, const Vector<double>& u_next,
                 std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;
    alignas(64) double vbuf[BLOCK_SIZE];

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);
      const fsils_int blk_nodes = k_end - k_start;

      const double* v_block = v + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] = v_block[static_cast<size_t>(idx)];
      }

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
          sum += uj_block[static_cast<size_t>(idx)] * vbuf[idx];
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }
}

double fused_dot_zz_s(const fsils_int mynNo, Array<double>& u, const int i, const Vector<double>& u_next,
                      std::vector<double>& h_col, double* thread_buf, const int thread_stride, const int num_threads)
{
  const int num_vecs = i + 1;
  const double* v = u_next.data();

  std::fill(thread_buf, thread_buf + static_cast<size_t>(num_threads) * thread_stride, 0.0);

  #pragma omp parallel
  {
    const int tid = omp_thread_num();
    double* local = thread_buf + static_cast<size_t>(tid) * thread_stride;
    alignas(64) double vbuf[BLOCK_SIZE];

    #pragma omp for schedule(static)
    for (fsils_int k_start = 0; k_start < mynNo; k_start += BLOCK_SIZE) {
      const fsils_int k_end = std::min(k_start + BLOCK_SIZE, mynNo);
      const fsils_int blk_nodes = k_end - k_start;

      const double* v_block = v + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] = v_block[static_cast<size_t>(idx)];
      }

      double zz = 0.0;
      #pragma omp simd reduction(+:zz)
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        zz += vbuf[idx] * vbuf[idx];
      }
      local[num_vecs] += zz;

      for (int j = 0; j < num_vecs; ++j) {
        const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
          sum += uj_block[static_cast<size_t>(idx)] * vbuf[idx];
        }
        local[j] += sum;
      }
    }
  } // omp parallel

  for (int j = 0; j < num_vecs; ++j) {
    double sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      sum += thread_buf[static_cast<size_t>(t) * thread_stride + j];
    }
    h_col[j] = sum;
  }

  double zz_sum = 0.0;
  for (int t = 0; t < num_threads; ++t) {
    zz_sum += thread_buf[static_cast<size_t>(t) * thread_stride + num_vecs];
  }
  return zz_sum;
}

double fused_update_norm_s(const fsils_int nNo, const fsils_int mynNo, fe_fsi_linear_solver::FSILS_commuType& commu,
                           Array<double>& u, const int i, Vector<double>& u_next, const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();
  double local_sq = 0.0;

  #pragma omp parallel for reduction(+:local_sq) schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;

    alignas(64) double vbuf[BLOCK_SIZE];
    double* v_block = v + static_cast<size_t>(k_start);

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      vbuf[idx] = v_block[static_cast<size_t>(idx)];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] -= hj * uj_block[static_cast<size_t>(idx)];
      }
    }

    fsils_int owned_nodes = 0;
    if (k_start < mynNo) {
      owned_nodes = std::min(k_end, mynNo) - k_start;
    }

    double blk_sq = 0.0;
    #pragma omp simd reduction(+:blk_sq)
    for (fsils_int idx = 0; idx < owned_nodes; ++idx) {
      blk_sq += vbuf[idx] * vbuf[idx];
    }
    local_sq += blk_sq;

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      v_block[static_cast<size_t>(idx)] = vbuf[idx];
    }
  }

  double global_sq = local_sq;
  if (commu.nTasks != 1) {
    double tmp = 0.0;
    MPI_Allreduce(&local_sq, &tmp, 1, cm_mod::mpreal, MPI_SUM, commu.comm);
    global_sq = tmp;
  }
  return std::sqrt(global_sq);
}

void fused_update_s_inplace(const fsils_int nNo, Array<double>& u, const int i, Vector<double>& u_next,
                            const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;

    alignas(64) double vbuf[BLOCK_SIZE];
    double* v_block = v + static_cast<size_t>(k_start);

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      vbuf[idx] = v_block[static_cast<size_t>(idx)];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] -= hj * uj_block[static_cast<size_t>(idx)];
      }
    }

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      v_block[static_cast<size_t>(idx)] = vbuf[idx];
    }
  }
}

void fused_copy_update_s_inplace(const fsils_int nNo, const double* src, Array<double>& u, const int i, Vector<double>& u_next,
                                 const std::vector<double>& h_factors)
{
  const int num_vecs = i + 1;
  double* v = u_next.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;

    alignas(64) double vbuf[BLOCK_SIZE];
    const double* src_block = src + static_cast<size_t>(k_start);

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      vbuf[idx] = src_block[static_cast<size_t>(idx)];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double hj = h_factors[j];
      if (hj == 0.0) {
        continue;
      }

      const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        vbuf[idx] -= hj * uj_block[static_cast<size_t>(idx)];
      }
    }

    double* v_block = v + static_cast<size_t>(k_start);
    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      v_block[static_cast<size_t>(idx)] = vbuf[idx];
    }
  }
}

void fused_recon_s(const fsils_int nNo, Array<double>& u, const int last_i, Vector<double>& X, const Vector<double>& y)
{
  const int num_vecs = last_i + 1;
  double* x = X.data();

  #pragma omp parallel for schedule(static)
  for (fsils_int k_start = 0; k_start < nNo; k_start += BLOCK_SIZE) {
    const fsils_int k_end = std::min(k_start + BLOCK_SIZE, nNo);
    const fsils_int blk_nodes = k_end - k_start;

    alignas(64) double xbuf[BLOCK_SIZE];
    double* x_block = x + static_cast<size_t>(k_start);

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      xbuf[idx] = x_block[static_cast<size_t>(idx)];
    }

    for (int j = 0; j < num_vecs; ++j) {
      const double yj = y(j);
      if (yj == 0.0) {
        continue;
      }

      const double* uj_block = u.col_data(j) + static_cast<size_t>(k_start);
      #pragma omp simd
      for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
        xbuf[idx] += yj * uj_block[static_cast<size_t>(idx)];
      }
    }

    #pragma omp simd
    for (fsils_int idx = 0; idx < blk_nodes; ++idx) {
      x_block[static_cast<size_t>(idx)] = xbuf[idx];
    }
  }
}

inline void iallreduce_sum_inplace(const int n, double* buf, fe_fsi_linear_solver::FSILS_commuType& commu, MPI_Request& req)
{
  if (commu.nTasks > 1) {
    MPI_Iallreduce(MPI_IN_PLACE, buf, n, cm_mod::mpreal, MPI_SUM, commu.comm, &req);
  } else {
    req = MPI_REQUEST_NULL;
  }
}

inline void wait_iallreduce(MPI_Request& req)
{
  if (req != MPI_REQUEST_NULL) {
    MPI_Wait(&req, MPI_STATUS_IGNORE);
  }
}

inline void apply_operator_v(fe_fsi_linear_solver::FSILS_lhsType& lhs, const int dof, const Array<double>& Val,
                             const Array<double>& X, Array<double>& Y, const bool has_coupled_bc, const bool apply_pre)
{
  spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, X, Y);
  add_bc_mul::add_bc_mul(lhs, fe_fsi_linear_solver::BcopType::BCOP_TYPE_ADD, dof, X, Y);
  if (apply_pre && has_coupled_bc) {
    add_bc_mul::add_bc_mul(lhs, fe_fsi_linear_solver::BcopType::BCOP_TYPE_PRE, dof, Y, Y);
  }
}

inline void apply_operator_s(fe_fsi_linear_solver::FSILS_lhsType& lhs, const Vector<double>& Val,
                             const Vector<double>& X, Vector<double>& Y)
{
  spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, Val, X, Y);
}

void gmres_pipelined(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
                     const Array<double>& Val, const Array<double>& R, Array<double>& X, const bool has_coupled_bc)
{
  using namespace fe_fsi_linear_solver;
  const fsils_int nNo = lhs.nNo;
  const fsils_int mynNo = lhs.mynNo;

  ls.ws.ensure_gmres_pipe_v(dof, nNo, ls.sD);
  auto& h = ls.ws.h;
  auto& v = ls.ws.u3;
  auto& z = ls.ws.z3;
  auto& y = ls.ws.y;
  auto& c = ls.ws.c;
  auto& s = ls.ws.s;
  auto& err = ls.ws.err;

  auto& h_col = ls.ws.h_col;
  const int thread_stride = dot_thread_stride(static_cast<int>(h_col.size()));
  const int num_threads = max_omp_threads();
  ls.ws.ensure_gmres_dot_thread(num_threads, thread_stride);
  auto& dot_thread = ls.ws.dot_thread;

  double time = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;
  double eps = 0.0;
  int last_i = 0;
  X = 0.0;

  // Stagnation detection (see gmres_v for rationale).
  double prev_fNorm = 0.0;
  int stagnation_count = 0;
  constexpr int max_stagnation_restarts = 5;
  constexpr double stagnation_ratio = 0.95;

  for (int l = 0; l < ls.mItr; l++) {
    auto v0 = v.rslice(0);

    if (l == 0) {
      v.set_slice(0, R);
    } else {
      apply_operator_v(lhs, dof, Val, X, v0, has_coupled_bc, /*apply_pre*/false);
      ls.itr = ls.itr + 1;
      omp_la::omp_axpby_v(dof, nNo, v0, R, -1.0, v0);
    }

    if (has_coupled_bc) {
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, v0, v0);
    }

    err[0] = norm::fsi_ls_normv(dof, mynNo, lhs.commu, v0);
    if (l == 0) {
      eps = err[0];
      ls.iNorm = eps;
      ls.fNorm = eps;
      eps = std::max(ls.absTol, ls.relTol * eps);
    }

    ls.dB = ls.fNorm;
    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true;
      break;
    }

    omp_la::omp_mul_v(dof, nNo, 1.0 / err[0], v0);

    // Pipeline fill: z0 = A*v0
    auto z0 = z.rslice(0);
    apply_operator_v(lhs, dof, Val, v0, z0, has_coupled_bc, /*apply_pre*/true);
    ls.itr = ls.itr + 1;

    for (int i = 0; i < ls.sD; i++) {
      last_i = i;

      auto zi = z.rslice(i);
      const double zz_local = fused_dot_zz_v(dof, mynNo, v, i, zi, h_col,
                                             dot_thread.data(), thread_stride, num_threads);
      h_col[i+1] = zz_local;

      MPI_Request req{};
      iallreduce_sum_inplace(i + 2, h_col.data(), lhs.commu, req);

      // Compute w = A*zi in the background (only needed if we might continue).
      if (i + 1 < ls.sD) {
        auto znext = z.rslice(i+1);
        apply_operator_v(lhs, dof, Val, zi, znext, has_coupled_bc, /*apply_pre*/true);
        ls.itr = ls.itr + 1;
      }

      wait_iallreduce(req);

      // Compute h(i+1,i) via ||z||^2 - ||h||^2 (single global reduction).
      double tt_sq = h_col[i+1];
      for (int j = 0; j <= i; ++j) {
        tt_sq -= h_col[j] * h_col[j];
      }

      // Roundoff guard (following PETSc PIPEFGMRES behavior).
      const double tt_eps = std::numeric_limits<double>::epsilon();
      if (tt_sq < 0.0 && tt_sq > -tt_eps) {
        tt_sq = 0.0;
      }

      // Square-root breakdown: end this restart cycle early and
      // reconstruct the best solution from the computed subspace.
      if (tt_sq < 0.0) {
        last_i = i - 1;
        break;
      }

      const double tt = std::sqrt(tt_sq);
      for (int j = 0; j <= i; ++j) {
        h(j,i) = h_col[j];
      }
      h(i+1,i) = tt;

      // Happy Breakdown Protection
      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (!(h(i+1,i) > breakdown_tol)) {
        h(i+1,i) = 0.0;
        breakdown = true;
      }

      for (int j = 0; j <= i-1; j++) {
        double tmp_h = c(j) * h(j,i) + s(j) * h(j+1,i);
        h(j+1,i) = -s(j) * h(j,i) + c(j) * h(j+1,i);
        h(j,i) = tmp_h;
      }

      double tmp_hypot = std::hypot(h(i,i), h(i+1,i));
      if (tmp_hypot == 0.0) {
        c(i) = 1.0;
        s(i) = 0.0;
      } else {
        c(i) = h(i,i) / tmp_hypot;
        s(i) = h(i+1,i) / tmp_hypot;
      }
      h(i,i) = tmp_hypot;
      h(i+1,i) = 0.0;

      err(i+1) = -s(i) * err(i);
      err(i) = c(i) * err(i);

      if (std::abs(err(i+1)) < eps) {
        ls.suc = true;
        break;
      }

      if (breakdown) {
        ls.suc = true;
        break;
      }

      if (i + 1 >= ls.sD) {
        continue;
      }

      // Build v(i+1) and z(i+1) (pipeline recursion).
      const double inv_tt = 1.0 / tt;

      auto vnext = v.rslice(i+1);
      fused_copy_update_v_inplace(dof, nNo, zi.data(), v, i, vnext, h_col);
      omp_la::omp_mul_v(dof, nNo, inv_tt, vnext);

      auto znext = z.rslice(i+1);
      fused_update_v_inplace(dof, nNo, z, i, znext, h_col);
      omp_la::omp_mul_v(dof, nNo, inv_tt, znext);
    }

    if (last_i >= ls.sD) {
      last_i = ls.sD - 1;
    }

    for (int i = 0; i <= last_i; i++) {
      y(i) = err(i);
    }

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) {
        y(j) -= h(j,k) * y(k);
      }
      y(j) /= h(j,j);
    }

    fused_recon_v(dof, nNo, v, last_i, X, y);

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) {
      break;
    }

    // Stagnation detection (see gmres_v for rationale).
    if (prev_fNorm > 0.0 && ls.fNorm >= stagnation_ratio * prev_fNorm) {
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }
    prev_fNorm = ls.fNorm;
    if (stagnation_count >= max_stagnation_restarts) {
      break;
    }
  }

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - time + ls.callD;
  ls.dB = 10.0 * std::log(ls.fNorm / ls.dB);
}

void gmres_s_pipelined(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls,
                       const Vector<double>& Val, Vector<double>& R)
{
  using namespace fe_fsi_linear_solver;
  const fsils_int nNo = lhs.nNo;
  const fsils_int mynNo = lhs.mynNo;

  ls.ws.ensure_gmres_pipe_s(nNo, ls.sD);
  auto& h = ls.ws.h;
  auto& v = ls.ws.u2;
  auto& z = ls.ws.z2;
  auto& X = ls.ws.Xs;
  auto& y = ls.ws.y;
  auto& c = ls.ws.c;
  auto& s = ls.ws.s;
  auto& err = ls.ws.err;

  auto& h_col = ls.ws.h_col;
  const int thread_stride = dot_thread_stride(static_cast<int>(h_col.size()));
  const int num_threads = max_omp_threads();
  ls.ws.ensure_gmres_dot_thread(num_threads, thread_stride);
  auto& dot_thread = ls.ws.dot_thread;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;

  double eps = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  ls.iNorm = eps;
  ls.fNorm = eps;
  eps = std::max(ls.absTol, ls.relTol * eps);
  ls.itr = 0;
  int last_i = 0;
  X = 0.0;

  if (ls.iNorm <= ls.absTol) {
    ls.callD = std::numeric_limits<double>::epsilon();
    ls.dB = 0.0;
    return;
  }

  // Stagnation detection (see gmres_v for rationale).
  double prev_fNorm = ls.fNorm;
  int stagnation_count = 0;
  constexpr int max_stagnation_restarts = 5;
  constexpr double stagnation_ratio = 0.95;

  for (int l = 0; l < ls.mItr; l++) {
    ls.dB = ls.fNorm;
    ls.itr++;

    auto v0 = v.rcol(0);
    apply_operator_s(lhs, Val, X, v0);
    omp_la::omp_axpby_s(nNo, v0, R, -1.0, v0);

    err[0] = norm::fsi_ls_norms(mynNo, lhs.commu, v0);
    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true;
      break;
    }

    omp_la::omp_mul_s(nNo, 1.0 / err[0], v0);

    // Pipeline fill: z0 = A*v0
    auto z0 = z.rcol(0);
    apply_operator_s(lhs, Val, v0, z0);

    for (int i = 0; i < ls.sD; i++) {
      ls.itr++;
      last_i = i;

      auto zi = z.rcol(i);
      const double zz_local = fused_dot_zz_s(mynNo, v, i, zi, h_col,
                                             dot_thread.data(), thread_stride, num_threads);
      h_col[i+1] = zz_local;

      MPI_Request req{};
      iallreduce_sum_inplace(i + 2, h_col.data(), lhs.commu, req);

      if (i + 1 < ls.sD) {
        auto znext = z.rcol(i+1);
        apply_operator_s(lhs, Val, zi, znext);
      }

      wait_iallreduce(req);

      double tt_sq = h_col[i+1];
      for (int j = 0; j <= i; ++j) {
        tt_sq -= h_col[j] * h_col[j];
      }

      const double tt_eps = std::numeric_limits<double>::epsilon();
      if (tt_sq < 0.0 && tt_sq > -tt_eps) {
        tt_sq = 0.0;
      }

      if (tt_sq < 0.0) {
        last_i = i - 1;
        break;
      }

      const double tt = std::sqrt(tt_sq);
      for (int j = 0; j <= i; ++j) {
        h(j,i) = h_col[j];
      }
      h(i+1,i) = tt;

      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (!(h(i+1,i) > breakdown_tol)) {
        h(i+1,i) = 0.0;
        breakdown = true;
      }

      for (int j = 0; j <= i-1; j++) {
        double tmp_h = c(j) * h(j,i) + s(j) * h(j+1,i);
        h(j+1,i) = -s(j) * h(j,i) + c(j) * h(j+1,i);
        h(j,i) = tmp_h;
      }

      double tmp_hypot = std::hypot(h(i,i), h(i+1,i));
      if (tmp_hypot == 0.0) {
        c(i) = 1.0;
        s(i) = 0.0;
      } else {
        c(i) = h(i,i) / tmp_hypot;
        s(i) = h(i+1,i) / tmp_hypot;
      }
      h(i,i) = tmp_hypot;
      h(i+1,i) = 0.0;

      err(i+1) = -s(i) * err(i);
      err(i) = c(i) * err(i);

      if (std::abs(err(i+1)) < eps) {
        ls.suc = true;
        break;
      }

      if (breakdown) {
        ls.suc = true;
        break;
      }

      if (i + 1 >= ls.sD) {
        continue;
      }

      const double inv_tt = 1.0 / tt;

      auto vnext = v.rcol(i+1);
      fused_copy_update_s_inplace(nNo, zi.data(), v, i, vnext, h_col);
      omp_la::omp_mul_s(nNo, inv_tt, vnext);

      auto znext = z.rcol(i+1);
      fused_update_s_inplace(nNo, z, i, znext, h_col);
      omp_la::omp_mul_s(nNo, inv_tt, znext);
    }

    if (last_i >= ls.sD) {
      last_i = ls.sD - 1;
    }

    for (int i = 0; i <= last_i; i++) {
      y(i) = err(i);
    }

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) {
        y(j) -= h(j,k) * y(k);
      }
      y(j) /= h(j,j);
    }

    fused_recon_s(nNo, v, last_i, X, y);

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) {
      break;
    }

    // Stagnation detection (see gmres_v for rationale).
    if (prev_fNorm > 0.0 && ls.fNorm >= stagnation_ratio * prev_fNorm) {
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }
    prev_fNorm = ls.fNorm;
    if (stagnation_count >= max_stagnation_restarts) {
      break;
    }
  }

  R = X;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB = 10.0 * std::log(ls.fNorm / ls.dB);
}

void gmres_v_pipelined(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
                       const Array<double>& Val, Array<double>& R, const bool has_coupled_bc)
{
  using namespace fe_fsi_linear_solver;
  const fsils_int nNo = lhs.nNo;
  const fsils_int mynNo = lhs.mynNo;

  ls.ws.ensure_gmres_pipe_v(dof, nNo, ls.sD);
  auto& h = ls.ws.h;
  auto& v = ls.ws.u3;
  auto& z = ls.ws.z3;
  auto& X = ls.ws.X2;
  auto& y = ls.ws.y;
  auto& c = ls.ws.c;
  auto& s = ls.ws.s;
  auto& err = ls.ws.err;

  auto& h_col = ls.ws.h_col;
  const int thread_stride = dot_thread_stride(static_cast<int>(h_col.size()));
  const int num_threads = max_omp_threads();
  ls.ws.ensure_gmres_dot_thread(num_threads, thread_stride);
  auto& dot_thread = ls.ws.dot_thread;

  ls.callD = fe_fsi_linear_solver::fsils_cpu_t();
  ls.suc = false;

  double eps = norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
  ls.iNorm = eps;
  ls.fNorm = eps;
  eps = std::max(ls.absTol, ls.relTol * eps);
  ls.itr = 0;
  int last_i = 0;
  X = 0.0;

  if (ls.iNorm <= ls.absTol) {
    ls.callD = std::numeric_limits<double>::epsilon();
    ls.dB = 0.0;
    return;
  }

  // Stagnation detection (see gmres_v for rationale).
  double prev_fNorm = ls.fNorm;
  int stagnation_count = 0;
  constexpr int max_stagnation_restarts = 5;
  constexpr double stagnation_ratio = 0.95;

  for (int l = 0; l < ls.mItr; l++) {
    ls.dB = ls.fNorm;
    ls.itr++;

    auto v0 = v.rslice(0);
    apply_operator_v(lhs, dof, Val, X, v0, has_coupled_bc, /*apply_pre*/false);
    omp_la::omp_axpby_v(dof, nNo, v0, R, -1.0, v0);

    if (has_coupled_bc) {
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, v0, v0);
    }

    err[0] = norm::fsi_ls_normv(dof, mynNo, lhs.commu, v0);
    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true;
      break;
    }

    omp_la::omp_mul_v(dof, nNo, 1.0 / err[0], v0);

    // Pipeline fill: z0 = A*v0
    auto z0 = z.rslice(0);
    apply_operator_v(lhs, dof, Val, v0, z0, has_coupled_bc, /*apply_pre*/true);

    for (int i = 0; i < ls.sD; i++) {
      ls.itr++;
      last_i = i;

      auto zi = z.rslice(i);
      const double zz_local = fused_dot_zz_v(dof, mynNo, v, i, zi, h_col,
                                             dot_thread.data(), thread_stride, num_threads);
      h_col[i+1] = zz_local;

      MPI_Request req{};
      iallreduce_sum_inplace(i + 2, h_col.data(), lhs.commu, req);

      if (i + 1 < ls.sD) {
        auto znext = z.rslice(i+1);
        apply_operator_v(lhs, dof, Val, zi, znext, has_coupled_bc, /*apply_pre*/true);
      }

      wait_iallreduce(req);

      double tt_sq = h_col[i+1];
      for (int j = 0; j <= i; ++j) {
        tt_sq -= h_col[j] * h_col[j];
      }

      const double tt_eps = std::numeric_limits<double>::epsilon();
      if (tt_sq < 0.0 && tt_sq > -tt_eps) {
        tt_sq = 0.0;
      }

      if (tt_sq < 0.0) {
        last_i = i - 1;
        break;
      }

      const double tt = std::sqrt(tt_sq);
      for (int j = 0; j <= i; ++j) {
        h(j,i) = h_col[j];
      }
      h(i+1,i) = tt;

      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (!(h(i+1,i) > breakdown_tol)) {
        h(i+1,i) = 0.0;
        breakdown = true;
      }

      for (int j = 0; j <= i-1; j++) {
        double tmp_h = c(j) * h(j,i) + s(j) * h(j+1,i);
        h(j+1,i) = -s(j) * h(j,i) + c(j) * h(j+1,i);
        h(j,i) = tmp_h;
      }

      double tmp_hypot = std::hypot(h(i,i), h(i+1,i));
      if (tmp_hypot == 0.0) {
        c(i) = 1.0;
        s(i) = 0.0;
      } else {
        c(i) = h(i,i) / tmp_hypot;
        s(i) = h(i+1,i) / tmp_hypot;
      }
      h(i,i) = tmp_hypot;
      h(i+1,i) = 0.0;

      err(i+1) = -s(i) * err(i);
      err(i) = c(i) * err(i);

      if (std::abs(err(i+1)) < eps) {
        ls.suc = true;
        break;
      }

      if (breakdown) {
        ls.suc = true;
        break;
      }

      if (i + 1 >= ls.sD) {
        continue;
      }

      const double inv_tt = 1.0 / tt;

      auto vnext = v.rslice(i+1);
      fused_copy_update_v_inplace(dof, nNo, zi.data(), v, i, vnext, h_col);
      omp_la::omp_mul_v(dof, nNo, inv_tt, vnext);

      auto znext = z.rslice(i+1);
      fused_update_v_inplace(dof, nNo, z, i, znext, h_col);
      omp_la::omp_mul_v(dof, nNo, inv_tt, znext);
    }

    if (last_i >= ls.sD) {
      last_i = ls.sD - 1;
    }

    for (int i = 0; i <= last_i; i++) {
      y(i) = err(i);
    }

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) {
        y(j) -= h(j,k) * y(k);
      }
      y(j) /= h(j,j);
    }

    fused_recon_v(dof, nNo, v, last_i, X, y);

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) {
      break;
    }

    // Stagnation detection (see gmres_v for rationale).
    if (prev_fNorm > 0.0 && ls.fNorm >= stagnation_ratio * prev_fNorm) {
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }
    prev_fNorm = ls.fNorm;
    if (stagnation_count >= max_stagnation_restarts) {
      break;
    }
  }

  R = X;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB = 10.0 * std::log(ls.fNorm / ls.dB);
}

} // namespace

/// @brief Pre-calculates the boundary condition normalization factor.
/// Removed O(N) heap allocation, using fast local accumulations.
void bc_pre(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
            const int mynNo, const int nNo)
{
  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto &face = lhs.face[faIn];

    if (face.coupledFlag) {
      // Use the face's own DOF count (set by the caller based on which
      // components participate in this coupled BC), clamped to the
      // system DOF count for safety.
      const int face_dof = std::min(face.dof, dof);

      if (face.sharedFlag) {
        double local_nS = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          if (Ac < mynNo) {
            for (int i = 0; i < face_dof; i++) {
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
          for (int i = 0; i < face_dof; i++) {
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
  const bool has_coupled_bc = std::any_of(lhs.face.begin(), lhs.face.end(),
      [](const auto& face) { return face.coupledFlag; });

  if (ls.pipelined_gmres) {
    gmres_pipelined(lhs, ls, dof, Val, R, X, has_coupled_bc);
    return;
  }

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

  auto& h_col = ls.ws.h_col;
  const int thread_stride = dot_thread_stride(static_cast<int>(h_col.size()));
  const int num_threads = max_omp_threads();
  ls.ws.ensure_gmres_dot_thread(num_threads, thread_stride);
  auto& dot_thread = ls.ws.dot_thread;

  // Stagnation detection (see gmres_v for rationale).
  double prev_fNorm = 0.0;
  int stagnation_count = 0;
  constexpr int max_stagnation_restarts = 5;
  constexpr double stagnation_ratio = 0.95;

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

    if (has_coupled_bc) {
      auto unCondU_ref = u.rslice(0);
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU_ref, u_slice);
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

      if (has_coupled_bc) {
        auto unCondU_ref = u.rslice(i+1);
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU_ref, u_slice_next);
      }

      // ----------------------------------------------------------------------
      // CGS2: Classical Gram-Schmidt with Pythagorean norm trick
      // Computes h[0..i] + self-dot in one fused pass, then recovers
      // ||w_orth||^2 = ||w||^2 - sum(h^2) without a second MPI_Allreduce.
      // ----------------------------------------------------------------------
      const double zz_local = fused_dot_zz_v(dof, mynNo, u, i, u_slice_next, h_col,
                                              dot_thread.data(), thread_stride, num_threads);
      h_col[static_cast<size_t>(i+1)] = zz_local;
      bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);

      double proj_sq_norm = 0.0;
      for (int j = 0; j <= i; j++) {
        h(j,i) = h_col[j];
        proj_sq_norm += h_col[j] * h_col[j];
      }

      double tt_sq = h_col[static_cast<size_t>(i+1)] - proj_sq_norm;

      const double tt_eps = std::numeric_limits<double>::epsilon();
      if (tt_sq < 0.0 && tt_sq > -tt_eps * h_col[static_cast<size_t>(i+1)]) {
        tt_sq = 0.0;
      }

      fused_update_v_inplace(dof, nNo, u, i, u_slice_next, h_col);

      // CGS2: Selective reorthogonalization (Pythagorean check)
      if (tt_sq < proj_sq_norm && tt_sq >= 0.0) {
        const double zz2_local = fused_dot_zz_v(dof, mynNo, u, i, u_slice_next, h_col,
                                                  dot_thread.data(), thread_stride, num_threads);
        h_col[static_cast<size_t>(i+1)] = zz2_local;
        bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);

        double corr_sq = 0.0;
        for (int j = 0; j <= i; ++j) {
          h(j,i) += h_col[j];
          corr_sq += h_col[j] * h_col[j];
        }

        fused_update_v_inplace(dof, nNo, u, i, u_slice_next, h_col);

        tt_sq = h_col[static_cast<size_t>(i+1)] - corr_sq;
        if (tt_sq < 0.0 && tt_sq > -tt_eps * h_col[static_cast<size_t>(i+1)]) {
          tt_sq = 0.0;
        }
      } else if (tt_sq < 0.0) {
        tt_sq = 0.0;
      }

      double new_norm = std::sqrt(tt_sq);
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

    fused_recon_v(dof, nNo, u, last_i, X, y);

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) break;

    // Stagnation detection (see gmres_v for rationale).
    if (prev_fNorm > 0.0 && ls.fNorm >= stagnation_ratio * prev_fNorm) {
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }
    prev_fNorm = ls.fNorm;
    if (stagnation_count >= max_stagnation_restarts) {
      break;
    }
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

  if (ls.pipelined_gmres) {
    gmres_s_pipelined(lhs, ls, Val, R);
    return;
  }

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

  auto& h_col = ls.ws.h_col;
  const int thread_stride = dot_thread_stride(static_cast<int>(h_col.size()));
  const int num_threads = max_omp_threads();
  ls.ws.ensure_gmres_dot_thread(num_threads, thread_stride);
  auto& dot_thread = ls.ws.dot_thread;

  // Stagnation detection (see gmres_v for rationale).
  double prev_fNorm = ls.fNorm;
  int stagnation_count = 0;
  constexpr int max_stagnation_restarts = 5;
  constexpr double stagnation_ratio = 0.95;

  for (int l = 0; l < ls.mItr; l++) {
    ls.dB = ls.fNorm;
    ls.itr++;

    auto u_col_curr = u.rcol(0);

    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, Val, X, u_col_curr);
    omp_la::omp_axpby_s(nNo, u_col_curr, R, -1.0, u_col_curr);

    err[0] = norm::fsi_ls_norms(mynNo, lhs.commu, u_col_curr);
    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true; break;
    }

    omp_la::omp_mul_s(nNo, 1.0 / err[0], u_col_curr);

    for (int i = 0; i < ls.sD; i++) {
      ls.itr++;
      last_i = i;

      auto u_col_prev = u.rcol(i);
      auto u_col_next = u.rcol(i+1);
      spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, Val, u_col_prev, u_col_next);

      // --- CGS Step 1 with Pythagorean trick ---
      const double zz_local = fused_dot_zz_s(mynNo, u, i, u_col_next, h_col,
                                              dot_thread.data(), thread_stride, num_threads);
      h_col[static_cast<size_t>(i+1)] = zz_local;
      bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);

      double proj_sq_norm = 0.0;
      for (int j = 0; j <= i; j++) {
        h(j,i) = h_col[j];
        proj_sq_norm += h_col[j] * h_col[j];
      }

      double tt_sq = h_col[static_cast<size_t>(i+1)] - proj_sq_norm;

      const double tt_eps = std::numeric_limits<double>::epsilon();
      if (tt_sq < 0.0 && tt_sq > -tt_eps * h_col[static_cast<size_t>(i+1)]) {
        tt_sq = 0.0;
      }

      fused_update_s_inplace(nNo, u, i, u_col_next, h_col);

      // CGS2: Selective reorthogonalization (Pythagorean check)
      if (tt_sq < proj_sq_norm && tt_sq >= 0.0) {
        const double zz2_local = fused_dot_zz_s(mynNo, u, i, u_col_next, h_col,
                                                  dot_thread.data(), thread_stride, num_threads);
        h_col[static_cast<size_t>(i+1)] = zz2_local;
        bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);

        double corr_sq = 0.0;
        for (int j = 0; j <= i; ++j) {
          h(j,i) += h_col[j];
          corr_sq += h_col[j] * h_col[j];
        }

        fused_update_s_inplace(nNo, u, i, u_col_next, h_col);

        tt_sq = h_col[static_cast<size_t>(i+1)] - corr_sq;
        if (tt_sq < 0.0 && tt_sq > -tt_eps * h_col[static_cast<size_t>(i+1)]) {
          tt_sq = 0.0;
        }
      } else if (tt_sq < 0.0) {
        tt_sq = 0.0;
      }

      double new_norm = std::sqrt(tt_sq);
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

    fused_recon_s(nNo, u, last_i, X, y);

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) break;

    // Stagnation detection (see gmres_v for rationale).
    if (prev_fNorm > 0.0 && ls.fNorm >= stagnation_ratio * prev_fNorm) {
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }
    prev_fNorm = ls.fNorm;
    if (stagnation_count >= max_stagnation_restarts) {
      break;
    }
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
  const bool has_coupled_bc = std::any_of(lhs.face.begin(), lhs.face.end(),
      [](const auto& face) { return face.coupledFlag; });

  if (ls.pipelined_gmres) {
    bc_pre(lhs, ls, dof, mynNo, nNo);
    gmres_v_pipelined(lhs, ls, dof, Val, R, has_coupled_bc);
    return;
  }

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

  // ===== TIMING PROFILE =====
  double tp_spmv = 0.0, tp_bc_mul = 0.0, tp_dot_gs = 0.0;
  double tp_allreduce = 0.0, tp_gs_update = 0.0, tp_norm = 0.0;
  double tp_vecops = 0.0, tp_givens = 0.0, tp_recon = 0.0;
  int tp_reorth_count = 0, tp_restarts = 0;
  auto TP = [](){ return fe_fsi_linear_solver::fsils_cpu_t(); };
  double tp0;
  // ===========================

  tp0 = TP();
  double eps = norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
  tp_norm += TP() - tp0;

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

  auto& h_col = ls.ws.h_col;
  const int thread_stride = dot_thread_stride(static_cast<int>(h_col.size()));
  const int num_threads = max_omp_threads();
  ls.ws.ensure_gmres_dot_thread(num_threads, thread_stride);
  auto& dot_thread = ls.ws.dot_thread;

  // Stagnation detection: terminate early if residual stops improving.
  // With restarted GMRES and a weak preconditioner, stagnation wastes
  // thousands of iterations (e.g. 51000 on an ill-conditioned system).
  double prev_fNorm = ls.fNorm;
  int stagnation_count = 0;
  constexpr int max_stagnation_restarts = 5;
  constexpr double stagnation_ratio = 0.95;

  for (int l = 0; l < ls.mItr; l++) {
    ls.dB = ls.fNorm;
    ls.itr++;
    if (l > 0) tp_restarts++;

    auto u_slice = u.rslice(0);

    tp0 = TP();
    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, X, u_slice);
    tp_spmv += TP() - tp0;

    tp0 = TP();
    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, X, u_slice);
    tp_bc_mul += TP() - tp0;

    tp0 = TP();
    omp_la::omp_axpby_v(dof, nNo, u_slice, R, -1.0, u_slice);
    tp_vecops += TP() - tp0;

    // Note: BCOP_TYPE_PRE is NOT applied in the standard GMRESV path.

    tp0 = TP();
    err[0] = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u.rslice(0));
    tp_norm += TP() - tp0;

    if (std::abs(err[0]) <= std::numeric_limits<double>::epsilon()) {
      ls.suc = true; break;
    }

    tp0 = TP();
    omp_la::omp_mul_v(dof, nNo, 1.0 / err[0], u_slice);
    tp_vecops += TP() - tp0;

    for (int i = 0; i < ls.sD; i++) {
      ls.itr++;
      last_i = i;
      auto u_slice_prev = u.rslice(i);
      auto u_slice_next = u.rslice(i+1);

      tp0 = TP();
      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, Val, u_slice_prev, u_slice_next);
      tp_spmv += TP() - tp0;

      tp0 = TP();
      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, u_slice_prev, u_slice_next);
      tp_bc_mul += TP() - tp0;

      // --- CGS Step 1 with Pythagorean trick ---
      tp0 = TP();
      const double zz_local = fused_dot_zz_v(dof, mynNo, u, i, u_slice_next, h_col,
                                              dot_thread.data(), thread_stride, num_threads);
      h_col[static_cast<size_t>(i+1)] = zz_local;
      tp_dot_gs += TP() - tp0;

      tp0 = TP();
      bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);
      tp_allreduce += TP() - tp0;

      double proj_sq_norm = 0.0;
      for (int j = 0; j <= i; j++) {
        h(j,i) = h_col[j];
        proj_sq_norm += h_col[j] * h_col[j];
      }

      double tt_sq = h_col[static_cast<size_t>(i+1)] - proj_sq_norm;

      // Roundoff guard: small negative values from floating-point cancellation
      const double tt_eps = std::numeric_limits<double>::epsilon();
      if (tt_sq < 0.0 && tt_sq > -tt_eps * h_col[static_cast<size_t>(i+1)]) {
        tt_sq = 0.0;
      }

      // Subtract projections in-place (no MPI needed)
      tp0 = TP();
      fused_update_v_inplace(dof, nNo, u, i, u_slice_next, h_col);
      tp_gs_update += TP() - tp0;

      // CGS2: Selective reorthogonalization (Pythagorean check)
      // tt_sq < proj_sq_norm means ||v_new|| < (1/sqrt(2))||v|| — loss of orthogonality.
      if (tt_sq < proj_sq_norm && tt_sq >= 0.0) {
        tp_reorth_count++;

        tp0 = TP();
        const double zz2_local = fused_dot_zz_v(dof, mynNo, u, i, u_slice_next, h_col,
                                                  dot_thread.data(), thread_stride, num_threads);
        h_col[static_cast<size_t>(i+1)] = zz2_local;
        tp_dot_gs += TP() - tp0;

        tp0 = TP();
        bcast::fsils_bcast_v(i + 2, h_col, lhs.commu);
        tp_allreduce += TP() - tp0;

        double corr_sq = 0.0;
        for (int j = 0; j <= i; ++j) {
          h(j,i) += h_col[j];
          corr_sq += h_col[j] * h_col[j];
        }

        tp0 = TP();
        fused_update_v_inplace(dof, nNo, u, i, u_slice_next, h_col);
        tp_gs_update += TP() - tp0;

        tt_sq = h_col[static_cast<size_t>(i+1)] - corr_sq;
        if (tt_sq < 0.0 && tt_sq > -tt_eps * h_col[static_cast<size_t>(i+1)]) {
          tt_sq = 0.0;
        }
      } else if (tt_sq < 0.0) {
        // True breakdown: Pythagorean identity completely fails
        tt_sq = 0.0;
      }

      double new_norm = std::sqrt(tt_sq);
      h(i+1,i) = new_norm;

      // Happy Breakdown Protection
      bool breakdown = false;
      const double breakdown_tol = std::numeric_limits<double>::epsilon() * std::max(ls.iNorm, 1.0) * 1e2;
      if (h(i+1,i) > breakdown_tol) {
        tp0 = TP();
        omp_la::omp_mul_v(dof, nNo, 1.0/h(i+1,i), u_slice_next);
        tp_vecops += TP() - tp0;
      } else {
        h(i+1,i) = 0.0;
        breakdown = true;
      }

      tp0 = TP();
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
      tp_givens += TP() - tp0;

      if (std::abs(err(i+1)) < eps || breakdown) {
        ls.suc = true;
        break;
      }
    }

    if (last_i >= ls.sD) last_i = ls.sD - 1;

    tp0 = TP();
    for (int i = 0; i <= last_i; i++) y(i) = err(i);

    for (int j = last_i; j >= 0; j--) {
      for (int k = j+1; k <= last_i; k++) y(j) -= h(j,k)*y(k);
      y(j) /= h(j,j);
    }

    fused_recon_v(dof, nNo, u, last_i, X, y);
    tp_recon += TP() - tp0;

    ls.fNorm = std::abs(err(last_i+1));
    if (ls.suc) break;

    // Stagnation detection: if residual hasn't improved by at least
    // (1 - stagnation_ratio) for max_stagnation_restarts consecutive
    // restarts, the preconditioner is too weak for this system.
    if (prev_fNorm > 0.0 && ls.fNorm >= stagnation_ratio * prev_fNorm) {
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }
    prev_fNorm = ls.fNorm;
    if (stagnation_count >= max_stagnation_restarts) {
      break;
    }
  }

  R = X;
  ls.callD = fe_fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB  = 10.0 * std::log(ls.fNorm / ls.dB);

  // ===== PRINT TIMING PROFILE =====
  double tp_total = tp_spmv + tp_bc_mul + tp_dot_gs + tp_allreduce +
                    tp_gs_update + tp_norm + tp_vecops + tp_givens + tp_recon;
  if (lhs.commu.task == 0 && tp_total > 0.0) {
    auto pct = [&](double t) { return 100.0 * t / tp_total; };
    fprintf(stderr,
      "\n=== GMRES_V TIMING PROFILE (rank 0) ===\n"
      "  Total GMRES time:     %10.6f s  (iters=%d, restarts=%d, reorth=%d)\n"
      "  SpMV:                 %10.6f s  (%5.1f%%)\n"
      "  BC multiply:          %10.6f s  (%5.1f%%)\n"
      "  GS dot products:      %10.6f s  (%5.1f%%)\n"
      "  MPI AllReduce:        %10.6f s  (%5.1f%%)\n"
      "  GS vector update:     %10.6f s  (%5.1f%%)\n"
      "  Norm computation:     %10.6f s  (%5.1f%%)\n"
      "  Vector ops (scale):   %10.6f s  (%5.1f%%)\n"
      "  Givens rotation:      %10.6f s  (%5.1f%%)\n"
      "  Reconstruction:       %10.6f s  (%5.1f%%)\n"
      "  dof=%d  nNo=%lld  mynNo=%lld  nnz=%lld  sD=%d\n"
      "========================================\n",
      tp_total, ls.itr, tp_restarts, tp_reorth_count,
      tp_spmv, pct(tp_spmv),
      tp_bc_mul, pct(tp_bc_mul),
      tp_dot_gs, pct(tp_dot_gs),
      tp_allreduce, pct(tp_allreduce),
      tp_gs_update, pct(tp_gs_update),
      tp_norm, pct(tp_norm),
      tp_vecops, pct(tp_vecops),
      tp_givens, pct(tp_givens),
      tp_recon, pct(tp_recon),
      dof, (long long)nNo, (long long)mynNo, (long long)lhs.nnz, ls.sD);
  }
  // ==================================
}

}; // namespace gmres

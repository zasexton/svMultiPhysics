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

// For calculating the norm of a scaler or vector based vector.
//
//
// Only the part of U which is owned by this processor is included in
// norm calculation, i.e. U(:,1:cS(tF)%ptr+cS(tF)%n-1)
// In order to have the correct answer it is needed that COMMU has
// been done before calling this function (or the ansesters of U
// are passed through COMMU)

#include "norm.h"

#include "CmMod.h"

#include "mpi.h"

#include <math.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace norm {

// ---------------------------------------------------------------------------
// Deterministic parallel squared-norm helper.
// Same strategy as deterministic_dot: per-thread partials summed in
// fixed thread-ID order for bit-reproducible results.
// ---------------------------------------------------------------------------
namespace {

inline double deterministic_norm_sq(const double* __restrict__ u, fsils_int n)
{
#ifdef _OPENMP
    if (n >= 10000 && omp_get_max_threads() > 1) {
        const int max_t = omp_get_max_threads();
        constexpr int kStackMax = 64;
        double stack_partials[kStackMax];
        double* partials = (max_t <= kStackMax) ? stack_partials : new double[max_t];
        for (int t = 0; t < max_t; ++t) partials[t] = 0.0;

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double local_sum = 0.0;
            #pragma omp for schedule(static)
            for (fsils_int i = 0; i < n; i++) {
                local_sum += u[i] * u[i];
            }
            partials[tid] = local_sum;
        }

        double result = 0.0;
        for (int t = 0; t < max_t; ++t) {
            result += partials[t];
        }
        if (max_t > kStackMax) delete[] partials;
        return result;
    }
#endif
    double result = 0.0;
    #pragma omp simd reduction(+:result)
    for (fsils_int i = 0; i < n; i++) {
        result += u[i] * u[i];
    }
    return result;
}

} // namespace

double fsi_ls_norm_sq_local_s(const fsils_int nNo, const Vector<double>& U)
{
  return deterministic_norm_sq(U.data(), nNo);
}

double fsi_ls_norm_sq_local_v(const int dof, const fsils_int nNo, const Array<double>& U)
{
  const fsils_int n = static_cast<fsils_int>(dof) * nNo;
  return deterministic_norm_sq(U.data(), n);
}

double fsi_ls_norms(const fsils_int nNo, FSILS_commuType& commu, const Vector<double>& U)
{
  double result = fsi_ls_norm_sq_local_s(nNo, U);

  if (commu.nTasks != 1) {
    double tmp;
    fsils_allreduce_sum(&result, &tmp, 1, cm_mod::mpreal, commu);
    result = tmp;
  }

  return sqrt(result);
}

double fsi_ls_normv(const int dof, const fsils_int nNo, FSILS_commuType& commu, const Array<double>& U)
{
  double result = fsi_ls_norm_sq_local_v(dof, nNo, U);

  if (commu.nTasks != 1) {
    double tmp;
    fsils_allreduce_sum(&result, &tmp, 1, cm_mod::mpreal, commu);
    result = tmp;
  }

  return sqrt(result);
}

};

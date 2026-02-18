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

// A bunch of operation that benefits from OMP hyperthreading

#include "omp_la.h"

namespace omp_la {

using fe_fsi_linear_solver::fsils_int;

/// @brief Reproduces 'SUBROUTINE OMPMULS (nNo, r, U)'.
//
void omp_mul_s(const fsils_int nNo, const double r, Vector<double>& U)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; i++) {
    U(i) = r * U(i);
  }
}

//====================================================================
// Templated omp_mul_v
//====================================================================

template <int DOF>
static void omp_mul_v_impl(const fsils_int nNo, const double r, Array<double>& U)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; i++) {
    for (int j = 0; j < DOF; j++) {
      U(j,i) = r * U(j,i);
    }
  }
}

static void omp_mul_v_dyn(const fsils_int nNo, const double r, Array<double>& U)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; i++) {
    for (int j = 0; j < U.nrows(); j++) {
      U(j,i) = r * U(j,i);
    }
  }
}

void omp_mul_v(const int dof, const fsils_int nNo, const double r, Array<double>& U)
{
  switch (dof) {
    case 1: omp_mul_v_impl<1>(nNo, r, U); break;
    case 2: omp_mul_v_impl<2>(nNo, r, U); break;
    case 3: omp_mul_v_impl<3>(nNo, r, U); break;
    case 4: omp_mul_v_impl<4>(nNo, r, U); break;
    default: omp_mul_v_dyn(nNo, r, U); break;
  }
}

/// @brief Reproduces 'SUBROUTINE OMPSUMS (nNo, r, U, V)'.
//
void omp_sum_s(const fsils_int nNo, const double r, Vector<double>& U, const Vector<double>& V)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; i++) {
    U(i) = U(i) + r*V(i);
  }
}

//====================================================================
// Templated omp_sum_v
//====================================================================

template <int DOF>
static void omp_sum_v_impl(const fsils_int nNo, const double r, Array<double>& U, const Array<double>& V)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; i++) {
    for (int j = 0; j < DOF; j++) {
      U(j,i) = U(j,i) + r * V(j,i);
    }
  }
}

static void omp_sum_v_dyn(const fsils_int nNo, const double r, Array<double>& U, const Array<double>& V)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; i++) {
    for (int j = 0; j < U.nrows(); j++) {
      U(j,i) = U(j,i) + r * V(j,i);
    }
  }
}

void omp_sum_v(const int dof, const fsils_int nNo, const double r, Array<double>& U, const Array<double>& V)
{
  switch (dof) {
    case 1: omp_sum_v_impl<1>(nNo, r, U, V); break;
    case 2: omp_sum_v_impl<2>(nNo, r, U, V); break;
    case 3: omp_sum_v_impl<3>(nNo, r, U, V); break;
    case 4: omp_sum_v_impl<4>(nNo, r, U, V); break;
    default: omp_sum_v_dyn(nNo, r, U, V); break;
  }
}

//====================================================================
// Fused vector operations
//====================================================================

/// @brief Y = A + alpha*B (vector version)
void omp_axpby_v(const int dof, const fsils_int nNo, Array<double>& Y,
                 const Array<double>& A, const double alpha, const Array<double>& B)
{
  fsils_int n = dof * nNo;
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < n; i++) {
    Y(i) = A(i) + alpha * B(i);
  }
}

/// @brief Y = A + alpha*B + beta*C (vector version)
void omp_axpbypgz_v(const int dof, const fsils_int nNo, Array<double>& Y,
                     const Array<double>& A, const double alpha, const Array<double>& B,
                     const double beta, const Array<double>& C)
{
  fsils_int n = dof * nNo;
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < n; i++) {
    Y(i) = A(i) + alpha * B(i) + beta * C(i);
  }
}

/// @brief Y = A + alpha*B (scalar version)
void omp_axpby_s(const fsils_int nNo, Vector<double>& Y,
                 const Vector<double>& A, const double alpha, const Vector<double>& B)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; i++) {
    Y(i) = A(i) + alpha * B(i);
  }
}

/// @brief Y = A + alpha*B + beta*C (scalar version)
void omp_axpbypgz_s(const fsils_int nNo, Vector<double>& Y,
                     const Vector<double>& A, const double alpha, const Vector<double>& B,
                     const double beta, const Vector<double>& C)
{
  #pragma omp parallel for schedule(static)
  for (fsils_int i = 0; i < nNo; i++) {
    Y(i) = A(i) + alpha * B(i) + beta * C(i);
  }
}

};

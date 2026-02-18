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

#include "fils_struct.hpp"

namespace omp_la {

using namespace fe_fsi_linear_solver;

void omp_mul_s(const fsils_int nNo, const double r, Vector<double>& U);

void omp_mul_v(const int dof, const fsils_int nNo, const double r, Array<double>& U);

void omp_sum_s(const fsils_int nNo, const double r, Vector<double>& U, const Vector<double>& V);

void omp_sum_v(const int dof, const fsils_int nNo, const double r, Array<double>& U, const Array<double>& V);

/// @brief Y = A + alpha*B (vector version)
void omp_axpby_v(const int dof, const fsils_int nNo, Array<double>& Y,
                 const Array<double>& A, const double alpha, const Array<double>& B);

/// @brief Y = A + alpha*B + beta*C (vector version)
void omp_axpbypgz_v(const int dof, const fsils_int nNo, Array<double>& Y,
                     const Array<double>& A, const double alpha, const Array<double>& B,
                     const double beta, const Array<double>& C);

/// @brief Y = A + alpha*B (scalar version)
void omp_axpby_s(const fsils_int nNo, Vector<double>& Y,
                 const Vector<double>& A, const double alpha, const Vector<double>& B);

/// @brief Y = A + alpha*B + beta*C (scalar version)
void omp_axpbypgz_s(const fsils_int nNo, Vector<double>& Y,
                     const Vector<double>& A, const double alpha, const Vector<double>& B,
                     const double beta, const Vector<double>& C);

};

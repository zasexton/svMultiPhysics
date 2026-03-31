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

// This subroutine solve a linear system of equations AX=B using
// Gauss elimination and replaces the B with X

#include "ge.h"

#include <Eigen/Dense>

#include <cmath>
#include <limits>

namespace ge {

bool ge(const int nV, const int N, const Array<double>& A, Vector<double>& B)
{
  (void)nV;

  if (N <= 0) {
    return false;
  }

  // Constructing a preconditioner. This is to prevent latter problem
  // with singular matrix

  Vector<double> W(N);
  double tol = std::numeric_limits<double>::denorm_min();
  double eps = std::numeric_limits<double>::epsilon();

  // Symmetric diagonal scaling keeps the tiny dense solve well-conditioned
  // while the rank-revealing solve handles dependent GCR basis vectors.
  for (int i = 0; i < N; i++) {
    const double aii = std::abs(A(i, i));
    W(i) = (aii < tol) ? 1.0 : (1.0 / std::sqrt(aii));
  }

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N, N);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(N);
  double max_abs = 0.0;
  for (int i = 0; i < N; ++i) {
    rhs(i) = W(i) * B(i);
    max_abs = std::max(max_abs, std::abs(rhs(i)));
    for (int j = 0; j < N; ++j) {
      M(i, j) = W(i) * W(j) * A(i, j);
      max_abs = std::max(max_abs, std::abs(M(i, j)));
    }
  }

  if (!(max_abs > 0.0) || !std::isfinite(max_abs)) {
    B = 0.0;
    return false;
  }

  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(M);
  cod.setThreshold(std::sqrt(eps) * max_abs);
  const int rank = cod.rank();
  if (rank <= 0) {
    B = 0.0;
    return false;
  }

  const Eigen::VectorXd x_scaled = cod.solve(rhs);
  if ((x_scaled.array().isFinite().all() == false)) {
    B = 0.0;
    return false;
  }

  Eigen::VectorXd residual = M * x_scaled - rhs;
  if ((residual.array().isFinite().all() == false)) {
    B = 0.0;
    return false;
  }

  const double rhs_norm = rhs.norm();
  const double residual_norm = residual.norm();
  if (rhs_norm > 0.0 && residual_norm > 1e6 * rhs_norm) {
    B = 0.0;
    return false;
  }

  for (int i = 0; i < N; i++) {
    B(i) = W(i) * x_scaled(i);
  }

  return true;
}

};


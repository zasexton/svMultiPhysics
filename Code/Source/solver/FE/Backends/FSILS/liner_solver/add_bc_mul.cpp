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

#include "add_bc_mul.h"

#include <vector>

namespace add_bc_mul {

/// @brief The contribution of coupled BCs is added to the matrix-vector
/// product operation. Depending on the type of operation (adding the
/// contribution or computing the PC contribution) different
/// coefficients are used.
///
/// For reference, see 
/// Moghadam et al. 2013 eq. 27 (https://doi.org/10.1016/j.jcp.2012.07.035) and
/// Moghadam et al. 2013b (https://doi.org/10.1007/s00466-013-0868-1).
///
/// Reproduces code in ADDBCMUL.f.
/// @param lhs The left-hand side of the linear system. 0D resistance is stored in the face(i).res field.
/// @param op_Type The type of operation (addition or PC contribution)
/// @param dof The number of degrees of freedom.
/// @param X The input vector.
/// @param Y The current matrix-vector product (Y = K*X), to which we add K^BC * X = res * v * v^T * X.
/// The expression is slightly different if preconditioning.
void add_bc_mul(FSILS_lhsType& lhs, const BcopType op_Type, const int dof, const Array<double>& X, Array<double>& Y)
{
  thread_local std::vector<int> shared_face_indices;
  thread_local std::vector<double> shared_face_dot;

  shared_face_indices.clear();
  shared_face_dot.clear();
  shared_face_indices.reserve(static_cast<size_t>(lhs.nFaces));
  shared_face_dot.reserve(static_cast<size_t>(lhs.nFaces));

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];
    const int face_dof = std::min(face.dof, dof);

    if (face.coupledFlag) {
      double coef;
      if (op_Type == BcopType::BCOP_TYPE_ADD) {
        coef = face.res;
      } else {
        coef = -face.res / (1.0 + (face.res * face.nS));
      }

      // If face is shared across procs: compute dot product directly
      // over boundary nodes and use MPI_Allreduce, avoiding a full-mesh
      // temporary vector allocation.
      if (face.sharedFlag) {
        double local_S = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          // Only sum owned nodes (Ac < mynNo) to avoid double-counting
          if (Ac < lhs.mynNo) {
            for (int i = 0; i < face_dof; i++) {
              local_S += face.valM(i,a) * X(i,Ac);
            }
          }
        }
        shared_face_indices.push_back(faIn);
        shared_face_dot.push_back(local_S);

      }
      // If face is not shared across procs
      else {
        // Computing S = coef * v^T * X
        double S = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < face_dof; i++) {
            S = S + face.valM(i,a)*X(i,Ac);
          }
        }
        S = coef * S;

        // Computing Y = Y + v * S
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < face_dof; i++) {
            Y(i,Ac) = Y(i,Ac) + face.valM(i,a)*S;
          }
        }
      }
    }
  }

  if (!shared_face_indices.empty()) {
    if (lhs.commu.nTasks > 1) {
      const int count = static_cast<int>(shared_face_dot.size());
      MPI_Allreduce(MPI_IN_PLACE, shared_face_dot.data(), count, cm_mod::mpreal, MPI_SUM, lhs.commu.comm);
    }

    for (size_t idx = 0; idx < shared_face_indices.size(); ++idx) {
      auto& face = lhs.face[shared_face_indices[idx]];
      const int face_dof = std::min(face.dof, dof);

      double coef;
      if (op_Type == BcopType::BCOP_TYPE_ADD) {
        coef = face.res;
      } else {
        coef = -face.res / (1.0 + (face.res * face.nS));
      }

      const double S = shared_face_dot[idx] * coef;

      // Computing Y = Y + valM * S
      for (int a = 0; a < face.nNo; a++) {
        const int Ac = face.glob(a);
        for (int i = 0; i < face_dof; i++) {
          Y(i,Ac) = Y(i,Ac) + face.valM(i,a) * S;
        }
      }
    }
  }

}

};

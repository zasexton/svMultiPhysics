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

#include <cmath>
#include <unordered_map>
#include <vector>

namespace add_bc_mul {

namespace {

struct ReducedEntryKey
{
  fsils_int node = -1;
  int full_component = -1;

  bool operator==(const ReducedEntryKey& other) const noexcept
  {
    return node == other.node && full_component == other.full_component;
  }
};

struct ReducedEntryKeyHash
{
  std::size_t operator()(const ReducedEntryKey& key) const noexcept
  {
    std::size_t seed = std::hash<fsils_int>{}(key.node);
    seed ^= std::hash<int>{}(key.full_component) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
    return seed;
  }
};

inline int entry_local_component(const FSILS_lhsType& lhs,
                                 const FSILS_reducedFieldUpdateType& update,
                                 const FSILS_reducedSparseEntry& entry,
                                 int dof)
{
  if (entry.node < 0) {
    return -1;
  }
  const int system_dof = (lhs.system_dof > 0) ? lhs.system_dof : dof;
  return fe_fsi_linear_solver::fsils_reduced_local_component(update,
                                                              entry.full_component,
                                                              dof,
                                                              system_dof);
}

inline double sparse_dot_owned(const FSILS_lhsType& lhs,
                               const FSILS_reducedFieldUpdateType& update,
                               const std::vector<FSILS_reducedSparseEntry>& entries,
                               const Array<double>& X,
                               int dof)
{
  double local_dot = 0.0;
  for (const auto& entry : entries) {
    const int comp = entry_local_component(lhs, update, entry, dof);
    if (comp < 0 || comp >= dof) {
      continue;
    }
    local_dot += entry.value * X(comp, entry.node);
  }
  return local_dot;
}

inline void sparse_axpy_full(const FSILS_lhsType& lhs,
                             const FSILS_reducedFieldUpdateType& update,
                             const std::vector<FSILS_reducedSparseEntry>& entries,
                             double scale,
                             int dof,
                             Array<double>& Y)
{
  if (std::abs(scale) <= 1e-30) {
    return;
  }
  for (const auto& entry : entries) {
    const int comp = entry_local_component(lhs, update, entry, dof);
    if (comp < 0 || comp >= dof || std::abs(entry.value) <= 1e-30) {
      continue;
    }
    Y(comp, entry.node) += entry.value * scale;
  }
}

} // namespace

void compute_reduced_update_preconditioner_coupling(FSILS_lhsType& lhs)
{
  if (lhs.reduced_updates.empty()) {
    return;
  }

  for (auto& update : lhs.reduced_updates) {
    if (!update.active) {
      update.nS = 0.0;
      continue;
    }

    std::unordered_map<ReducedEntryKey, double, ReducedEntryKeyHash> left_values;
    left_values.reserve(update.left_scaled_owned.size());
    for (const auto& entry : update.left_scaled_owned) {
      if (entry.node < 0 || entry.full_component < 0 || std::abs(entry.value) <= 1e-30) {
        continue;
      }
      left_values[{entry.node, entry.full_component}] += entry.value;
    }

    double local_nS = 0.0;
    for (const auto& entry : update.right_scaled_owned) {
      if (entry.node < 0 || entry.full_component < 0 || std::abs(entry.value) <= 1e-30) {
        continue;
      }
      const auto it = left_values.find({entry.node, entry.full_component});
      if (it != left_values.end()) {
        local_nS += entry.value * it->second;
      }
    }

    double global_nS = local_nS;
    if (lhs.commu.nTasks > 1) {
      fsils_allreduce_sum(&local_nS, &global_nS, 1, cm_mod::mpreal, lhs.commu);
    }
    update.nS = global_nS;
  }
}

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
      fsils_allreduce_sum_in_place(shared_face_dot.data(), count, cm_mod::mpreal, lhs.commu);
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

  if (!lhs.reduced_updates.empty()) {
    std::vector<double> reduced_dots(lhs.reduced_updates.size(), 0.0);
    for (std::size_t idx = 0; idx < lhs.reduced_updates.size(); ++idx) {
      const auto& update = lhs.reduced_updates[idx];
      if (!update.active) {
        continue;
      }
      const auto& right_entries =
          update.right_scaled_owned.empty() ? update.right_owned : update.right_scaled_owned;
      reduced_dots[idx] = sparse_dot_owned(lhs, update, right_entries, X, dof);
    }

    if (lhs.commu.nTasks > 1 && !reduced_dots.empty()) {
      fsils_allreduce_sum_in_place(reduced_dots.data(),
                                   static_cast<int>(reduced_dots.size()),
                                   cm_mod::mpreal,
                                   lhs.commu);
    }

    for (std::size_t idx = 0; idx < lhs.reduced_updates.size(); ++idx) {
      const auto& update = lhs.reduced_updates[idx];
      if (!update.active) {
        continue;
      }

      double coef = update.sigma;
      if (op_Type == BcopType::BCOP_TYPE_PRE) {
        coef = -update.sigma / (1.0 + update.sigma * update.nS);
      }

      const auto& left_entries =
          update.left_scaled.empty() ? update.left : update.left_scaled;
      sparse_axpy_full(lhs, update, left_entries, coef * reduced_dots[idx], dof, Y);
    }
  }

}

};

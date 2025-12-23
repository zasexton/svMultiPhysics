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

#ifndef SVMP_ENTITY_TRANSFER_MAP_H
#define SVMP_ENTITY_TRANSFER_MAP_H

#include "MeshTypes.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace svmp {

/**
 * @brief Sparse entity-to-entity transfer map (CSR-style)
 *
 * Represents a linear mapping from a source entity field to a destination entity field:
 *
 *   dst[i] = sum_j weights[j] * src[src_indices[j]]
 *
 * where j ranges over [dst_offsets[i], dst_offsets[i+1]).
 *
 * The same weights are applied to each component of a multi-component field.
 */
struct EntityTransferMap {
  EntityKind kind = EntityKind::Vertex;
  size_t src_count = 0;
  size_t dst_count = 0;

  std::vector<offset_t> dst_offsets;   // length dst_count + 1, starts with 0
  std::vector<index_t> src_indices;    // flattened src indices
  std::vector<real_t> weights;         // flattened weights (same length as src_indices)

  bool empty() const noexcept { return dst_count == 0; }

  void validate(bool require_weights = true) const {
    if (dst_offsets.size() != dst_count + 1) {
      throw std::invalid_argument("EntityTransferMap::validate: dst_offsets size mismatch");
    }
    if (!dst_offsets.empty() && dst_offsets.front() != 0) {
      throw std::invalid_argument("EntityTransferMap::validate: dst_offsets[0] must be 0");
    }
    for (size_t i = 0; i + 1 < dst_offsets.size(); ++i) {
      if (dst_offsets[i + 1] < dst_offsets[i]) {
        throw std::invalid_argument("EntityTransferMap::validate: dst_offsets must be non-decreasing");
      }
    }

    const size_t nnz = src_indices.size();
    if (!dst_offsets.empty() && static_cast<size_t>(dst_offsets.back()) != nnz) {
      throw std::invalid_argument("EntityTransferMap::validate: dst_offsets.back() must equal src_indices.size()");
    }
    if (require_weights && weights.size() != nnz) {
      throw std::invalid_argument("EntityTransferMap::validate: weights size mismatch");
    }

    for (size_t k = 0; k < nnz; ++k) {
      const index_t s = src_indices[k];
      if (s < 0 || static_cast<size_t>(s) >= src_count) {
        throw std::invalid_argument("EntityTransferMap::validate: src index out of range");
      }
    }
  }

  bool is_injection(real_t weight_tol = 0.0) const {
    validate(true);
    for (size_t i = 0; i < dst_count; ++i) {
      const size_t begin = static_cast<size_t>(dst_offsets[i]);
      const size_t end = static_cast<size_t>(dst_offsets[i + 1]);
      if (end - begin != 1) {
        return false;
      }
      const real_t w = weights[begin];
      if (std::abs(w - 1.0) > weight_tol) {
        return false;
      }
    }
    return true;
  }

  void normalize_weights(real_t min_abs_sum = 0.0) {
    validate(true);
    for (size_t i = 0; i < dst_count; ++i) {
      const size_t begin = static_cast<size_t>(dst_offsets[i]);
      const size_t end = static_cast<size_t>(dst_offsets[i + 1]);
      real_t sum = 0.0;
      for (size_t k = begin; k < end; ++k) {
        sum += weights[k];
      }
      if (std::abs(sum) <= min_abs_sum) {
        throw std::invalid_argument("EntityTransferMap::normalize_weights: near-zero weight sum");
      }
      for (size_t k = begin; k < end; ++k) {
        weights[k] /= sum;
      }
    }
  }

  static EntityTransferMap identity(EntityKind kind_in, size_t n) {
    EntityTransferMap map;
    map.kind = kind_in;
    map.src_count = n;
    map.dst_count = n;
    map.dst_offsets.resize(n + 1);
    map.src_indices.resize(n);
    map.weights.resize(n, 1.0);
    for (size_t i = 0; i < n; ++i) {
      map.dst_offsets[i] = static_cast<offset_t>(i);
      map.src_indices[i] = static_cast<index_t>(i);
    }
    map.dst_offsets[n] = static_cast<offset_t>(n);
    return map;
  }

  static EntityTransferMap injection(EntityKind kind_in,
                                     size_t src_count_in,
                                     const std::vector<index_t>& dst_to_src) {
    EntityTransferMap map;
    map.kind = kind_in;
    map.src_count = src_count_in;
    map.dst_count = dst_to_src.size();
    map.dst_offsets.resize(map.dst_count + 1);
    map.src_indices = dst_to_src;
    map.weights.assign(map.dst_count, 1.0);
    for (size_t i = 0; i < map.dst_count; ++i) {
      map.dst_offsets[i] = static_cast<offset_t>(i);
    }
    map.dst_offsets[map.dst_count] = static_cast<offset_t>(map.dst_count);
    map.validate(true);
    return map;
  }

  static EntityTransferMap from_lists(EntityKind kind_in,
                                      size_t src_count_in,
                                      const std::vector<std::vector<index_t>>& dst_to_src,
                                      const std::vector<std::vector<real_t>>* dst_to_weights = nullptr) {
    EntityTransferMap map;
    map.kind = kind_in;
    map.src_count = src_count_in;
    map.dst_count = dst_to_src.size();
    map.dst_offsets.resize(map.dst_count + 1, 0);

    size_t nnz = 0;
    for (size_t i = 0; i < map.dst_count; ++i) {
      map.dst_offsets[i] = static_cast<offset_t>(nnz);
      nnz += dst_to_src[i].size();
    }
    map.dst_offsets[map.dst_count] = static_cast<offset_t>(nnz);

    map.src_indices.reserve(nnz);
    map.weights.reserve(nnz);

    for (size_t i = 0; i < map.dst_count; ++i) {
      const auto& srcs = dst_to_src[i];
      const bool has_w = (dst_to_weights != nullptr);
      if (has_w && (*dst_to_weights)[i].size() != srcs.size()) {
        throw std::invalid_argument("EntityTransferMap::from_lists: weight list size mismatch");
      }
      for (size_t k = 0; k < srcs.size(); ++k) {
        map.src_indices.push_back(srcs[k]);
        map.weights.push_back(has_w ? (*dst_to_weights)[i][k] : 1.0);
      }
    }

    map.validate(true);
    return map;
  }
};

} // namespace svmp

#endif // SVMP_ENTITY_TRANSFER_MAP_H


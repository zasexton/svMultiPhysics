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

#ifndef SVMP_HIGH_ORDER_EMBEDDING_H
#define SVMP_HIGH_ORDER_EMBEDDING_H

/**
 * @file HighOrderEmbedding.h
 * @brief Mesh-owned contract for p>2 high-order geometry refinement
 *
 * The Mesh library owns topology/geometry refinement. For p>2 curved elements,
 * Mesh needs FE-specific "embedding" information to create child geometry nodes
 * consistently with the FE library's reference-node layout and basis.
 *
 * This header defines dependency-light data structures for that embedding.
 * The FE library supplies `HighOrderEmbedding` data via a provider interface,
 * and the Mesh library caches it for efficient reuse.
 *
 * Key design constraints:
 * - No FE headers are included here (dependency inversion).
 * - Embedding is expressed as sparse weights from child local nodes to parent
 *   local nodes (isoparametric refinement).
 * - All runtime mesh-specific mapping is performed by Mesh by translating
 *   parent-local node indices to global vertex indices (GID-preserving).
 */

#include "../Core/MeshTypes.h"
#include "RefinementRules.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {

/**
 * @brief Lightweight sparse row matrix for per-node embedding weights.
 *
 * Represents a matrix W (n_rows x n_cols) in CSR format:
 * - row_offsets: size n_rows + 1, row_offsets[0] == 0, row_offsets[n_rows] == nnz
 * - col_indices/values: size nnz
 */
struct CSRWeights {
  std::vector<std::int32_t> row_offsets;
  std::vector<std::int32_t> col_indices;
  std::vector<double> values;

  size_t rows() const noexcept {
    return row_offsets.empty() ? 0u : static_cast<size_t>(row_offsets.size() - 1);
  }
  size_t nnz() const noexcept { return values.size(); }

  /**
   * @brief Validate structural invariants and index bounds.
   *
   * @param n_cols Number of parent-local nodes (matrix columns).
   * @throws std::runtime_error if invalid.
   */
  void validate(std::int32_t n_cols) const;
};

/**
 * @brief Cache key for a high-order embedding rule.
 */
struct HighOrderEmbeddingKey {
  CellFamily parent_family = CellFamily::Point;
  int parent_order = 1;
  int parent_num_nodes = 0;
  RefinementSpec spec{};

  bool operator==(const HighOrderEmbeddingKey& o) const noexcept {
    return parent_family == o.parent_family &&
           parent_order == o.parent_order &&
           parent_num_nodes == o.parent_num_nodes &&
           spec.pattern == o.spec.pattern &&
           spec.selector == o.spec.selector;
  }
};

struct HighOrderEmbeddingKeyHash {
  size_t operator()(const HighOrderEmbeddingKey& k) const noexcept;
};

/**
 * @brief Embedding for one child element.
 *
 * Each child node i is defined by a sparse set of weights w_{ij} on parent nodes j:
 *   x_child_i = sum_j w_{ij} * x_parent_j.
 */
struct HighOrderChildEmbedding {
  CellFamily child_family = CellFamily::Point;
  int child_order = 1;
  int child_num_nodes = 0;
  CSRWeights child_node_parent_weights;
};

/**
 * @brief Complete embedding rule for a parent cell family/order/spec.
 */
struct HighOrderEmbedding {
  HighOrderEmbeddingKey key{};
  std::vector<HighOrderChildEmbedding> children;

  /**
   * @brief Validate embedding data.
   *
   * @throws std::runtime_error if invalid.
   */
  void validate() const;
};

/**
 * @brief Provider interface for FE-supplied high-order embedding rules.
 *
 * Implemented by the FE library; owned/injected by the application.
 */
class HighOrderEmbeddingProvider {
public:
  virtual ~HighOrderEmbeddingProvider() = default;

  /**
   * @brief Provide embedding for a given key.
   *
   * Implementations should:
   * - Fill `out.key` and `out.children` (children may be mixed families).
   * - Use parent-local node indices [0, parent_num_nodes) for weight columns.
   *
   * @return true if embedding is available for this key.
   */
  virtual bool get_high_order_embedding(
      const HighOrderEmbeddingKey& key,
      HighOrderEmbedding& out) const = 0;
};

/**
 * @brief Cache for FE-provided high-order embedding rules.
 */
class HighOrderEmbeddingCache {
public:
  const HighOrderEmbedding* find(const HighOrderEmbeddingKey& key) const;
  bool has(const HighOrderEmbeddingKey& key) const { return find(key) != nullptr; }

  void insert(HighOrderEmbedding embedding);
  void clear();

  /**
   * @brief Get an embedding rule, requesting it from `provider` on cache miss.
   *
   * @throws std::runtime_error if provider is null or cannot supply the embedding.
   */
  const HighOrderEmbedding& get_or_request(
      const HighOrderEmbeddingKey& key,
      const HighOrderEmbeddingProvider* provider);

private:
  std::unordered_map<HighOrderEmbeddingKey, HighOrderEmbedding, HighOrderEmbeddingKeyHash> cache_;
};

} // namespace svmp

#endif // SVMP_HIGH_ORDER_EMBEDDING_H


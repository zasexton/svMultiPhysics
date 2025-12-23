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

#include "HighOrderEmbedding.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace svmp {

namespace {

std::string key_string(const HighOrderEmbeddingKey& k) {
  std::ostringstream os;
  os << "family=" << static_cast<int>(k.parent_family)
     << " order=" << k.parent_order
     << " n_nodes=" << k.parent_num_nodes
     << " pattern=" << static_cast<int>(k.spec.pattern)
     << " selector=" << k.spec.selector;
  return os.str();
}

} // namespace

void CSRWeights::validate(std::int32_t n_cols) const {
  if (row_offsets.empty()) {
    throw std::runtime_error("CSRWeights: row_offsets is empty");
  }
  if (row_offsets.front() != 0) {
    throw std::runtime_error("CSRWeights: row_offsets[0] must be 0");
  }
  if (col_indices.size() != values.size()) {
    throw std::runtime_error("CSRWeights: col_indices/values size mismatch");
  }
  for (size_t i = 1; i < row_offsets.size(); ++i) {
    if (row_offsets[i] < row_offsets[i - 1]) {
      throw std::runtime_error("CSRWeights: row_offsets must be nondecreasing");
    }
  }
  const std::int32_t nnz = static_cast<std::int32_t>(values.size());
  if (row_offsets.back() != nnz) {
    throw std::runtime_error("CSRWeights: row_offsets.back() must equal nnz");
  }
  if (n_cols <= 0) {
    throw std::runtime_error("CSRWeights: invalid n_cols");
  }
  for (size_t k = 0; k < col_indices.size(); ++k) {
    const auto j = col_indices[k];
    if (j < 0 || j >= n_cols) {
      throw std::runtime_error("CSRWeights: column index out of range");
    }
  }
}

size_t HighOrderEmbeddingKeyHash::operator()(const HighOrderEmbeddingKey& k) const noexcept {
  size_t h = 0;
  auto mix = [&](size_t x) {
    h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  mix(std::hash<int>{}(static_cast<int>(k.parent_family)));
  mix(std::hash<int>{}(k.parent_order));
  mix(std::hash<int>{}(k.parent_num_nodes));
  mix(std::hash<int>{}(static_cast<int>(k.spec.pattern)));
  mix(std::hash<int>{}(k.spec.selector));
  return h;
}

void HighOrderEmbedding::validate() const {
  if (key.parent_num_nodes <= 0) {
    throw std::runtime_error("HighOrderEmbedding: parent_num_nodes must be > 0");
  }
  if (children.empty()) {
    throw std::runtime_error("HighOrderEmbedding: no children provided");
  }

  const std::int32_t n_parent = static_cast<std::int32_t>(key.parent_num_nodes);
  for (const auto& child : children) {
    if (child.child_num_nodes <= 0) {
      throw std::runtime_error("HighOrderEmbedding: child_num_nodes must be > 0");
    }
    if (child.child_node_parent_weights.rows() != static_cast<size_t>(child.child_num_nodes)) {
      throw std::runtime_error("HighOrderEmbedding: CSR rows != child_num_nodes");
    }
    child.child_node_parent_weights.validate(n_parent);
  }
}

const HighOrderEmbedding* HighOrderEmbeddingCache::find(const HighOrderEmbeddingKey& key) const {
  auto it = cache_.find(key);
  if (it == cache_.end()) return nullptr;
  return &it->second;
}

void HighOrderEmbeddingCache::insert(HighOrderEmbedding embedding) {
  embedding.validate();
  cache_[embedding.key] = std::move(embedding);
}

void HighOrderEmbeddingCache::clear() {
  cache_.clear();
}

const HighOrderEmbedding& HighOrderEmbeddingCache::get_or_request(
    const HighOrderEmbeddingKey& key,
    const HighOrderEmbeddingProvider* provider) {
  if (const auto* found = find(key)) return *found;
  if (!provider) {
    throw std::runtime_error("HighOrderEmbeddingCache: provider is null for key: " + key_string(key));
  }

  HighOrderEmbedding embedding;
  if (!provider->get_high_order_embedding(key, embedding)) {
    throw std::runtime_error("HighOrderEmbeddingCache: provider has no embedding for key: " + key_string(key));
  }

  // If the provider didn't populate key (allowed), force it here.
  embedding.key = key;
  embedding.validate();

  auto [it, inserted] = cache_.try_emplace(key, std::move(embedding));
  if (!inserted) it->second = std::move(embedding);
  return it->second;
}

} // namespace svmp

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
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
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
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_MESH_SEARCH_COORDINATE_KEY_H
#define SVMP_MESH_SEARCH_COORDINATE_KEY_H

#include "../Core/MeshBase.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace svmp {
namespace search {

struct CoordinateKey {
  std::array<std::uint64_t, 3> components{{0u, 0u, 0u}};
  std::uint8_t dim{0u};

  bool operator==(const CoordinateKey& other) const noexcept
  {
    return dim == other.dim && components == other.components;
  }

  bool operator!=(const CoordinateKey& other) const noexcept
  {
    return !(*this == other);
  }
};

inline std::uint64_t canonical_real_bits(real_t value) noexcept
{
  if (value == real_t(0)) {
    value = real_t(0);
  }

  if (std::isnan(static_cast<double>(value))) {
    return UINT64_C(0x7ff8000000000000);
  }

  if constexpr (sizeof(real_t) == sizeof(std::uint64_t)) {
    std::uint64_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  } else if constexpr (sizeof(real_t) == sizeof(std::uint32_t)) {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint64_t>(bits);
  } else {
    const double widened = static_cast<double>(value);
    std::uint64_t bits = 0u;
    std::memcpy(&bits, &widened, sizeof(bits));
    return bits;
  }
}

inline CoordinateKey make_coordinate_key(const std::vector<real_t>& coordinates,
                                         int dim,
                                         index_t vertex)
{
  if (dim <= 0 || dim > 3) {
    throw std::invalid_argument("make_coordinate_key: dimension must be in [1,3]");
  }
  if (vertex < 0) {
    throw std::out_of_range("make_coordinate_key: negative vertex index");
  }
  const auto v = static_cast<std::size_t>(vertex);
  const auto d = static_cast<std::size_t>(dim);
  if ((v + 1u) > (std::numeric_limits<std::size_t>::max() / d) ||
      (v + 1u) * d > coordinates.size()) {
    throw std::out_of_range("make_coordinate_key: vertex index outside coordinate array");
  }

  CoordinateKey key;
  key.dim = static_cast<std::uint8_t>(dim);
  const auto base = v * d;
  for (int i = 0; i < dim; ++i) {
    key.components[static_cast<std::size_t>(i)] =
        canonical_real_bits(coordinates[base + static_cast<std::size_t>(i)]);
  }
  return key;
}

struct CoordinateKeyHash {
  std::size_t operator()(const CoordinateKey& key) const noexcept
  {
    std::uint64_t h = UINT64_C(0x9e3779b97f4a7c15) ^ key.dim;
    for (const auto component : key.components) {
      std::uint64_t x = component + UINT64_C(0x9e3779b97f4a7c15);
      x = (x ^ (x >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
      x = (x ^ (x >> 27u)) * UINT64_C(0x94d049bb133111eb);
      h ^= (x ^ (x >> 31u)) + UINT64_C(0x9e3779b97f4a7c15) + (h << 6u) + (h >> 2u);
    }
    return static_cast<std::size_t>(h);
  }
};

class VertexCoordinateLocator {
public:
  explicit VertexCoordinateLocator(const MeshBase& mesh)
      : coordinates_(&mesh.X_ref()),
        dim_(mesh.dim())
  {
    if (dim_ <= 0 || dim_ > 3) {
      throw std::invalid_argument("VertexCoordinateLocator: mesh dimension must be in [1,3]");
    }
    vertex_by_coordinate_.reserve(mesh.n_vertices());
    for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
      vertex_by_coordinate_.emplace(make_coordinate_key(*coordinates_, dim_, v), v);
    }
  }

  [[nodiscard]] index_t find(const std::vector<real_t>& coordinates,
                             int dim,
                             index_t vertex) const
  {
    const auto key = make_coordinate_key(coordinates, dim, vertex);
    return find(key);
  }

  [[nodiscard]] index_t find(const CoordinateKey& key) const noexcept
  {
    const auto it = vertex_by_coordinate_.find(key);
    return it == vertex_by_coordinate_.end() ? INVALID_INDEX : it->second;
  }

  [[nodiscard]] CoordinateKey key_for_vertex(index_t vertex) const
  {
    return make_coordinate_key(*coordinates_, dim_, vertex);
  }

  [[nodiscard]] std::size_t size() const noexcept
  {
    return vertex_by_coordinate_.size();
  }

private:
  const std::vector<real_t>* coordinates_{nullptr};
  int dim_{0};
  std::unordered_map<CoordinateKey, index_t, CoordinateKeyHash> vertex_by_coordinate_;
};

} // namespace search
} // namespace svmp

#endif // SVMP_MESH_SEARCH_COORDINATE_KEY_H

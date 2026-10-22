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

#ifndef SVMP_STARTUP_PARTITION_WEIGHTS_H
#define SVMP_STARTUP_PARTITION_WEIGHTS_H

// Startup partition weights from the sign of a scalar vertex field, for
// example the initial level set of a free-surface deck.  DistributedMesh
// attaches them as a temporary integer cell field before the startup
// ParMETIS repartition, which then balances two constraints per rank: the
// cell count and the summed weight.  Requested by the MeshIOOptions keys
//   partition_weight_vertex_field  name of a scalar Float32/Float64 vertex field
//   partition_weight_isovalue      (default 0)
//   partition_weight_negative      weight of a cell whose vertex values all
//                                  lie below the isovalue (default 1)
//   partition_weight_positive      ... all above it (default 1)
//   partition_weight_cut           ... straddling or touching it (default 1)
//   partition_weight_fallback      "unweighted": when the weights cannot be
//                                  computed (for example the field is
//                                  missing), use the unweighted partition
//                                  instead of failing (default: fail)
// Without partition_weight_vertex_field (or with an empty name) nothing is
// weighted and the partition is the unweighted one.

#include "MeshBase.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {

struct StartupPartitionSignWeights {
  std::string vertex_field{};
  double isovalue{0.0};
  double negative{1.0};
  double positive{1.0};
  double cut{1.0};
  bool fallback_to_unweighted{false};
};

// Name of the temporary cell field that carries the weights through the
// startup repartition; it is removed again afterwards.
inline constexpr const char* kStartupPartitionWeightField = "svmp_startup_partition_weight";

// A weight w becomes max(1, round(kStartupPartitionWeightScale * w)) integer
// units.  ParMETIS needs integer weights; at 100 units per unit weight a
// zero weight (floored at one unit) counts 1% of a unit-weight cell.
inline constexpr double kStartupPartitionWeightScale = 100.0;

inline std::int32_t startup_partition_weight_units(double weight) {
  const long long units = std::llround(kStartupPartitionWeightScale * weight);
  return static_cast<std::int32_t>(units < 1 ? 1 : units);
}

inline double startup_partition_weight_option(const std::unordered_map<std::string, std::string>& kv,
                                              const char* key,
                                              double default_value) {
  const auto it = kv.find(key);
  if (it == kv.end() || it->second.empty()) {
    return default_value;
  }
  std::size_t used = 0;
  double value = 0.0;
  try {
    value = std::stod(it->second, &used);
  } catch (...) {
    used = 0;
  }
  // Larger weights would only risk overflowing the 32-bit ParMETIS weight
  // sums (DistributedMesh also checks the global sum).
  if (used != it->second.size() || !std::isfinite(value) || value < 0.0 || value > 1.0e4) {
    throw std::invalid_argument(std::string("DistributedMesh: invalid ") + key + "='" + it->second +
                                "' (expected a finite number in [0, 1e4])");
  }
  return value;
}

// The weights requested by `kv`; std::nullopt when no vertex field is named.
inline std::optional<StartupPartitionSignWeights> startup_partition_sign_weights(
    const std::unordered_map<std::string, std::string>& kv) {
  const auto field_it = kv.find("partition_weight_vertex_field");
  if (field_it == kv.end() || field_it->second.empty()) {
    return std::nullopt;
  }
  StartupPartitionSignWeights weights;
  weights.vertex_field = field_it->second;
  const auto isovalue_it = kv.find("partition_weight_isovalue");
  if (isovalue_it != kv.end() && !isovalue_it->second.empty()) {
    std::size_t used = 0;
    double value = 0.0;
    try {
      value = std::stod(isovalue_it->second, &used);
    } catch (...) {
      used = 0;
    }
    if (used != isovalue_it->second.size() || !std::isfinite(value)) {
      throw std::invalid_argument("DistributedMesh: invalid partition_weight_isovalue='" +
                                  isovalue_it->second + "' (expected a finite number)");
    }
    weights.isovalue = value;
  }
  weights.negative = startup_partition_weight_option(kv, "partition_weight_negative", 1.0);
  weights.positive = startup_partition_weight_option(kv, "partition_weight_positive", 1.0);
  weights.cut = startup_partition_weight_option(kv, "partition_weight_cut", 1.0);
  if (const auto it = kv.find("partition_weight_fallback"); it != kv.end() && !it->second.empty()) {
    if (it->second != "unweighted") {
      throw std::invalid_argument("DistributedMesh: invalid partition_weight_fallback='" + it->second +
                                  "' (expected 'unweighted')");
    }
    weights.fallback_to_unweighted = true;
  }
  return weights;
}

// Integer weight units of every local cell of `mesh`: a cell whose vertex
// values all lie below the isovalue weighs `negative`, all above it
// `positive`, otherwise (straddling, or a vertex on the isovalue) `cut`.
inline std::vector<std::int32_t> startup_partition_cell_weights(const MeshBase& mesh,
                                                                const StartupPartitionSignWeights& weights) {
  std::vector<std::int32_t> out(mesh.n_cells(), 1);
  if (mesh.n_cells() == 0) {
    return out;
  }
  const std::string& field = weights.vertex_field;
  if (!mesh.has_field(EntityKind::Vertex, field)) {
    throw std::runtime_error("DistributedMesh: partition_weight_vertex_field '" + field +
                             "' is not a vertex field of the mesh");
  }
  const auto handle = mesh.field_handle(EntityKind::Vertex, field);
  if (mesh.field_components(handle) != 1) {
    throw std::runtime_error("DistributedMesh: partition_weight_vertex_field '" + field +
                             "' must be a scalar field");
  }
  const auto type = mesh.field_type(handle);
  const void* raw = mesh.field_data(handle);
  if ((type != FieldScalarType::Float64 && type != FieldScalarType::Float32) || raw == nullptr) {
    throw std::runtime_error("DistributedMesh: partition_weight_vertex_field '" + field +
                             "' must hold Float32 or Float64 data");
  }
  const auto value_at = [&](index_t v) -> double {
    if (type == FieldScalarType::Float64) {
      return static_cast<const double*>(raw)[static_cast<std::size_t>(v)];
    }
    return static_cast<double>(static_cast<const float*>(raw)[static_cast<std::size_t>(v)]);
  };

  const std::int32_t w_negative = startup_partition_weight_units(weights.negative);
  const std::int32_t w_positive = startup_partition_weight_units(weights.positive);
  const std::int32_t w_cut = startup_partition_weight_units(weights.cut);
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    auto [vptr, nv] = mesh.cell_vertices_span(c);
    bool below = nv > 0;
    bool above = nv > 0;
    for (std::size_t k = 0; k < nv; ++k) {
      const double value = value_at(vptr[k]);
      below = below && value < weights.isovalue;
      above = above && value > weights.isovalue;
    }
    out[static_cast<std::size_t>(c)] = below ? w_negative : (above ? w_positive : w_cut);
  }
  return out;
}

} // namespace svmp

#endif // SVMP_STARTUP_PARTITION_WEIGHTS_H

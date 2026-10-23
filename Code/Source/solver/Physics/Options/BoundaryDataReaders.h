/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_OPTIONS_BOUNDARY_DATA_READERS_H
#define SVMP_PHYSICS_OPTIONS_BOUNDARY_DATA_READERS_H

/**
 * @file BoundaryDataReaders.h
 * @brief Readers of boundary data files and boundary-marker geometry
 *
 * Temporal_and_spatial_values files (gathered over the ranks that share a
 * boundary marker), node-pressure CSV files, the area, centre and normal of
 * a marker, and flat and parabolic inlet profiles.  Moved unchanged from
 * NavierStokesRegister.cpp (decision D52).
 */

#include "Physics/Options/InputReaders.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Core/Types.h"
#include "Mesh/Core/MeshTypes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
class MeshBase;
} // namespace svmp

namespace svmp::FE::systems {
class FESystem;
} // namespace svmp::FE::systems

namespace svmp::Physics::options {

#if FE_HAS_MPI
using MarkerCommunicator = MPI_Comm;
#else
using MarkerCommunicator = int;
#endif

struct TemporalSpatialValues {
  struct Key {
    std::int64_t x{0};
    std::int64_t y{0};
    std::int64_t z{0};

    friend bool operator==(const Key& a, const Key& b) noexcept
    {
      return a.x == b.x && a.y == b.y && a.z == b.z;
    }
  };

  struct KeyHash {
    size_t operator()(const Key& k) const noexcept
    {
      size_t h = 1469598103934665603ull;
      auto mix = [&](std::int64_t v) {
        const size_t x = std::hash<std::int64_t>{}(v);
        h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      };
      mix(k.x);
      mix(k.y);
      mix(k.z);
      return h;
    }
  };

  int dim{0};
  int dof{0};
  int num_time_points{0};
  int boundary_marker{0};
  std::string file_path{};

  std::vector<double> t{};
  double period{0.0};

  std::vector<svmp::index_t> node_ids{};
  std::vector<std::array<svmp::FE::Real, 3>> coords{};

  // Data layout: d[((node * num_time_points) + it) * dof + comp]
  std::vector<svmp::FE::Real> d{};

  std::unordered_map<Key, std::size_t, KeyHash> node_index_by_key{};
  bool has_x_only_interpolant{false};
  std::vector<svmp::FE::Real> x_interpolation_coords{};
  std::vector<std::size_t> x_interpolation_nodes{};

  // Uniform bucket grid over the stored coordinates.  The nearest-node
  // interpolant is evaluated at every quadrature point, and a scan over all
  // stored nodes per evaluation made a 1,573-node 2D table cost about 18 s
  // per time step; the grid returns the same neighbours (ties broken by node
  // index, as the scan did) after visiting a few buckets.
  static constexpr std::size_t kNearestNodes = 8;
  std::array<double, 3> grid_lo{0.0, 0.0, 0.0};
  std::array<double, 3> grid_h{1.0, 1.0, 1.0};
  std::array<long long, 3> grid_n{1, 1, 1};
  std::vector<std::size_t> grid_offsets{};
  std::vector<std::size_t> grid_items{};

  [[nodiscard]] static Key quantize(const std::array<svmp::FE::Real, 3>& p, int dim_in) noexcept
  {
    constexpr double scale = 1e12;
    auto q = [&](svmp::FE::Real v) { return static_cast<std::int64_t>(std::llround(static_cast<double>(v) * scale)); };
    Key k{};
    k.x = q(p[0]);
    k.y = (dim_in >= 2) ? q(p[1]) : 0;
    k.z = (dim_in >= 3) ? q(p[2]) : 0;
    return k;
  }

  [[nodiscard]] svmp::FE::Real sample(std::size_t node_idx, int time_idx, int comp) const
  {
    const auto idx = ((node_idx * static_cast<std::size_t>(num_time_points) + static_cast<std::size_t>(time_idx)) *
                          static_cast<std::size_t>(dof) +
                      static_cast<std::size_t>(comp));
    if (idx >= d.size()) {
      throw std::runtime_error("[svMultiPhysics::Physics] Internal error: temporal/spatial BC index out of range.");
    }
    return d[idx];
  }

  [[nodiscard]] static bool closeValues(svmp::FE::Real a, svmp::FE::Real b) noexcept
  {
    const double da = static_cast<double>(a);
    const double db = static_cast<double>(b);
    const double scale = std::max({1.0, std::abs(da), std::abs(db)});
    return std::abs(da - db) <= 1.0e-10 * scale;
  }

  void buildNearestGrid()
  {
    grid_offsets.clear();
    grid_items.clear();
    grid_n = {1, 1, 1};
    if (coords.empty()) {
      return;
    }
    std::array<double, 3> hi{};
    for (int k = 0; k < 3; ++k) {
      grid_lo[static_cast<std::size_t>(k)] = std::numeric_limits<double>::infinity();
      hi[static_cast<std::size_t>(k)] = -std::numeric_limits<double>::infinity();
    }
    for (const auto& c : coords) {
      for (std::size_t k = 0; k < 3; ++k) {
        grid_lo[k] = std::min(grid_lo[k], static_cast<double>(c[k]));
        hi[k] = std::max(hi[k], static_cast<double>(c[k]));
      }
    }
    // About two nodes per bucket on the spanned axes; an axis spans the
    // nodes if its extent exceeds 1e-9 of the longest one, and the bucket
    // count stays below four per node.
    double longest = 0.0;
    for (std::size_t k = 0; k < 3; ++k) {
      longest = std::max(longest, hi[k] - grid_lo[k]);
    }
    std::array<bool, 3> spans{false, false, false};
    double measure = 1.0;
    int spanned = 0;
    for (std::size_t k = 0; k < 3; ++k) {
      const double extent = hi[k] - grid_lo[k];
      if (extent > 1.0e-9 * longest && extent > 0.0) {
        spans[k] = true;
        measure *= extent;
        ++spanned;
      }
    }
    double h = longest;
    if (spanned > 0 && coords.size() > 2) {
      h = std::pow(2.0 * measure / static_cast<double>(coords.size()), 1.0 / spanned);
    }
    if (!(h > 0.0) || !std::isfinite(h)) {
      h = 1.0;
    }
    std::size_t buckets = 1;
    for (int attempt = 0; attempt < 64; ++attempt) {
      buckets = 1;
      for (std::size_t k = 0; k < 3; ++k) {
        const double extent = hi[k] - grid_lo[k];
        grid_n[k] = spans[k] ? std::max<long long>(1, static_cast<long long>(std::ceil(extent / h))) : 1;
        grid_h[k] = spans[k] ? extent / static_cast<double>(grid_n[k]) : std::max(extent, 1.0);
        buckets *= static_cast<std::size_t>(grid_n[k]);
      }
      if (buckets <= 4 * coords.size() + 1) {
        break;
      }
      h *= 2.0;
    }
    std::vector<std::size_t> bucket_of(coords.size());
    grid_offsets.assign(buckets + 1, 0);
    for (std::size_t i = 0; i < coords.size(); ++i) {
      std::array<long long, 3> b{};
      for (std::size_t k = 0; k < 3; ++k) {
        b[k] = bucketCoordinate(static_cast<double>(coords[i][k]), k);
      }
      bucket_of[i] = bucketIndex(b);
      ++grid_offsets[bucket_of[i] + 1];
    }
    for (std::size_t b = 0; b < buckets; ++b) {
      grid_offsets[b + 1] += grid_offsets[b];
    }
    grid_items.resize(coords.size());
    std::vector<std::size_t> fill(grid_offsets.begin(), grid_offsets.end() - 1);
    for (std::size_t i = 0; i < coords.size(); ++i) {
      grid_items[fill[bucket_of[i]]++] = i;
    }
  }

  [[nodiscard]] long long bucketCoordinate(double x, std::size_t k) const noexcept
  {
    const double s = (x - grid_lo[k]) / grid_h[k];
    if (!(s > 0.0)) {
      return 0;
    }
    return std::min(grid_n[k] - 1, static_cast<long long>(std::floor(s)));
  }

  [[nodiscard]] std::size_t bucketIndex(const std::array<long long, 3>& b) const noexcept
  {
    return static_cast<std::size_t>((b[2] * grid_n[1] + b[1]) * grid_n[0] + b[0]);
  }

  // The kNearestNodes stored nodes closest to p, ordered by (squared
  // distance, node index): the order of the former full partial_sort.
  [[nodiscard]] std::vector<std::pair<double, std::size_t>> nearestNodes(
      const std::array<svmp::FE::Real, 3>& p) const
  {
    const std::size_t k_wanted = std::min<std::size_t>(kNearestNodes, coords.size());
    std::vector<std::pair<double, std::size_t>> best;
    best.reserve(k_wanted + 1);
    if (k_wanted == 0) {
      return best;
    }
    const auto consider = [&](std::size_t i) {
      const auto& c = coords[i];
      const double dx = static_cast<double>(p[0] - c[0]);
      const double dy = static_cast<double>(p[1] - c[1]);
      const double dz = static_cast<double>(p[2] - c[2]);
      const std::pair<double, std::size_t> candidate{dx * dx + dy * dy + dz * dz, i};
      if (best.size() == k_wanted && !(candidate < best.back())) {
        return;
      }
      best.insert(std::upper_bound(best.begin(), best.end(), candidate), candidate);
      if (best.size() > k_wanted) {
        best.pop_back();
      }
    };
    std::array<long long, 3> center{};
    for (std::size_t k = 0; k < 3; ++k) {
      center[k] = bucketCoordinate(static_cast<double>(p[k]), k);
    }
    const long long max_ring = std::max({grid_n[0], grid_n[1], grid_n[2]});
    for (long long ring = 0; ring <= max_ring; ++ring) {
      std::array<long long, 3> lo{};
      std::array<long long, 3> hi{};
      for (std::size_t k = 0; k < 3; ++k) {
        lo[k] = std::max<long long>(0, center[k] - ring);
        hi[k] = std::min<long long>(grid_n[k] - 1, center[k] + ring);
      }
      for (long long bz = lo[2]; bz <= hi[2]; ++bz) {
        for (long long by = lo[1]; by <= hi[1]; ++by) {
          for (long long bx = lo[0]; bx <= hi[0]; ++bx) {
            const bool on_shell = std::llabs(bx - center[0]) == ring ||
                                  std::llabs(by - center[1]) == ring ||
                                  std::llabs(bz - center[2]) == ring;
            if (!on_shell) {
              continue;       // visited in an earlier ring
            }
            const auto b = bucketIndex({bx, by, bz});
            for (std::size_t j = grid_offsets[b]; j < grid_offsets[b + 1]; ++j) {
              consider(grid_items[j]);
            }
          }
        }
      }
      // Every unvisited node lies outside the visited bucket box; stop once
      // the k-th candidate is closer than that box's nearest open face.
      double gap = std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < 3; ++k) {
        if (lo[k] > 0) {
          gap = std::min(gap, static_cast<double>(p[k]) - (grid_lo[k] + static_cast<double>(lo[k]) * grid_h[k]));
        }
        if (hi[k] < grid_n[k] - 1) {
          gap = std::min(gap, (grid_lo[k] + static_cast<double>(hi[k] + 1) * grid_h[k]) - static_cast<double>(p[k]));
        }
      }
      if (gap == std::numeric_limits<double>::infinity()) {
        break;                // every bucket visited
      }
      gap -= 1.0e-12 * std::max({grid_h[0], grid_h[1], grid_h[2]});   // bucket rounding
      if (best.size() == k_wanted && gap > 0.0 && best.back().first < gap * gap) {
        break;
      }
    }
    return best;
  }

  void buildInterpolationMetadata()
  {
    buildNearestGrid();
    has_x_only_interpolant = false;
    x_interpolation_coords.clear();
    x_interpolation_nodes.clear();
    if (coords.size() < 2 || num_time_points <= 0 || dof <= 0) {
      return;
    }

    std::vector<std::size_t> order(coords.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    std::sort(order.begin(), order.end(), [this](std::size_t a, std::size_t b) {
      return coords[a][0] < coords[b][0];
    });

    constexpr double coord_tol = 1.0e-12;
    std::size_t group_begin = 0;
    bool values_depend_only_on_x = true;
    while (group_begin < order.size()) {
      const auto representative = order[group_begin];
      const double x0 = static_cast<double>(coords[representative][0]);
      std::size_t group_end = group_begin + 1;
      while (group_end < order.size() &&
             std::abs(static_cast<double>(coords[order[group_end]][0]) - x0) <= coord_tol) {
        ++group_end;
      }

      for (std::size_t j = group_begin + 1; j < group_end && values_depend_only_on_x; ++j) {
        const auto node = order[j];
        for (int it = 0; it < num_time_points && values_depend_only_on_x; ++it) {
          for (int comp = 0; comp < dof; ++comp) {
            if (!closeValues(sample(representative, it, comp), sample(node, it, comp))) {
              values_depend_only_on_x = false;
              break;
            }
          }
        }
      }
      if (!values_depend_only_on_x) {
        break;
      }

      x_interpolation_coords.push_back(coords[representative][0]);
      x_interpolation_nodes.push_back(representative);
      group_begin = group_end;
    }

    has_x_only_interpolant =
        values_depend_only_on_x && x_interpolation_coords.size() >= 2 &&
        std::abs(static_cast<double>(x_interpolation_coords.back() - x_interpolation_coords.front())) > 1.0e-14;
    if (!has_x_only_interpolant) {
      x_interpolation_coords.clear();
      x_interpolation_nodes.clear();
    }
  }

  [[nodiscard]] std::size_t findNodeIndex(const std::array<svmp::FE::Real, 3>& p) const
  {
    const auto key = quantize(p, dim);
    if (const auto it = node_index_by_key.find(key); it != node_index_by_key.end()) {
      return it->second;
    }

    // Fallback: nearest-node match (robust to minor floating differences).
    std::size_t best = coords.size();
    double best_d2 = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < coords.size(); ++i) {
      const auto& c = coords[i];
      const double dx = static_cast<double>(p[0] - c[0]);
      const double dy = static_cast<double>(p[1] - c[1]);
      const double dz = static_cast<double>(p[2] - c[2]);
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = i;
      }
    }

    constexpr double tol = 1e-8;
    if (best < coords.size() && best_d2 <= tol * tol) {
      return best;
    }

    if (temporalSpatialBcTraceEnabled()) {
      int rank = 0;
#if FE_HAS_MPI
      int initialized = 0;
      MPI_Initialized(&initialized);
      if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      }
#endif
      std::ostringstream oss;
      oss << "TemporalSpatialValues: rank=" << rank
          << " marker=" << boundary_marker
          << " file='" << file_path << "'"
          << " failed coordinate match"
          << " query=(" << p[0] << "," << p[1] << "," << p[2] << ")"
          << " stored_nodes=" << coords.size();
      if (best < coords.size()) {
        const auto& c = coords[best];
        oss << " nearest=(" << c[0] << "," << c[1] << "," << c[2] << ")"
            << " nearest_d=" << std::sqrt(best_d2);
      }
      navierStokesTraceLog(oss.str());
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Failed to match a temporal/spatial BC value to a boundary node by coordinate.");
  }

  [[nodiscard]] double wrapTime(double time) const noexcept
  {
    if (!(period > 0.0) || !std::isfinite(period) || num_time_points < 2) {
      return time;
    }
    double tmod = std::fmod(time, period);
    if (tmod < 0.0) {
      tmod += period;
    }
    return tmod;
  }

  [[nodiscard]] svmp::FE::Real interpolate(std::size_t node_idx, svmp::FE::Real time, int comp) const
  {
    if (num_time_points <= 0) {
      return svmp::FE::Real{0.0};
    }
    if (num_time_points == 1) {
      return sample(node_idx, 0, comp);
    }

    const double tt = wrapTime(static_cast<double>(time));

    // First interval whose right end is at or after tt (0 if none), found
    // by bisection: the table is evaluated at every quadrature point.
    int i0 = 0;
    const auto right = std::lower_bound(t.begin() + 1, t.begin() + num_time_points, tt);
    if (right != t.begin() + num_time_points) {
      i0 = static_cast<int>(std::distance(t.begin() + 1, right));
    }

    const double t0 = t[static_cast<std::size_t>(i0)];
    const double t1 = t[static_cast<std::size_t>(i0 + 1)];
    const double dt = t1 - t0;
    const double alpha = (dt > 0.0) ? ((tt - t0) / dt) : 0.0;

    const auto v0 = static_cast<double>(sample(node_idx, i0, comp));
    const auto v1 = static_cast<double>(sample(node_idx, i0 + 1, comp));
    return static_cast<svmp::FE::Real>((1.0 - alpha) * v0 + alpha * v1);
  }

  [[nodiscard]] svmp::FE::Real interpolateAlongX(svmp::FE::Real x, svmp::FE::Real time, int comp) const
  {
    if (!has_x_only_interpolant || x_interpolation_coords.empty()) {
      throw std::runtime_error("[svMultiPhysics::Physics] Internal error: missing x-axis temporal/spatial interpolant.");
    }

    if (x <= x_interpolation_coords.front()) {
      return interpolate(x_interpolation_nodes.front(), time, comp);
    }
    if (x >= x_interpolation_coords.back()) {
      return interpolate(x_interpolation_nodes.back(), time, comp);
    }

    const auto upper = std::upper_bound(x_interpolation_coords.begin(), x_interpolation_coords.end(), x);
    const auto hi = static_cast<std::size_t>(std::distance(x_interpolation_coords.begin(), upper));
    const auto lo = hi - 1;
    const double x0 = static_cast<double>(x_interpolation_coords[lo]);
    const double x1 = static_cast<double>(x_interpolation_coords[hi]);
    const double denom = x1 - x0;
    const double alpha = (denom > 0.0) ? ((static_cast<double>(x) - x0) / denom) : 0.0;
    const double v0 = static_cast<double>(interpolate(x_interpolation_nodes[lo], time, comp));
    const double v1 = static_cast<double>(interpolate(x_interpolation_nodes[hi], time, comp));
    return static_cast<svmp::FE::Real>((1.0 - alpha) * v0 + alpha * v1);
  }

  [[nodiscard]] svmp::FE::Real interpolateNearestSpatial(const std::array<svmp::FE::Real, 3>& p,
                                                         svmp::FE::Real time,
                                                         int comp) const
  {
    if (coords.empty()) {
      return svmp::FE::Real{0.0};
    }

    const auto distances = nearestNodes(p);
    const auto k = distances.size();

    constexpr double exact_tol2 = 1.0e-16;
    if (distances.front().first <= exact_tol2) {
      return interpolate(distances.front().second, time, comp);
    }

    double weighted_sum = 0.0;
    double weight_total = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      const double w = 1.0 / std::max(distances[i].first, 1.0e-24);
      weighted_sum += w * static_cast<double>(interpolate(distances[i].second, time, comp));
      weight_total += w;
    }
    return static_cast<svmp::FE::Real>(weight_total > 0.0 ? weighted_sum / weight_total : 0.0);
  }

  [[nodiscard]] svmp::FE::Real interpolateSpatial(const std::array<svmp::FE::Real, 3>& p,
                                                  svmp::FE::Real time,
                                                  int comp) const
  {
    const auto key = quantize(p, dim);
    if (const auto it = node_index_by_key.find(key); it != node_index_by_key.end()) {
      return interpolate(it->second, time, comp);
    }
    if (has_x_only_interpolant) {
      return interpolateAlongX(p[0], time, comp);
    }
    return interpolateNearestSpatial(p, time, comp);
  }
};

std::shared_ptr<TemporalSpatialValues> read_temporal_and_spatial_values_file(const svmp::MeshBase& mesh,
                                                                            int boundary_marker,
                                                                            const std::string& file_path,
                                                                            MarkerCommunicator comm);

svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType
parse_node_pressure_constraint_id_type(std::string_view raw);

std::vector<svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::NodePressureConstraint>
read_node_pressure_constraints_csv(const std::string& path);

struct Vec3d {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

inline Vec3d to_vec3(const std::array<svmp::FE::Real, 3>& v)
{
  return Vec3d{static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

inline std::array<svmp::FE::Real, 3> to_array(const Vec3d& v)
{
  return {static_cast<svmp::FE::Real>(v.x), static_cast<svmp::FE::Real>(v.y), static_cast<svmp::FE::Real>(v.z)};
}

inline Vec3d operator+(const Vec3d& a, const Vec3d& b) { return Vec3d{a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3d operator-(const Vec3d& a, const Vec3d& b) { return Vec3d{a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3d operator*(double s, const Vec3d& a) { return Vec3d{s * a.x, s * a.y, s * a.z}; }

inline double dot(const Vec3d& a, const Vec3d& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline double norm2(const Vec3d& a) { return dot(a, a); }

inline double norm(const Vec3d& a) { return std::sqrt(norm2(a)); }

inline Vec3d cross(const Vec3d& a, const Vec3d& b)
{
  return Vec3d{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline Vec3d normalized(const Vec3d& v)
{
  const double n = norm(v);
  if (!(n > 0.0)) {
    return Vec3d{};
  }
  return (1.0 / n) * v;
}

struct MarkerGeometry {
  double area{0.0};
  Vec3d center_sum{};
  Vec3d normal_sum{};
};

MarkerGeometry local_marker_geometry(const svmp::MeshBase& mesh, int boundary_marker);

[[nodiscard]] MarkerCommunicator markerCommunicator(
    const svmp::FE::systems::FESystem& system) noexcept;

MarkerGeometry global_marker_geometry(const svmp::MeshBase& mesh,
                                      int boundary_marker,
                                      MarkerCommunicator comm);

struct ParabolicProfileData {
  Vec3d center{};
  std::vector<Vec3d> perimeter_unit_dirs{};
  std::vector<double> perimeter_r2{};
};

ParabolicProfileData build_parabolic_profile_data(const svmp::MeshBase& mesh,
                                                  int boundary_marker,
                                                  MarkerCommunicator comm);
double parabolic_weight(const ParabolicProfileData& data, const Vec3d& x);
double integrate_parabolic_weight_over_marker(const svmp::MeshBase& mesh,
                                              int boundary_marker,
                                              const ParabolicProfileData& data,
                                              MarkerCommunicator comm);

enum class InletProfileType { Flat, Parabolic };

struct InletProfileContext {
  int dim{0};
  InletProfileType profile{InletProfileType::Flat};
  bool use_normal_direction{true};
  std::array<int, 3> active_components{1, 1, 1}; // used when use_normal_direction==false

  Vec3d normal{};   // unit
  double scale{0.0};

  std::optional<ParabolicProfileData> parabolic{};

  double weight(const Vec3d& x) const
  {
    switch (profile) {
      case InletProfileType::Flat: return 1.0;
      case InletProfileType::Parabolic:
        if (!parabolic.has_value()) {
          return 0.0;
        }
        return parabolic_weight(*parabolic, x);
    }
    return 1.0;
  }

  double componentValue(int component, const Vec3d& x) const
  {
    if (component < 0 || component >= 3) {
      return 0.0;
    }
    const double w = weight(x);
    if (use_normal_direction) {
      const double nd = (component == 0) ? normal.x : (component == 1) ? normal.y : normal.z;
      return scale * w * nd;
    }
    const int active = active_components[static_cast<std::size_t>(component)];
    if (active == 0) {
      return 0.0;
    }
    return scale * w;
  }
};

template <class ScalarValue>
void fill_vector(std::array<ScalarValue, 3>& dst, int dim, const std::vector<int>& effective_dir,
                 svmp::FE::Real magnitude)
{
  dst = {ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
  for (int d = 0; d < dim; ++d) {
    const auto scale =
        effective_dir.empty() ? static_cast<svmp::FE::Real>(1.0) : direction_component(effective_dir, d);
    dst[static_cast<std::size_t>(d)] = ScalarValue{static_cast<svmp::FE::Real>(magnitude * scale)};
  }
}

} // namespace svmp::Physics::options

#endif // SVMP_PHYSICS_OPTIONS_BOUNDARY_DATA_READERS_H

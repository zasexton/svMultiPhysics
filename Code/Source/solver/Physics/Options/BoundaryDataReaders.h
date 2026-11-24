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
#include <span>
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

  /// The values at @p time at the mesh vertices of the stored nodes: component
  /// c of the node at local vertex v goes to out[v * components + c], and the
  /// components past the table's are zero.
  void fillVertexValues(svmp::FE::Real time, int components, std::span<svmp::FE::Real> out) const
  {
    const auto n = static_cast<std::size_t>(components);
    for (std::size_t node = 0; node < node_ids.size(); ++node) {
      const auto base = static_cast<std::size_t>(node_ids[node]) * n;
      if (node_ids[node] < 0 || base + n > out.size()) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Internal error: temporal/spatial values node outside the vertex range.");
      }
      for (int c = 0; c < components; ++c) {
        out[base + static_cast<std::size_t>(c)] =
            c < dof ? interpolate(node, time, c) : svmp::FE::Real{0.0};
      }
    }
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

/**
 * @brief The strong velocity data of an unsteady inflow
 *
 * The flow rate @p flow_rate becomes a time function of the system
 * (declared through dir.time_functions) and component d of the velocity the
 * flow rate times the profile of @p ctx, a coefficient of space.  At every
 * node and time each value is the product the time callback of a weak inflow
 * forms, bit for bit.
 */
void set_strong_unsteady_inflow(
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC& dir,
    int dim,
    std::shared_ptr<const svmp::FE::systems::TimeSeries> flow_rate,
    std::shared_ptr<const InletProfileContext> ctx);

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

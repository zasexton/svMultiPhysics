#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Core/TemporalValues.h"
#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"

#include "FE/Core/Logger.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace {

std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool parse_bool_relaxed(std::string_view raw)
{
  const auto v = lower_copy(trim_copy(std::string(raw)));
  if (v == "true" || v == "1" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "false" || v == "0" || v == "no" || v == "off") {
    return false;
  }
  return false;
}

[[nodiscard]] bool navierStokesTraceEnabled() noexcept
{
  const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE");
  return env != nullptr && env[0] != '\0';
}

void navierStokesTraceLog(const std::string& message)
{
  if (navierStokesTraceEnabled()) {
    FE_LOG_INFO(message);
  }
}

[[nodiscard]] bool temporalSpatialBcTraceEnabled() noexcept
{
  return navierStokesTraceEnabled();
}

double parse_double(std::string_view raw, std::string_view context)
{
  const auto s = trim_copy(std::string(raw));
  try {
    size_t pos = 0;
    const double v = std::stod(s, &pos);
    if (pos != s.size()) {
      throw std::runtime_error("");
    }
    return v;
  } catch (...) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse numeric value '" + std::string(raw) +
                             "' for " + std::string(context) + ".");
  }
}

int parse_positive_int(std::string_view raw, std::string_view context)
{
  const auto s = trim_copy(std::string(raw));
  try {
    size_t pos = 0;
    const int v = std::stoi(s, &pos);
    if (pos != s.size() || v < 1) {
      throw std::runtime_error("");
    }
    return v;
  } catch (...) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse positive integer value '" +
                             std::string(raw) + "' for " + std::string(context) + ".");
  }
}

const svmp::Physics::ParameterValue* find_param(const svmp::Physics::ParameterMap& params,
                                                std::string_view key)
{
  const auto it = params.find(std::string(key));
  if (it == params.end()) {
    return nullptr;
  }
  return &it->second;
}

bool has_nonempty_defined(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p) {
    return false;
  }
  return p->defined && !trim_copy(p->value).empty();
}

std::optional<double> get_defined_double(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p || !p->defined) {
    return std::nullopt;
  }
  return parse_double(p->value, key);
}

std::optional<bool> get_defined_bool(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p || !p->defined) {
    return std::nullopt;
  }
  return parse_bool_relaxed(p->value);
}

std::optional<std::string> get_defined_string(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p || !p->defined) {
    return std::nullopt;
  }
  auto value = trim_copy(p->value);
  if (value.empty()) {
    return std::nullopt;
  }
  return value;
}

svmp::FE::forms::GeometryTangentPath parse_geometry_tangent_path(std::string_view raw,
                                                                 std::string_view context)
{
  const auto path = lower_copy(trim_copy(std::string(raw)));
  if (path == "symbolic" || path == "symbolic_required" || path == "required") {
    return svmp::FE::forms::GeometryTangentPath::SymbolicRequired;
  }
  if (path == "ad" || path == "ad_reference" || path == "reference_ad") {
    return svmp::FE::forms::GeometryTangentPath::ADReference;
  }
  if (path == "symbolic_ad_check" || path == "symbolic_with_ad_check" ||
      path == "check" || path == "parity_check") {
    return svmp::FE::forms::GeometryTangentPath::SymbolicWithADCheck;
  }
  if (path == "auto") {
    return svmp::FE::forms::GeometryTangentPath::Auto;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of 'symbolic', 'ad', 'symbolic_ad_check', or 'auto'.");
}

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

    int i0 = 0;
    for (int i = 0; i < num_time_points - 1; ++i) {
      if (t[static_cast<std::size_t>(i + 1)] >= tt) {
        i0 = i;
        break;
      }
    }

    const double t0 = t[static_cast<std::size_t>(i0)];
    const double t1 = t[static_cast<std::size_t>(i0 + 1)];
    const double dt = t1 - t0;
    const double alpha = (dt > 0.0) ? ((tt - t0) / dt) : 0.0;

    const auto v0 = static_cast<double>(sample(node_idx, i0, comp));
    const auto v1 = static_cast<double>(sample(node_idx, i0 + 1, comp));
    return static_cast<svmp::FE::Real>((1.0 - alpha) * v0 + alpha * v1);
  }
};

std::unordered_set<svmp::gid_t> collect_boundary_vertex_gids(const svmp::MeshBase& mesh, int boundary_marker)
{
  std::unordered_set<svmp::gid_t> gids;
  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
  const auto& vgids = mesh.vertex_gids();
  for (const auto f : faces) {
    for (const auto v : mesh.face_vertices(f)) {
      if (v < 0) {
        continue;
      }
      const auto idx = static_cast<std::size_t>(v);
      if (idx >= vgids.size()) {
        continue;
      }
      gids.insert(vgids[idx]);
    }
  }
  return gids;
}

std::shared_ptr<TemporalSpatialValues> read_temporal_and_spatial_values_file(const svmp::MeshBase& mesh,
                                                                            int boundary_marker,
                                                                            const std::string& file_path)
{
  std::ifstream in(file_path);
  if (!in.is_open()) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to open temporal/spatial BC file '" + file_path + "'.");
  }

  int ndof = 0;
  int num_ts = 0;
  int num_nodes = 0;
  in >> ndof >> num_ts >> num_nodes;
  if (ndof <= 0 || num_ts <= 0 || num_nodes <= 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Invalid header in temporal/spatial BC file '" + file_path +
        "' (expected: <ndof> <num_ts> <num_nodes>).");
  }

  const int dim = mesh.dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error("[svMultiPhysics::Physics] Invalid mesh dimension for temporal/spatial BC parsing.");
  }
  if (ndof > dim) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "' specifies ndof=" + std::to_string(ndof) +
        ", but mesh dimension is " + std::to_string(dim) + ".");
  }

  const auto boundary_gids = collect_boundary_vertex_gids(mesh, boundary_marker);
  const bool identity_vertex_gids = [&] {
    const auto& vgids = mesh.vertex_gids();
    if (vgids.size() != mesh.n_vertices()) {
      return false;
    }
    for (std::size_t i = 0; i < vgids.size(); ++i) {
      if (vgids[i] != static_cast<svmp::gid_t>(i)) {
        return false;
      }
    }
    return true;
  }();

  if (identity_vertex_gids && !boundary_gids.empty() && static_cast<int>(boundary_gids.size()) != num_nodes) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "' specifies num_nodes=" +
        std::to_string(num_nodes) + ", but boundary marker " + std::to_string(boundary_marker) + " has " +
        std::to_string(boundary_gids.size()) + " unique nodes.");
  }

  auto out = std::make_shared<TemporalSpatialValues>();
  out->dim = dim;
  out->dof = ndof;
  out->num_time_points = num_ts;
  out->boundary_marker = boundary_marker;
  out->file_path = file_path;
  out->t.resize(static_cast<std::size_t>(num_ts));

  // Time sequence (t0 must be 0 and increasing).
  for (int i = 0; i < num_ts; ++i) {
    double ti = 0.0;
    in >> ti;
    out->t[static_cast<std::size_t>(i)] = ti;
    if (i == 0) {
      if (std::abs(ti) > 1e-14) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "': first time value must be 0.");
      }
    } else {
      const double dt = ti - out->t[static_cast<std::size_t>(i - 1)];
      if (!(dt > 0.0)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "': time values must be increasing.");
      }
    }
  }
  out->period = out->t.back();

  out->node_ids.clear();
  out->coords.clear();
  out->d.clear();
  out->node_ids.reserve(static_cast<std::size_t>(num_nodes));
  out->coords.reserve(static_cast<std::size_t>(num_nodes));
  out->d.reserve(static_cast<std::size_t>(num_nodes) * static_cast<std::size_t>(num_ts) * static_cast<std::size_t>(ndof));

  int missing_local_vertex_count = 0;
  int non_boundary_file_node_count = 0;
  int raw_gid_fallback_count = 0;

  for (int b = 0; b < num_nodes; ++b) {
    long long node_id_1based = 0;
    in >> node_id_1based;
    const long long node_gid0_ll = node_id_1based - 1;
    if (node_gid0_ll < 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "': invalid negative node id: " + std::to_string(node_id_1based) + ".");
    }

    struct NodeCandidate {
      svmp::gid_t gid{svmp::INVALID_GID};
      svmp::index_t local{svmp::INVALID_INDEX};
      bool on_boundary{false};
    };
    const auto make_candidate = [&](svmp::gid_t gid) {
      NodeCandidate c{};
      c.gid = gid;
      c.local = mesh.global_to_local_vertex(gid);
      c.on_boundary = boundary_gids.empty() || boundary_gids.count(gid) != 0u;
      return c;
    };

    const auto zero_based = make_candidate(static_cast<svmp::gid_t>(node_gid0_ll));
    NodeCandidate chosen = zero_based;
    bool used_raw_gid = false;
    if (!identity_vertex_gids) {
      const auto raw = make_candidate(static_cast<svmp::gid_t>(node_id_1based));
      if (!(zero_based.local != svmp::INVALID_INDEX && zero_based.on_boundary)) {
        if (raw.local != svmp::INVALID_INDEX && raw.on_boundary) {
          chosen = raw;
          used_raw_gid = true;
        } else if (zero_based.local == svmp::INVALID_INDEX && raw.local != svmp::INVALID_INDEX) {
          chosen = raw;
          used_raw_gid = true;
        }
      }
    }

    const auto node_gid = chosen.gid;
    const auto node_idx = chosen.local;
    if (used_raw_gid) {
      ++raw_gid_fallback_count;
    }

    if (identity_vertex_gids && node_idx == svmp::INVALID_INDEX) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "': node id out of range: " + std::to_string(node_id_1based) + ".");
    }
    if (identity_vertex_gids && node_idx != svmp::INVALID_INDEX && !boundary_gids.empty() &&
        boundary_gids.count(node_gid) == 0u) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "': node id " +
          std::to_string(node_id_1based) + " is not on boundary marker " + std::to_string(boundary_marker) + ".");
    }

    const bool has_local_vertex = node_idx != svmp::INVALID_INDEX;
    const bool is_boundary_node = chosen.on_boundary;
    if (!has_local_vertex) {
      ++missing_local_vertex_count;
    } else if (!is_boundary_node) {
      ++non_boundary_file_node_count;
    }
    const bool keep = has_local_vertex && is_boundary_node;
    if (keep) {
      out->node_ids.push_back(node_idx);

      const auto& X = mesh.X_ref();
      std::array<svmp::FE::Real, 3> p{0.0, 0.0, 0.0};
      const auto base = static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(dim);
      p[0] = static_cast<svmp::FE::Real>(X.at(base + 0));
      if (dim >= 2) {
        p[1] = static_cast<svmp::FE::Real>(X.at(base + 1));
      }
      if (dim >= 3) {
        p[2] = static_cast<svmp::FE::Real>(X.at(base + 2));
      }
      out->coords.push_back(p);

      const auto stored_idx = out->coords.size() - 1;
      out->node_index_by_key.emplace(TemporalSpatialValues::quantize(p, dim), stored_idx);
    }

    for (int i = 0; i < num_ts; ++i) {
      for (int k = 0; k < ndof; ++k) {
        double value = 0.0;
        in >> value;
        if (keep) {
          out->d.push_back(static_cast<svmp::FE::Real>(value));
        }
      }
    }
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
        << " ndof=" << ndof
        << " time_points=" << num_ts
        << " file_nodes=" << num_nodes
        << " kept_nodes=" << out->coords.size()
        << " boundary_marker_nodes=" << boundary_gids.size()
        << " missing_local_vertex_nodes=" << missing_local_vertex_count
        << " non_boundary_file_nodes=" << non_boundary_file_node_count
        << " raw_gid_fallback_nodes=" << raw_gid_fallback_count
        << " identity_vertex_gids=" << (identity_vertex_gids ? 1 : 0);
    navierStokesTraceLog(oss.str());
  }

  return out;
}

svmp::FE::ElementType infer_base_element_type(const svmp::MeshBase& mesh)
{
  if (mesh.n_cells() == 0) {
    throw std::runtime_error("[svMultiPhysics::Physics] Mesh has no cells; cannot infer FE element type.");
  }

  const auto& shapes = mesh.cell_shapes();
  if (shapes.empty()) {
    throw std::runtime_error("[svMultiPhysics::Physics] Mesh has no cell shapes; cannot infer FE element type.");
  }

  if (shapes.front().is_mixed_order) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Mixed-order meshes are not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto family = shapes.front().family;
  for (const auto& s : shapes) {
    if (s.family != family) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Mixed cell families are not supported by the new solver yet. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
  }

  switch (family) {
    case svmp::CellFamily::Line: return svmp::FE::ElementType::Line2;
    case svmp::CellFamily::Triangle: return svmp::FE::ElementType::Triangle3;
    case svmp::CellFamily::Quad: return svmp::FE::ElementType::Quad4;
    case svmp::CellFamily::Tetra: return svmp::FE::ElementType::Tetra4;
    case svmp::CellFamily::Hex: return svmp::FE::ElementType::Hex8;
    case svmp::CellFamily::Wedge: return svmp::FE::ElementType::Wedge6;
    case svmp::CellFamily::Pyramid: return svmp::FE::ElementType::Pyramid5;
    default:
      break;
  }

  throw std::runtime_error(
      "[svMultiPhysics::Physics] Unsupported mesh cell family for new solver. "
      "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
}

int infer_polynomial_order(const svmp::MeshBase& mesh)
{
  const auto& shapes = mesh.cell_shapes();
  if (shapes.empty()) {
    return 1;
  }

  const int order = shapes.front().order > 0 ? shapes.front().order : 1;
  for (const auto& s : shapes) {
    const int s_order = s.order > 0 ? s.order : 1;
    if (s_order != order) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Mixed polynomial orders are not supported by the new solver yet. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
  }

  return order;
}

int resolve_element_order(const svmp::Physics::EquationModuleInput& input, int inferred_order)
{
  if (const auto* p = find_param(input.equation_params, "Element_order"); p && p->defined) {
    return parse_positive_int(p->value, "Element_order");
  }
  return inferred_order;
}

const svmp::Physics::DomainInput& select_single_domain(const svmp::Physics::EquationModuleInput& input,
                                                       std::string_view module_name)
{
  if (!input.domains.empty()) {
    if (input.domains.size() != 1) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Multiple <Domain> blocks are not supported for the new solver " +
          std::string(module_name) +
          " module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
    return input.domains.front();
  }
  return input.default_domain;
}

void apply_fluid_properties(const svmp::Physics::DomainInput& domain,
                            svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  if (const auto rho = get_defined_double(domain.params, "Density")) {
    options.density = static_cast<svmp::FE::Real>(*rho);
  } else if (const auto rho2 = get_defined_double(domain.params, "Fluid_density")) {
    options.density = static_cast<svmp::FE::Real>(*rho2);
  }

  if (const auto fx = get_defined_double(domain.params, "Force_x")) {
    options.body_force[0] = static_cast<svmp::FE::Real>(*fx);
  }
  if (const auto fy = get_defined_double(domain.params, "Force_y")) {
    options.body_force[1] = static_cast<svmp::FE::Real>(*fy);
  }
  if (const auto fz = get_defined_double(domain.params, "Force_z")) {
    options.body_force[2] = static_cast<svmp::FE::Real>(*fz);
  }

  const auto* model_param = find_param(domain.params, "Viscosity.model");
  if (!model_param || !model_param->defined || trim_copy(model_param->value).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] <Viscosity model=\"...\"> is required for the new solver Navier-Stokes module. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto model_raw = trim_copy(model_param->value);
  const auto model = lower_copy(model_raw);

  if (model == "constant") {
    const auto* mu_param = find_param(domain.params, "Viscosity.Value");
    const double mu = mu_param ? parse_double(mu_param->value, "Viscosity/Value") : 0.0;
    if (!(mu > 0.0)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Constant viscosity must be > 0 for the new solver Navier-Stokes module. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
    options.viscosity = static_cast<svmp::FE::Real>(mu);
    options.viscosity_model.reset();
    return;
  }

  if (model == "carreau-yasuda" || model == "carreau_yasuda") {
    const auto* p_mu_inf = find_param(domain.params, "Viscosity.Limiting_high_shear_rate_viscosity");
    const auto* p_mu0 = find_param(domain.params, "Viscosity.Limiting_low_shear_rate_viscosity");
    const auto* p_lambda = find_param(domain.params, "Viscosity.Shear_rate_tensor_multiplier");
    const auto* p_n = find_param(domain.params, "Viscosity.Power_law_index");
    const auto* p_a = find_param(domain.params, "Viscosity.Shear_rate_tensor_exponent");

    const auto mu_inf = static_cast<svmp::FE::Real>(
        p_mu_inf ? parse_double(p_mu_inf->value, "Viscosity/Limiting_high_shear_rate_viscosity") : 0.0);
    const auto mu0 = static_cast<svmp::FE::Real>(
        p_mu0 ? parse_double(p_mu0->value, "Viscosity/Limiting_low_shear_rate_viscosity") : 0.0);
    const auto lambda = static_cast<svmp::FE::Real>(
        p_lambda ? parse_double(p_lambda->value, "Viscosity/Shear_rate_tensor_multiplier") : 0.0);
    const auto n =
        static_cast<svmp::FE::Real>(p_n ? parse_double(p_n->value, "Viscosity/Power_law_index") : 0.0);
    const auto a = static_cast<svmp::FE::Real>(
        p_a ? parse_double(p_a->value, "Viscosity/Shear_rate_tensor_exponent") : 0.0);

    try {
      options.viscosity_model = std::make_shared<svmp::Physics::materials::fluid::CarreauYasudaViscosity>(
          mu0, mu_inf, lambda, n, a);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string("[svMultiPhysics::Physics] Invalid Carreau-Yasuda viscosity parameters for the new solver "
                      "Navier-Stokes module: ") +
          e.what() + ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (p_mu0 && p_mu0->defined) {
      options.viscosity = mu0;
    }
    return;
  }

  if (model == "cassons" || model == "casson") {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Cassons viscosity model is not supported by the new solver Navier-Stokes module yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  throw std::runtime_error(
      "[svMultiPhysics::Physics] Fluid viscosity model '" + model_raw +
      "' is not supported by the new solver Navier-Stokes module. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> "
      "to use the legacy solver.");
}

void apply_fluid_moving_domain_params(
    const svmp::Physics::ParameterMap& params,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  namespace ns = svmp::Physics::formulations::navier_stokes;

  constexpr std::array<std::string_view, 5> kAleKeys = {
      "ALE",
      "Enable_ALE",
      "Use_ALE",
      "Moving_mesh",
      "Use_moving_mesh",
  };
  for (const auto key : kAleKeys) {
    if (const auto value = get_defined_bool(params, key)) {
      options.enable_ale = *value;
    }
  }

  constexpr std::array<std::string_view, 3> kMovingVolumeKeys = {
      "Moving_control_volume_transient",
      "Include_moving_control_volume_transient",
      "ALE_moving_control_volume_transient",
  };
  for (const auto key : kMovingVolumeKeys) {
    if (const auto value = get_defined_bool(params, key)) {
      options.include_moving_control_volume_transient = *value;
    }
  }

  constexpr std::array<std::string_view, 3> kMeshVelocityFieldKeys = {
      "Mesh_velocity_field",
      "MeshVelocityField",
      "Mesh_motion_velocity_field",
  };
  for (const auto key : kMeshVelocityFieldKeys) {
    if (const auto value = get_defined_string(params, key)) {
      options.mesh_velocity_field_name = *value;
    }
  }

  constexpr std::array<std::string_view, 4> kMeshVelocitySourceKeys = {
      "Mesh_velocity_source",
      "MeshVelocitySource",
      "ALE_mesh_velocity_source",
      "ALEMeshVelocitySource",
  };
  for (const auto key : kMeshVelocitySourceKeys) {
    if (const auto value = get_defined_string(params, key)) {
      const auto source = lower_copy(trim_copy(*value));
      if (source == "prescribed" || source == "prescribed_data" ||
          source == "data" || source == "mesh_motion_data") {
        options.mesh_velocity_source = ns::ALEMeshVelocitySource::PrescribedData;
      } else if (source == "coupled" || source == "coupled_displacement" ||
                 source == "derived" || source == "derived_from_displacement" ||
                 source == "monolithic") {
        options.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
      } else {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Mesh_velocity_source must be one of "
            "'prescribed_data' or 'coupled_displacement'.");
      }
    }
  }

  constexpr std::array<std::string_view, 3> kMeshDisplacementFieldKeys = {
      "Mesh_displacement_field",
      "MeshDisplacementField",
      "Mesh_motion_displacement_field",
  };
  for (const auto key : kMeshDisplacementFieldKeys) {
    if (const auto value = get_defined_string(params, key)) {
      options.mesh_displacement_field_name = *value;
    }
  }

  constexpr std::array<std::string_view, 3> kAutoRegisterMeshDisplacementKeys = {
      "Auto_register_mesh_displacement_field",
      "AutoRegisterMeshDisplacementField",
      "ALE_auto_register_mesh_displacement_field",
  };
  for (const auto key : kAutoRegisterMeshDisplacementKeys) {
    if (const auto value = get_defined_bool(params, key)) {
      options.auto_register_mesh_displacement_field = *value;
    }
  }

  constexpr std::array<std::string_view, 4> kMovingMeshTangentPathKeys = {
      "MovingMeshTangentPath",
      "Moving_mesh_tangent_path",
      "Moving_mesh_geometry_tangent_path",
      "ALE_moving_mesh_tangent_path",
  };
  for (const auto key : kMovingMeshTangentPathKeys) {
    if (const auto value = get_defined_string(params, key)) {
      options.moving_mesh_tangent_path =
          parse_geometry_tangent_path(*value, key);
    }
  }
}

void apply_fluid_moving_domain_options(
    const svmp::Physics::EquationModuleInput& input,
    const svmp::Physics::DomainInput& domain,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  apply_fluid_moving_domain_params(input.equation_params, options);
  apply_fluid_moving_domain_params(domain.params, options);
  if (trim_copy(options.mesh_velocity_field_name).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Mesh_velocity_field must be non-empty when configuring Navier-Stokes ALE.");
  }
  if (options.mesh_velocity_source ==
          svmp::Physics::formulations::navier_stokes::ALEMeshVelocitySource::CoupledDisplacement &&
      trim_copy(options.mesh_displacement_field_name).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Mesh_displacement_field must be non-empty when configuring coupled ALE.");
  }
}

std::vector<int> parse_int_list(std::string_view raw)
{
  std::istringstream iss{std::string(raw)};
  std::vector<int> out;
  int v = 0;
  while (iss >> v) {
    out.push_back(v);
  }
  return out;
}

svmp::FE::Real direction_component(const std::vector<int>& effective_dir, int component)
{
  if (component < 0) {
    return static_cast<svmp::FE::Real>(0.0);
  }
  const auto idx = static_cast<std::size_t>(component);
  if (idx >= effective_dir.size()) {
    return static_cast<svmp::FE::Real>(1.0);
  }
  return static_cast<svmp::FE::Real>(effective_dir[idx]);
}

struct Vec3d {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

Vec3d to_vec3(const std::array<svmp::FE::Real, 3>& v)
{
  return Vec3d{static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

std::array<svmp::FE::Real, 3> to_array(const Vec3d& v)
{
  return {static_cast<svmp::FE::Real>(v.x), static_cast<svmp::FE::Real>(v.y), static_cast<svmp::FE::Real>(v.z)};
}

Vec3d operator+(const Vec3d& a, const Vec3d& b) { return Vec3d{a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3d operator-(const Vec3d& a, const Vec3d& b) { return Vec3d{a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3d operator*(double s, const Vec3d& a) { return Vec3d{s * a.x, s * a.y, s * a.z}; }

double dot(const Vec3d& a, const Vec3d& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

double norm2(const Vec3d& a) { return dot(a, a); }

double norm(const Vec3d& a) { return std::sqrt(norm2(a)); }

Vec3d cross(const Vec3d& a, const Vec3d& b)
{
  return Vec3d{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vec3d normalized(const Vec3d& v)
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

MarkerGeometry local_marker_geometry(const svmp::MeshBase& mesh, int boundary_marker)
{
  MarkerGeometry out{};
  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
  for (const auto f : faces) {
    const double a = static_cast<double>(mesh.face_area(f));
    const auto c = to_vec3(mesh.face_center(f));
    const auto n = to_vec3(mesh.face_normal(f));
    out.area += a;
    out.center_sum = out.center_sum + (a * c);
    out.normal_sum = out.normal_sum + (a * n);
  }
  return out;
}

MarkerGeometry global_marker_geometry(const svmp::MeshBase& mesh, int boundary_marker);

struct ParabolicProfileData {
  Vec3d center{};
  std::vector<Vec3d> perimeter_unit_dirs{};
  std::vector<double> perimeter_r2{};
};

struct GidPairHash {
  std::size_t operator()(const std::pair<svmp::gid_t, svmp::gid_t>& p) const noexcept
  {
    const std::size_t h1 = std::hash<svmp::gid_t>{}(p.first);
    const std::size_t h2 = std::hash<svmp::gid_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct GidSequenceHash {
  std::size_t operator()(const std::vector<svmp::gid_t>& values) const noexcept
  {
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto value : values) {
      const auto mixed = static_cast<std::uint64_t>(std::hash<svmp::gid_t>{}(value));
      hash ^= mixed + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    hash ^= static_cast<std::uint64_t>(values.size()) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return static_cast<std::size_t>(hash);
  }
};

[[nodiscard]] bool mpiMultiRankActive() noexcept
{
#if FE_HAS_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    return false;
  }

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  return world_size > 1;
#else
  return false;
#endif
}

template <std::size_t N>
struct MarkerFacePayload {
  std::vector<svmp::gid_t> ordered_vertex_gids{};
  std::array<double, N> values{};
};

template <std::size_t N, class PayloadBuilder>
std::vector<MarkerFacePayload<N>> gather_unique_marker_face_payloads(
    const svmp::MeshBase& mesh,
    int boundary_marker,
    PayloadBuilder&& payload_builder)
{
  const bool mpi_active = mpiMultiRankActive();
  const auto& vgids = mesh.vertex_gids();
  if (mpi_active && vgids.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] MPI boundary-face normalization for marker " +
        std::to_string(boundary_marker) + " requires stable vertex GIDs.");
  }

  std::vector<MarkerFacePayload<N>> local_payloads;
  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
  local_payloads.reserve(faces.size());

  for (const auto f : faces) {
    MarkerFacePayload<N> payload;
    const auto verts = mesh.face_vertices(f);
    payload.ordered_vertex_gids.reserve(verts.size());
    for (const auto v : verts) {
      if (v == svmp::INVALID_INDEX) {
        continue;
      }
      const auto idx = static_cast<std::size_t>(v);
      if (idx >= mesh.n_vertices()) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Boundary face references an out-of-range vertex while processing marker " +
            std::to_string(boundary_marker) + ".");
      }
      const auto gid = (vgids.size() == mesh.n_vertices())
                           ? vgids[idx]
                           : static_cast<svmp::gid_t>(v);
      payload.ordered_vertex_gids.push_back(gid);
    }
    if (payload.ordered_vertex_gids.size() < 2u) {
      continue;
    }
    payload_builder(f, payload.values);
    local_payloads.push_back(std::move(payload));
  }

  auto deduplicate = [](std::vector<MarkerFacePayload<N>> payloads) {
    std::unordered_set<std::vector<svmp::gid_t>, GidSequenceHash> seen;
    std::vector<MarkerFacePayload<N>> unique;
    unique.reserve(payloads.size());
    for (auto& payload : payloads) {
      auto key = payload.ordered_vertex_gids;
      std::sort(key.begin(), key.end());
      key.erase(std::unique(key.begin(), key.end()), key.end());
      if (key.size() < 2u) {
        continue;
      }
      if (!seen.emplace(std::move(key)).second) {
        continue;
      }
      unique.push_back(std::move(payload));
    }
    return unique;
  };

  if (!mpi_active) {
    return deduplicate(std::move(local_payloads));
  }

#if FE_HAS_MPI
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int local_face_count = static_cast<int>(local_payloads.size());
  std::vector<int> face_counts(static_cast<std::size_t>(world_size), 0);
  MPI_Allgather(&local_face_count, 1, MPI_INT, face_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> face_displs(static_cast<std::size_t>(world_size), 0);
  int total_faces = 0;
  for (int r = 0; r < world_size; ++r) {
    face_displs[static_cast<std::size_t>(r)] = total_faces;
    total_faces += face_counts[static_cast<std::size_t>(r)];
  }

  std::vector<int> local_gid_sizes;
  local_gid_sizes.reserve(local_payloads.size());
  std::vector<svmp::gid_t> local_gid_data;
  for (const auto& payload : local_payloads) {
    local_gid_sizes.push_back(static_cast<int>(payload.ordered_vertex_gids.size()));
    local_gid_data.insert(local_gid_data.end(),
                          payload.ordered_vertex_gids.begin(),
                          payload.ordered_vertex_gids.end());
  }

  const int local_gid_count = static_cast<int>(local_gid_data.size());
  std::vector<int> gid_counts(static_cast<std::size_t>(world_size), 0);
  MPI_Allgather(&local_gid_count, 1, MPI_INT, gid_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> gid_displs(static_cast<std::size_t>(world_size), 0);
  int total_gid_count = 0;
  for (int r = 0; r < world_size; ++r) {
    gid_displs[static_cast<std::size_t>(r)] = total_gid_count;
    total_gid_count += gid_counts[static_cast<std::size_t>(r)];
  }

  std::vector<int> all_gid_sizes(static_cast<std::size_t>(total_faces), 0);
  MPI_Allgatherv(local_gid_sizes.data(),
                 local_face_count,
                 MPI_INT,
                 all_gid_sizes.data(),
                 face_counts.data(),
                 face_displs.data(),
                 MPI_INT,
                 MPI_COMM_WORLD);

  std::vector<svmp::gid_t> all_gid_data(static_cast<std::size_t>(total_gid_count), svmp::gid_t{0});
  MPI_Allgatherv(local_gid_data.data(),
                 local_gid_count,
                 MPI_INT64_T,
                 all_gid_data.data(),
                 gid_counts.data(),
                 gid_displs.data(),
                 MPI_INT64_T,
                 MPI_COMM_WORLD);

  std::vector<double> local_values;
  local_values.reserve(local_payloads.size() * N);
  for (const auto& payload : local_payloads) {
    local_values.insert(local_values.end(), payload.values.begin(), payload.values.end());
  }

  std::vector<int> value_counts(static_cast<std::size_t>(world_size), 0);
  std::vector<int> value_displs(static_cast<std::size_t>(world_size), 0);
  int total_value_count = 0;
  for (int r = 0; r < world_size; ++r) {
    value_displs[static_cast<std::size_t>(r)] = total_value_count;
    value_counts[static_cast<std::size_t>(r)] = face_counts[static_cast<std::size_t>(r)] * static_cast<int>(N);
    total_value_count += value_counts[static_cast<std::size_t>(r)];
  }

  std::vector<double> all_values(static_cast<std::size_t>(total_value_count), 0.0);
  MPI_Allgatherv(local_values.data(),
                 static_cast<int>(local_values.size()),
                 MPI_DOUBLE,
                 all_values.data(),
                 value_counts.data(),
                 value_displs.data(),
                 MPI_DOUBLE,
                 MPI_COMM_WORLD);

  std::vector<MarkerFacePayload<N>> gathered;
  gathered.reserve(static_cast<std::size_t>(total_faces));
  std::size_t gid_offset = 0u;
  for (int i = 0; i < total_faces; ++i) {
    const auto count = static_cast<std::size_t>(all_gid_sizes[static_cast<std::size_t>(i)]);
    if (gid_offset + count > all_gid_data.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Corrupt MPI face-GID gather while processing marker " +
          std::to_string(boundary_marker) + ".");
    }
    MarkerFacePayload<N> payload;
    payload.ordered_vertex_gids.assign(all_gid_data.begin() + static_cast<std::ptrdiff_t>(gid_offset),
                                       all_gid_data.begin() + static_cast<std::ptrdiff_t>(gid_offset + count));
    gid_offset += count;

    const std::size_t value_offset = static_cast<std::size_t>(i) * N;
    if (value_offset + N > all_values.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Corrupt MPI face-payload gather while processing marker " +
          std::to_string(boundary_marker) + ".");
    }
    std::copy_n(all_values.begin() + static_cast<std::ptrdiff_t>(value_offset), N, payload.values.begin());
    gathered.push_back(std::move(payload));
  }

  return deduplicate(std::move(gathered));
#else
  return deduplicate(std::move(local_payloads));
#endif
}

MarkerGeometry global_marker_geometry(const svmp::MeshBase& mesh, int boundary_marker)
{
  MarkerGeometry out{};
  const auto payloads = gather_unique_marker_face_payloads<7>(
      mesh, boundary_marker,
      [&](svmp::index_t f, std::array<double, 7>& values) {
        const double a = static_cast<double>(mesh.face_area(f));
        const auto c = to_vec3(mesh.face_center(f));
        const auto n = to_vec3(mesh.face_normal(f));
        values[0] = a;
        values[1] = a * c.x;
        values[2] = a * c.y;
        values[3] = a * c.z;
        values[4] = a * n.x;
        values[5] = a * n.y;
        values[6] = a * n.z;
      });
  for (const auto& payload : payloads) {
    out.area += payload.values[0];
    out.center_sum = out.center_sum + Vec3d{payload.values[1], payload.values[2], payload.values[3]};
    out.normal_sum = out.normal_sum + Vec3d{payload.values[4], payload.values[5], payload.values[6]};
  }
  return out;
}

std::vector<std::pair<svmp::gid_t, Vec3d>> gather_perimeter_vertex_coords(const svmp::MeshBase& mesh,
                                                                          int boundary_marker)
{
  const auto& vgids = mesh.vertex_gids();
  if (vgids.size() != mesh.n_vertices()) {
    return {};
  }

  // Determine the perimeter of the marker surface as the set of boundary edges of the
  // marker patch itself (edges that appear only once among unique marker faces).
  const auto marker_faces = gather_unique_marker_face_payloads<1>(
      mesh, boundary_marker,
      [](svmp::index_t /*face*/, std::array<double, 1>& values) {
        values[0] = 0.0;
      });

  std::vector<svmp::gid_t> local_edge_gids;
  local_edge_gids.reserve(marker_faces.size() * 6u);

  for (const auto& face : marker_faces) {
    const auto& face_vgids = face.ordered_vertex_gids;
    for (std::size_t i = 0; i < face_vgids.size(); ++i) {
      svmp::gid_t a = face_vgids[i];
      svmp::gid_t b = face_vgids[(i + 1u) % face_vgids.size()];
      if (a == b) {
        continue;
      }
      if (a > b) {
        std::swap(a, b);
      }
      local_edge_gids.push_back(a);
      local_edge_gids.push_back(b);
    }
  }

  const auto& all_edge_gids = local_edge_gids;

  std::unordered_map<std::pair<svmp::gid_t, svmp::gid_t>, int, GidPairHash> edge_counts;
  edge_counts.reserve(all_edge_gids.size() / 2u);
  for (std::size_t i = 0; i + 1u < all_edge_gids.size(); i += 2u) {
    const auto key = std::make_pair(all_edge_gids[i], all_edge_gids[i + 1u]);
    ++edge_counts[key];
  }

  std::unordered_set<svmp::gid_t> perimeter_gids;
  perimeter_gids.reserve(edge_counts.size() * 2u);
  for (const auto& [e, count] : edge_counts) {
    if (count == 1) {
      perimeter_gids.insert(e.first);
      perimeter_gids.insert(e.second);
    }
  }

  std::vector<svmp::gid_t> local_gids;
  std::vector<double> local_xyz;
  local_gids.reserve(perimeter_gids.size());
  local_xyz.reserve(perimeter_gids.size() * 3u);

  for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
    const auto idx = static_cast<std::size_t>(v);
    if (idx >= vgids.size()) {
      continue;
    }
    const auto gid = vgids[idx];
    if (perimeter_gids.find(gid) == perimeter_gids.end()) {
      continue;
    }
    local_gids.push_back(gid);
    const auto p = to_vec3(mesh.get_vertex_coords(v));
    local_xyz.push_back(p.x);
    local_xyz.push_back(p.y);
    local_xyz.push_back(p.z);
  }

  std::vector<svmp::gid_t> all_gids = local_gids;
  std::vector<double> all_xyz = local_xyz;

#if FE_HAS_MPI
  if (mpiMultiRankActive()) {
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int local_gid_count = static_cast<int>(local_gids.size());
    std::vector<int> gid_counts(static_cast<std::size_t>(world_size), 0);
    MPI_Allgather(&local_gid_count, 1, MPI_INT, gid_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> gid_displs(static_cast<std::size_t>(world_size), 0);
    int total_gid_count = 0;
    for (int r = 0; r < world_size; ++r) {
      gid_displs[static_cast<std::size_t>(r)] = total_gid_count;
      total_gid_count += gid_counts[static_cast<std::size_t>(r)];
    }

    all_gids.assign(static_cast<std::size_t>(total_gid_count), svmp::gid_t{0});
    MPI_Allgatherv(local_gids.data(),
                   local_gid_count,
                   MPI_INT64_T,
                   all_gids.data(),
                   gid_counts.data(),
                   gid_displs.data(),
                   MPI_INT64_T,
                   MPI_COMM_WORLD);

    std::vector<int> xyz_counts(static_cast<std::size_t>(world_size), 0);
    std::vector<int> xyz_displs(static_cast<std::size_t>(world_size), 0);
    int total_xyz_count = 0;
    for (int r = 0; r < world_size; ++r) {
      xyz_displs[static_cast<std::size_t>(r)] = total_xyz_count;
      xyz_counts[static_cast<std::size_t>(r)] = gid_counts[static_cast<std::size_t>(r)] * 3;
      total_xyz_count += xyz_counts[static_cast<std::size_t>(r)];
    }

    all_xyz.assign(static_cast<std::size_t>(total_xyz_count), 0.0);
    MPI_Allgatherv(local_xyz.data(),
                   static_cast<int>(local_xyz.size()),
                   MPI_DOUBLE,
                   all_xyz.data(),
                   xyz_counts.data(),
                   xyz_displs.data(),
                   MPI_DOUBLE,
                   MPI_COMM_WORLD);
  }
#endif

  std::vector<std::pair<svmp::gid_t, Vec3d>> out;
  out.reserve(all_gids.size());
  for (std::size_t i = 0; i < all_gids.size(); ++i) {
    const std::size_t j = i * 3u;
    if (j + 2u >= all_xyz.size()) {
      break;
    }
    out.emplace_back(all_gids[i], Vec3d{all_xyz[j + 0], all_xyz[j + 1], all_xyz[j + 2]});
  }
  return out;
}

ParabolicProfileData build_parabolic_profile_data(const svmp::MeshBase& mesh,
                                                  int boundary_marker)
{
  const auto g = global_marker_geometry(mesh, boundary_marker);
  if (!(g.area > 0.0)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary marker " + std::to_string(boundary_marker) +
        " has zero area; cannot construct parabolic profile.");
  }
  ParabolicProfileData out{};
  out.center = (1.0 / g.area) * g.center_sum;

  const auto all_perim = gather_perimeter_vertex_coords(mesh, boundary_marker);
  std::unordered_map<svmp::gid_t, Vec3d> unique;
  unique.reserve(all_perim.size());
  for (const auto& [gid, p] : all_perim) {
    if (unique.find(gid) == unique.end()) {
      unique.emplace(gid, p);
    }
  }

  out.perimeter_unit_dirs.clear();
  out.perimeter_r2.clear();
  out.perimeter_unit_dirs.reserve(unique.size());
  out.perimeter_r2.reserve(unique.size());

  for (const auto& [_, p] : unique) {
    const Vec3d v = p - out.center;
    const double r2 = norm2(v);
    if (!(r2 > 0.0)) {
      continue;
    }
    out.perimeter_r2.push_back(r2);
    out.perimeter_unit_dirs.push_back((1.0 / std::sqrt(r2)) * v);
  }

  if (out.perimeter_unit_dirs.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Parabolic inflow profile for boundary marker " + std::to_string(boundary_marker) +
        " requires a non-empty perimeter. Ensure this face shares an edge with another boundary face "
        "(e.g., a wall surface), or set <Profile>Flat</Profile>.");
  }

  return out;
}

double parabolic_weight(const ParabolicProfileData& data, const Vec3d& x)
{
  const Vec3d r = x - data.center;
  const double r2 = norm2(r);
  if (!(r2 > 0.0)) {
    return 1.0;
  }

  double best = -std::numeric_limits<double>::infinity();
  std::size_t best_i = 0;
  for (std::size_t i = 0; i < data.perimeter_unit_dirs.size(); ++i) {
    const double d = dot(r, data.perimeter_unit_dirs[i]);
    if (d > best) {
      best = d;
      best_i = i;
    }
  }

  const double R2 = data.perimeter_r2[best_i];
  if (!(R2 > 0.0)) {
    return 0.0;
  }

  const double w = 1.0 - (r2 / R2);
  return (w > 0.0) ? w : 0.0;
}

double integrate_parabolic_weight_over_marker(const svmp::MeshBase& mesh,
                                              int boundary_marker,
                                              const ParabolicProfileData& data)
{
  auto tri_area = [](const Vec3d& a, const Vec3d& b, const Vec3d& c) {
    const Vec3d ab = b - a;
    const Vec3d ac = c - a;
    return 0.5 * norm(cross(ab, ac));
  };

  auto tri_integral = [&](const Vec3d& p0, const Vec3d& p1, const Vec3d& p2) {
    const double a = tri_area(p0, p1, p2);
    if (!(a > 0.0)) {
      return 0.0;
    }
    // Degree-2 symmetric rule (3 points), weights sum to 1.
    const Vec3d x1 = (1.0 / 6.0) * p0 + (1.0 / 6.0) * p1 + (2.0 / 3.0) * p2;
    const Vec3d x2 = (1.0 / 6.0) * p0 + (2.0 / 3.0) * p1 + (1.0 / 6.0) * p2;
    const Vec3d x3 = (2.0 / 3.0) * p0 + (1.0 / 6.0) * p1 + (1.0 / 6.0) * p2;

    const double f = (parabolic_weight(data, x1) + parabolic_weight(data, x2) + parabolic_weight(data, x3)) / 3.0;
    return a * f;
  };

  auto face_integral = [&](const std::vector<svmp::index_t>& verts) {
    if (verts.size() < 3u) {
      return 0.0;
    }

    // Treat any face as a polygon and fan-triangulate it.
    // This is robust to meshes that store boundary faces as CellFamily::Polygon.
    std::vector<Vec3d> pts;
    pts.reserve(verts.size());
    for (const auto v : verts) {
      if (v == svmp::INVALID_INDEX) {
        continue;
      }
      pts.push_back(to_vec3(mesh.get_vertex_coords(v)));
    }
    if (pts.size() < 3u) {
      return 0.0;
    }

    double val = 0.0;
    const Vec3d p0 = pts[0];
    for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
      val += tri_integral(p0, pts[i], pts[i + 1]);
    }
    return val;
  };

  double sum = 0.0;
  const auto payloads = gather_unique_marker_face_payloads<1>(
      mesh, boundary_marker,
      [&](svmp::index_t f, std::array<double, 1>& values) {
        values[0] = face_integral(mesh.face_vertices(f));
      });
  for (const auto& payload : payloads) {
    sum += payload.values[0];
  }
  return sum;
}

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

void apply_fluid_bcs(const svmp::Physics::EquationModuleInput& input,
                     const svmp::Physics::DomainInput& domain,
                     svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Navier-Stokes BC translation received null mesh.");
  }

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes BC translation: " +
        std::to_string(dim) +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  svmp::FE::Real backflow_beta = 0.0;
  if (const auto* p = find_param(domain.params, "Backflow_stabilization_coefficient")) {
    if (!trim_copy(p->value).empty()) {
      backflow_beta = static_cast<svmp::FE::Real>(parse_double(p->value, "Backflow_stabilization_coefficient"));
    }
  }

  for (const auto& bc : input.boundary_conditions) {
    if (bc.boundary_marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto* bc_type_raw = find_param(bc.params, "Type");
    const std::string bc_type = bc_type_raw ? trim_copy(bc_type_raw->value) : std::string{};
    const std::string bc_type_lc = lower_copy(bc_type);

    const auto* time_dep = find_param(bc.params, "Time_dependence");
    const std::string time_value_raw =
        (time_dep && time_dep->defined) ? trim_copy(time_dep->value) : std::string("Steady");
    const std::string time_value_lc = lower_copy(time_value_raw);

    const bool is_steady = time_value_lc.empty() || time_value_lc == "steady";
    const bool is_general = time_value_lc == "general";
    const bool is_unsteady = time_value_lc == "unsteady";
    const bool is_resistance = time_value_lc == "resistance";
    const bool is_rcr = (time_value_lc == "rcr" || time_value_lc == "windkessel");
    const bool is_rcrcr =
        (time_value_lc == "rcrcr" || time_value_lc == "windkessel2c" || time_value_lc == "windkessel_2c");

    const bool has_temp_spat = has_nonempty_defined(bc.params, "Temporal_and_spatial_values_file_path");
    const bool has_other_files = has_nonempty_defined(bc.params, "Temporal_values_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_values_file_path") || has_nonempty_defined(bc.params, "Bct_file_path") ||
        has_nonempty_defined(bc.params, "Traction_values_file_path") ||
        has_nonempty_defined(bc.params, "Fourier_coefficients_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_profile_file_path");

    if (is_steady && (has_temp_spat || has_other_files)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Spatial/temporal boundary-condition files are not supported for the new solver "
          "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    const auto* value_param = find_param(bc.params, "Value");
    const svmp::FE::Real magnitude =
        static_cast<svmp::FE::Real>(value_param ? parse_double(value_param->value, "Add_BC/Value") : 0.0);

    std::vector<int> effective_dir{};
    if (const auto* dir_param = find_param(bc.params, "Effective_direction")) {
      const auto s = trim_copy(dir_param->value);
      if (!s.empty()) {
        effective_dir = parse_int_list(s);
      }
    }

    if (bc_type_lc == "dir" || bc_type_lc == "dirichlet") {
      if (!is_steady && !is_general && !is_unsteady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Only Steady, General, and Unsteady boundary conditions are supported for the new solver "
            "Navier-Stokes module Dirichlet BCs (got Time_dependence='" +
            time_value_raw + "'). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }

      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        // Legacy inputs sometimes set this flag for clarity; the new solver applies
        // velocity Dirichlet along the boundary normal by default when Effective_direction is unset.
        // Accept the flag for Dirichlet to improve legacy compatibility.
      }

      const auto* weak_param = find_param(bc.params, "Weakly_applied");
      const bool weak = weak_param && weak_param->defined && parse_bool_relaxed(weak_param->value);

      if (is_general) {
        if (!has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Navier-Stokes Dirichlet BC currently supports only "
              "<Temporal_and_spatial_values_file_path>. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }
        if (weak) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Dirichlet BCs from temporal/spatial files are only supported as strong "
              "Dirichlet (Weakly_applied=false). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto* file_param = find_param(bc.params, "Temporal_and_spatial_values_file_path");
        const std::string file_path =
            (file_param && file_param->defined) ? trim_copy(file_param->value) : std::string{};
        if (file_path.empty()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Dirichlet BC is missing Temporal_and_spatial_values_file_path.");
        }

        auto data = read_temporal_and_spatial_values_file(*input.mesh, bc.boundary_marker, file_path);

        IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir{};
        dir.boundary_marker = bc.boundary_marker;

        for (int d = 0; d < dim; ++d) {
          if (d < data->dof) {
            const int comp = d;
            dir.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
                [data, comp](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real t) -> svmp::FE::Real {
                  const std::array<svmp::FE::Real, 3> p{x, y, z};
                  const auto node = data->findNodeIndex(p);
                  return data->interpolate(node, t, comp);
                });
          } else {
            dir.value[static_cast<std::size_t>(d)] = IncompressibleNavierStokesVMSOptions::ScalarValue{0.0};
          }
        }

        options.velocity_dirichlet.push_back(std::move(dir));
        continue;
      }

      if (is_unsteady) {
        // Unsteady Dirichlet BC: time-varying magnitude from a .flow file + spatial profile (Flat/Parabolic).
        const auto* file_param = find_param(bc.params, "Temporal_values_file_path");
        const std::string flow_file =
            (file_param && file_param->defined) ? trim_copy(file_param->value) : std::string{};
        if (flow_file.empty()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Unsteady Dirichlet BC '" + bc.name +
              "' is missing <Temporal_values_file_path>.");
        }

        auto temporal = svmp::Physics::readTemporalValuesFile(
            flow_file, /*num_components=*/1, svmp::Physics::TemporalEndBehavior::Periodic);

        const auto* impose_flux_param_u = find_param(bc.params, "Impose_flux");
        const bool impose_flux_u = impose_flux_param_u ? parse_bool_relaxed(impose_flux_param_u->value) : false;

        const auto* profile_param_u = find_param(bc.params, "Profile");
        const std::string profile_raw_u =
            (profile_param_u && profile_param_u->defined) ? trim_copy(profile_param_u->value) : std::string("Flat");
        const std::string profile_lc_u = lower_copy(profile_raw_u);

        InletProfileType profile_u = InletProfileType::Flat;
        if (profile_lc_u == "flat") {
          profile_u = InletProfileType::Flat;
        } else if (profile_lc_u == "parabolic") {
          profile_u = InletProfileType::Parabolic;
        } else {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Unknown Dirichlet BC Profile='" + profile_raw_u +
              "' for Unsteady BC '" + bc.name + "'. Supported profiles: Flat, Parabolic.");
        }

        int active_count_u = 0;
        std::array<int, 3> active_u{0, 0, 0};
        for (int d = 0; d < dim; ++d) {
          const int flag = (static_cast<std::size_t>(d) < effective_dir.size() && effective_dir[static_cast<std::size_t>(d)] != 0) ? 1 : 0;
          active_u[static_cast<std::size_t>(d)] = flag;
          active_count_u += flag;
        }
        const bool use_normal_u = (active_count_u == 0 || active_count_u == dim);

        auto ctx_u = std::make_shared<InletProfileContext>();
        ctx_u->dim = dim;
        ctx_u->profile = profile_u;
        ctx_u->use_normal_direction = use_normal_u;
        ctx_u->active_components = active_u;

        const auto g_u = global_marker_geometry(*input.mesh, bc.boundary_marker);
        if (!(g_u.area > 0.0)) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
              " has zero area; cannot apply Unsteady Dirichlet BC '" + bc.name + "'.");
        }

        if (use_normal_u) {
          const Vec3d n = normalized(g_u.normal_sum);
          if (!(norm2(n) > 0.0)) {
            throw std::runtime_error(
                "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
                " has a degenerate normal; cannot apply Unsteady Dirichlet BC '" + bc.name + "'.");
          }
          ctx_u->normal = n;
        }

        if (profile_u == InletProfileType::Parabolic) {
          ctx_u->parabolic = build_parabolic_profile_data(*input.mesh, bc.boundary_marker);
        }

      double normalization_u = 1.0;
      if (impose_flux_u) {
        if (profile_u == InletProfileType::Flat) {
          normalization_u = g_u.area;
        } else {
            normalization_u = integrate_parabolic_weight_over_marker(*input.mesh, bc.boundary_marker, *ctx_u->parabolic);
          }
          if (!(normalization_u > 0.0)) {
            throw std::runtime_error(
                "[svMultiPhysics::Physics] Failed to compute positive normalization for <Impose_flux> on Unsteady "
                "Dirichlet BC '" + bc.name + "'.");
        }
      }

        // Set scale=1/normalization; the time-dependent flow rate is multiplied in the callback.
        ctx_u->scale = 1.0 / normalization_u;

        if (navierStokesTraceEnabled()) {
          const auto local_g = local_marker_geometry(*input.mesh, bc.boundary_marker);
          const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
          std::ostringstream oss;
          oss << "NavierStokes BC setup: unsteady Dirichlet marker=" << bc.boundary_marker
              << " name='" << bc.name << "'"
              << " profile=" << ((profile_u == InletProfileType::Parabolic) ? "Parabolic" : "Flat")
              << " impose_flux=" << (impose_flux_u ? 1 : 0)
              << " local_faces=" << local_faces.size()
              << " local_area=" << local_g.area
              << " global_area=" << g_u.area
              << " normalization=" << normalization_u
              << " scale=" << ctx_u->scale
              << " center=(" << ((g_u.area > 0.0) ? g_u.center_sum.x / g_u.area : 0.0)
              << "," << ((g_u.area > 0.0) ? g_u.center_sum.y / g_u.area : 0.0)
              << "," << ((g_u.area > 0.0) ? g_u.center_sum.z / g_u.area : 0.0) << ")"
              << " normal=(" << ctx_u->normal.x << "," << ctx_u->normal.y << "," << ctx_u->normal.z << ")"
              << " perim_points=" << (ctx_u->parabolic.has_value() ? ctx_u->parabolic->perimeter_unit_dirs.size() : 0u);
          navierStokesTraceLog(oss.str());
        }

        IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir_u{};
        dir_u.boundary_marker = bc.boundary_marker;

        for (int d = 0; d < dim; ++d) {
          const int comp = d;
          dir_u.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
              [ctx_u, comp, temporal](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real t) -> svmp::FE::Real {
                const double flow_rate = temporal->interpolate(static_cast<double>(t));
                const Vec3d p{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
                // componentValue = scale * weight(x) * direction_factor
                // With scale = 1/normalization, this gives flow_rate / normalization * weight * direction
                return static_cast<svmp::FE::Real>(flow_rate * ctx_u->componentValue(comp, p));
              });
        }

        if (weak) {
          options.velocity_dirichlet_weak.push_back(std::move(dir_u));
        } else {
          options.velocity_dirichlet.push_back(std::move(dir_u));
        }
        continue;
      }

      IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir{};
      dir.boundary_marker = bc.boundary_marker;

      // Legacy-style Dirichlet velocity BCs:
      // - default direction: boundary outward normal
      // - profile: Flat or Parabolic
      // - Impose_flux: if true, normalize profile so ∫ profile ds = 1 and interpret Value as flow rate
      const auto* impose_flux_param = find_param(bc.params, "Impose_flux");
      const bool impose_flux = impose_flux_param ? parse_bool_relaxed(impose_flux_param->value) : false;

      const auto* profile_param = find_param(bc.params, "Profile");
      const std::string profile_raw =
          (profile_param && profile_param->defined) ? trim_copy(profile_param->value) : std::string("Flat");
      const std::string profile_lc = lower_copy(profile_raw);

      InletProfileType profile = InletProfileType::Flat;
      if (profile_lc == "flat") {
        profile = InletProfileType::Flat;
      } else if (profile_lc == "parabolic") {
        profile = InletProfileType::Parabolic;
      } else if (profile_lc == "user_defined" || profile_lc == "user-defined" || profile_lc == "userdefined") {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Dirichlet BC Profile='User_defined' is not supported for the new solver "
            "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      } else {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Unknown Dirichlet BC Profile='" + profile_raw +
            "' for the new solver Navier-Stokes module. Supported profiles: Flat, Parabolic.");
      }

      // Common case: no-slip wall (Value=0) and other zero Dirichlet constraints.
      // Avoid expensive global geometry computations (MPI allgathers) when the imposed value is zero.
      if (magnitude == static_cast<svmp::FE::Real>(0.0)) {
        if (weak) {
          options.velocity_dirichlet_weak.push_back(std::move(dir));
        } else {
          options.velocity_dirichlet.push_back(std::move(dir));
        }
        continue;
      }

      int active_count = 0;
      std::array<int, 3> active{0, 0, 0};
      for (int d = 0; d < dim; ++d) {
        const int flag = (static_cast<std::size_t>(d) < effective_dir.size() && effective_dir[static_cast<std::size_t>(d)] != 0) ? 1 : 0;
        active[static_cast<std::size_t>(d)] = flag;
        active_count += flag;
      }
      const bool use_normal_direction = (active_count == 0 || active_count == dim);

      auto ctx = std::make_shared<InletProfileContext>();
      ctx->dim = dim;
      ctx->profile = profile;
      ctx->use_normal_direction = use_normal_direction;
      ctx->active_components = active;

      // Compute normal/area/center and profile normalization as needed.
      const auto g = global_marker_geometry(*input.mesh, bc.boundary_marker);
      if (!(g.area > 0.0)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
            " has zero area; cannot apply Dirichlet BC '" + bc.name + "'.");
      }

      if (use_normal_direction) {
        const Vec3d n = normalized(g.normal_sum);
        if (!(norm2(n) > 0.0)) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
              " has a degenerate normal; cannot apply Dirichlet BC '" + bc.name + "'.");
        }
        ctx->normal = n;
      }

      if (profile == InletProfileType::Parabolic) {
        ctx->parabolic = build_parabolic_profile_data(*input.mesh, bc.boundary_marker);
      }

      double normalization = 1.0;
      if (impose_flux) {
        if (profile == InletProfileType::Flat) {
          normalization = g.area;
        } else {
          normalization = integrate_parabolic_weight_over_marker(*input.mesh, bc.boundary_marker, *ctx->parabolic);
        }
        if (!(normalization > 0.0)) {
          const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
          const std::size_t perim_points =
              (ctx->parabolic.has_value()) ? ctx->parabolic->perimeter_unit_dirs.size() : 0u;
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Failed to compute a positive normalization for <Impose_flux>true</Impose_flux> "
              "on Dirichlet BC '" + bc.name + "' (marker=" + std::to_string(bc.boundary_marker) +
              ", normalization=" + std::to_string(normalization) + ", area=" + std::to_string(g.area) +
              ", perim_points=" + std::to_string(perim_points) + ", local_faces=" + std::to_string(local_faces.size()) +
              ").");
        }
      }

      ctx->scale = static_cast<double>(magnitude) / normalization;

      if (navierStokesTraceEnabled()) {
        const auto local_g = local_marker_geometry(*input.mesh, bc.boundary_marker);
        const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
        std::ostringstream oss;
        oss << "NavierStokes BC setup: steady Dirichlet marker=" << bc.boundary_marker
            << " name='" << bc.name << "'"
            << " profile=" << ((profile == InletProfileType::Parabolic) ? "Parabolic" : "Flat")
            << " impose_flux=" << (impose_flux ? 1 : 0)
            << " magnitude=" << magnitude
            << " local_faces=" << local_faces.size()
            << " local_area=" << local_g.area
            << " global_area=" << g.area
            << " normalization=" << normalization
            << " scale=" << ctx->scale
            << " center=(" << ((g.area > 0.0) ? g.center_sum.x / g.area : 0.0)
            << "," << ((g.area > 0.0) ? g.center_sum.y / g.area : 0.0)
            << "," << ((g.area > 0.0) ? g.center_sum.z / g.area : 0.0) << ")"
            << " normal=(" << ctx->normal.x << "," << ctx->normal.y << "," << ctx->normal.z << ")"
            << " perim_points=" << (ctx->parabolic.has_value() ? ctx->parabolic->perimeter_unit_dirs.size() : 0u);
        navierStokesTraceLog(oss.str());
      }

      for (int d = 0; d < dim; ++d) {
        const int comp = d;
        dir.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
            [ctx, comp](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real /*t*/) -> svmp::FE::Real {
              const Vec3d p{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
              return static_cast<svmp::FE::Real>(ctx->componentValue(comp, p));
            });
      }

      if (weak) {
        options.velocity_dirichlet_weak.push_back(std::move(dir));
      } else {
        options.velocity_dirichlet.push_back(std::move(dir));
      }
      continue;
    }

    if (bc_type_lc == "neu" || bc_type_lc == "neumann") {
      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] <Apply_along_normal_direction>true</Apply_along_normal_direction> is not "
            "supported for the new solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to "
            "use the legacy solver.");
      }

      if (is_steady) {
        IncompressibleNavierStokesVMSOptions::PressureOutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.pressure = IncompressibleNavierStokesVMSOptions::ScalarValue{magnitude};
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.pressure_outflow.push_back(std::move(out));
        continue;
      }

      if (is_resistance) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Resistance outflow BCs do not support spatial/temporal files for the new "
              "solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = 0.0;
        out.C = 0.0;
        out.Rd = static_cast<svmp::FE::Real>(magnitude);
        out.Pd = 0.0;
        out.X0 = 0.0;
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcr.push_back(std::move(out));
        continue;
      }

      if (is_rcr) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCR outflow BCs do not support spatial/temporal files for the new solver "
              "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto Rp = get_defined_double(bc.params, "RCR.Proximal_resistance");
        const auto C = get_defined_double(bc.params, "RCR.Capacitance");
        const auto Rd = get_defined_double(bc.params, "RCR.Distal_resistance");
        const auto Pd = get_defined_double(bc.params, "RCR.Distal_pressure");
        const auto X0 = get_defined_double(bc.params, "RCR.Initial_pressure");

        if (!Rp.has_value() || !C.has_value() || !Rd.has_value()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCR outflow BC '" + bc.name + "' is missing required <RCR_values> entries "
              "(Proximal_resistance, Capacitance, Distal_resistance).");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = static_cast<svmp::FE::Real>(*Rp);
        out.C = static_cast<svmp::FE::Real>(*C);
        out.Rd = static_cast<svmp::FE::Real>(*Rd);
        out.Pd = static_cast<svmp::FE::Real>(Pd.value_or(0.0));
        out.X0 = static_cast<svmp::FE::Real>(X0.value_or(0.0));
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcr.push_back(std::move(out));
        continue;
      }

      if (is_rcrcr) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCRCR outflow BCs do not support spatial/temporal files for the new solver "
              "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto Rp = get_defined_double(bc.params, "RCRCR.Proximal_resistance");
        const auto C1 = get_defined_double(bc.params, "RCRCR.Proximal_capacitance");
        const auto Rm = get_defined_double(bc.params, "RCRCR.Intermediate_resistance");
        const auto C2 = get_defined_double(bc.params, "RCRCR.Distal_capacitance");
        const auto Rd = get_defined_double(bc.params, "RCRCR.Distal_resistance");
        const auto Pd = get_defined_double(bc.params, "RCRCR.Distal_pressure");
        const auto P10 = get_defined_double(bc.params, "RCRCR.Initial_pressure_1");
        const auto P20 = get_defined_double(bc.params, "RCRCR.Initial_pressure_2");

        if (!Rp.has_value() || !C1.has_value() || !Rm.has_value() || !C2.has_value() || !Rd.has_value()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCRCR outflow BC '" + bc.name +
              "' is missing required <RCRCR_values> entries "
              "(Proximal_resistance, Proximal_capacitance, Intermediate_resistance, Distal_capacitance, "
              "Distal_resistance).");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = static_cast<svmp::FE::Real>(*Rp);
        out.C1 = static_cast<svmp::FE::Real>(*C1);
        out.Rm = static_cast<svmp::FE::Real>(*Rm);
        out.C2 = static_cast<svmp::FE::Real>(*C2);
        out.Rd = static_cast<svmp::FE::Real>(*Rd);
        out.Pd = static_cast<svmp::FE::Real>(Pd.value_or(0.0));
        out.P10 = static_cast<svmp::FE::Real>(P10.value_or(Pd.value_or(0.0)));
        out.P20 = static_cast<svmp::FE::Real>(P20.value_or(Pd.value_or(0.0)));
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcrcr.push_back(std::move(out));
        continue;
      }

      throw std::runtime_error(
          "[svMultiPhysics::Physics] Neumann BC Time_dependence='" + time_value_raw +
          "' is not supported for the new solver Navier-Stokes module. Supported: Steady, Resistance, RCR, RCRCR. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (bc_type_lc == "trac" || bc_type_lc == "traction") {
      if (!is_steady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] General/Unsteady traction BCs are not supported for the new solver Navier-Stokes "
            "module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
      IncompressibleNavierStokesVMSOptions::TractionNeumannBC trac{};
      trac.boundary_marker = bc.boundary_marker;
      fill_vector(trac.traction, dim, effective_dir, magnitude);
      options.traction_neumann.push_back(std::move(trac));
      continue;
    }

    if (bc_type_lc == "rbn" || bc_type_lc == "robin") {
      if (!is_steady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] General/Unsteady Robin BCs are not supported for the new solver Navier-Stokes "
            "module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] <Apply_along_normal_direction>true</Apply_along_normal_direction> is not "
            "supported for the new solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to "
            "use the legacy solver.");
      }
      const auto* stiff_param = find_param(bc.params, "Stiffness");
      const svmp::FE::Real stiff =
          static_cast<svmp::FE::Real>(stiff_param ? parse_double(stiff_param->value, "Add_BC/Stiffness") : 0.0);

      IncompressibleNavierStokesVMSOptions::TractionRobinBC robin{};
      robin.boundary_marker = bc.boundary_marker;
      robin.alpha = IncompressibleNavierStokesVMSOptions::ScalarValue{stiff};
      fill_vector(robin.rhs, dim, effective_dir, magnitude);
      options.traction_robin.push_back(std::move(robin));
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary condition type '" + bc_type +
        "' is not supported for the new solver Navier-Stokes module. Supported types: Dir, Dirichlet, Neu, Neumann, "
        "Trac, Traction, Robin, Rbn. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_navier_stokes_from_input(const svmp::Physics::EquationModuleInput& input,
                                svmp::FE::systems::FESystem& system)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Navier-Stokes module factory received null mesh.");
  }

  const auto& domain = select_single_domain(input, "Navier-Stokes");

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes spaces: " + std::to_string(dim) +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int vel_order = resolve_element_order(input, infer_polynomial_order(*input.mesh));

  bool taylor_hood = false;
  if (const auto* p = find_param(input.equation_params, "Use_taylor_hood_type_basis"); p && p->defined) {
    taylor_hood = parse_bool_relaxed(p->value);
  }
  const int p_order = taylor_hood ? std::max(1, vel_order - 1) : vel_order;

  auto velocity_space =
      svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, element_type, vel_order, dim);
  auto pressure_space = svmp::FE::spaces::SpaceFactory::create_h1(element_type, p_order);

  svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions options{};
  options.velocity_field_name = "Velocity";
  options.pressure_field_name = "Pressure";
  options.enable_convection = (input.equation_type != "stokes");
  options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);

  apply_fluid_moving_domain_options(input, domain, options);
  apply_fluid_properties(domain, options);
  apply_fluid_bcs(input, domain, options);

  auto module = std::make_unique<svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule>(
      std::move(velocity_space), std::move(pressure_space), std::move(options));
  module->registerOn(system);
  return module;
}

} // namespace

SVMP_REGISTER_EQUATION("fluid", &create_navier_stokes_from_input);
SVMP_REGISTER_EQUATION("stokes", &create_navier_stokes_from_input);

namespace svmp::Physics::formulations::navier_stokes {

void forceLink_NavierStokesRegister() {}

} // namespace svmp::Physics::formulations::navier_stokes

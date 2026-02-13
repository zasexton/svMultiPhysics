#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"

#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"

#include <algorithm>
#include <array>
#include <cctype>
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

  for (int b = 0; b < num_nodes; ++b) {
    long long node_id_1based = 0;
    in >> node_id_1based;
    const long long node_gid0_ll = node_id_1based - 1;
    if (node_gid0_ll < 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "': invalid negative node id: " + std::to_string(node_id_1based) + ".");
    }

    const auto node_gid = static_cast<svmp::gid_t>(node_gid0_ll);
    const auto node_idx = mesh.global_to_local_vertex(node_gid);

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

    const bool keep = (node_idx != svmp::INVALID_INDEX) && (boundary_gids.empty() || boundary_gids.count(node_gid) != 0u);
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

MarkerGeometry global_marker_geometry(const svmp::MeshBase& mesh, int boundary_marker)
{
  // Use a simple sum + reduction that does not depend on face GIDs.
  //
  // Some meshes populate face_gids() with local indices (not globally unique across ranks),
  // so de-duplicating by face_gid can silently drop unrelated faces and corrupt marker
  // geometry. A plain reduction is robust; it may double-count ghost faces, but the
  // centroid/normal are typically unaffected when ghosts mirror owned entities.
  auto out = local_marker_geometry(mesh, boundary_marker);
#if FE_HAS_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    double local_area = out.area;
    double global_area = 0.0;
    MPI_Allreduce(&local_area, &global_area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    out.area = global_area;

    const std::array<double, 3> local_center = {out.center_sum.x, out.center_sum.y, out.center_sum.z};
    const std::array<double, 3> local_normal = {out.normal_sum.x, out.normal_sum.y, out.normal_sum.z};
    std::array<double, 3> global_center{0.0, 0.0, 0.0};
    std::array<double, 3> global_normal{0.0, 0.0, 0.0};

    MPI_Allreduce(local_center.data(), global_center.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_normal.data(), global_normal.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    out.center_sum = Vec3d{global_center[0], global_center[1], global_center[2]};
    out.normal_sum = Vec3d{global_normal[0], global_normal[1], global_normal[2]};
  }
#endif
  return out;
}

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

std::vector<std::pair<svmp::gid_t, Vec3d>> gather_perimeter_vertex_coords(const svmp::MeshBase& mesh,
                                                                          int boundary_marker)
{
  const auto& vgids = mesh.vertex_gids();
  if (vgids.size() != mesh.n_vertices()) {
    return {};
  }

  // Determine the perimeter of the marker surface as the set of boundary edges of the
  // marker patch itself (edges that appear only once among marker faces).
  //
  // This is robust in MPI even when boundary labels are not synchronized to ghost faces.
  const auto marker_faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));

  std::vector<svmp::gid_t> local_edge_gids;
  local_edge_gids.reserve(marker_faces.size() * 6u);

  for (const auto f : marker_faces) {
    const auto verts = mesh.face_vertices(f);
    if (verts.size() < 2u) {
      continue;
    }

    std::vector<svmp::gid_t> face_vgids;
    face_vgids.reserve(verts.size());
    for (const auto v : verts) {
      if (v == svmp::INVALID_INDEX) {
        continue;
      }
      const auto idx = static_cast<std::size_t>(v);
      if (idx >= vgids.size()) {
        continue;
      }
      face_vgids.push_back(vgids[idx]);
    }
    if (face_vgids.size() < 2u) {
      continue;
    }

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

  std::vector<svmp::gid_t> all_edge_gids = local_edge_gids;
  bool mpi_active = false;

#if FE_HAS_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  mpi_active = (mpi_initialized != 0);
  if (mpi_active) {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    const int local_n = static_cast<int>(local_edge_gids.size());
    std::vector<int> counts(comm_size, 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(comm_size, 0);
    int total = 0;
    for (int i = 0; i < comm_size; ++i) {
      displs[i] = total;
      total += counts[i];
    }

    all_edge_gids.assign(static_cast<std::size_t>(total), svmp::gid_t{0});
    MPI_Allgatherv(local_edge_gids.data(),
                   local_n,
                   MPI_INT64_T,
                   all_edge_gids.data(),
                   counts.data(),
                   displs.data(),
                   MPI_INT64_T,
                   MPI_COMM_WORLD);
  }
#endif

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
  if (mpi_active) {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    const int local_n = static_cast<int>(local_gids.size());
    std::vector<int> counts(comm_size, 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(comm_size, 0);
    int total = 0;
    for (int i = 0; i < comm_size; ++i) {
      displs[i] = total;
      total += counts[i];
    }

    all_gids.assign(static_cast<std::size_t>(total), svmp::gid_t{0});
    MPI_Allgatherv(local_gids.data(),
                   local_n,
                   MPI_INT64_T,
                   all_gids.data(),
                   counts.data(),
                   displs.data(),
                   MPI_INT64_T,
                   MPI_COMM_WORLD);

    std::vector<int> counts3(comm_size, 0);
    std::vector<int> displs3(comm_size, 0);
    int total3 = 0;
    for (int i = 0; i < comm_size; ++i) {
      counts3[i] = counts[i] * 3;
      displs3[i] = total3;
      total3 += counts3[i];
    }
    all_xyz.assign(static_cast<std::size_t>(total3), 0.0);
    MPI_Allgatherv(local_xyz.data(),
                   local_n * 3,
                   MPI_DOUBLE,
                   all_xyz.data(),
                   counts3.data(),
                   displs3.data(),
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
  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
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

  double local_sum = 0.0;
  for (const auto f : faces) {
    local_sum += face_integral(mesh.face_vertices(f));
  }

  double sum = local_sum;
#if FE_HAS_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
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
    const bool is_resistance = time_value_lc == "resistance";
    const bool is_rcr = (time_value_lc == "rcr" || time_value_lc == "windkessel");

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
      if (!is_steady && !is_general) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Only Steady and General boundary conditions are supported for the new solver "
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

      throw std::runtime_error(
          "[svMultiPhysics::Physics] Neumann BC Time_dependence='" + time_value_raw +
          "' is not supported for the new solver Navier-Stokes module. Supported: Steady, Resistance, RCR. "
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
  const int vel_order = infer_polynomial_order(*input.mesh);

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

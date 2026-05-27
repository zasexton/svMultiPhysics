#include "Application/Translators/LevelSetEquationTranslator.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/PhysicsModule.h"

#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/JIT/LLVMJITBuildInfo.h"
#include "FE/LevelSet/LevelSetTransport.h"
#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace ls = svmp::FE::level_set;

struct LevelSetTemporalSpatialInflowValues {
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
    std::size_t operator()(const Key& key) const noexcept
    {
      std::size_t h = 1469598103934665603ull;
      auto mix = [&](std::int64_t value) {
        const auto x = std::hash<std::int64_t>{}(value);
        h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      };
      mix(key.x);
      mix(key.y);
      mix(key.z);
      return h;
    }
  };

  int dim{0};
  int num_time_points{0};
  std::string file_path{};
  std::vector<double> t{};
  double period{0.0};
  std::vector<std::array<svmp::FE::Real, 3>> coords{};
  std::vector<svmp::FE::Real> values{};
  std::unordered_map<Key, std::size_t, KeyHash> node_index_by_key{};
  int line_axis{-1};
  std::vector<svmp::FE::Real> line_coords{};
  std::vector<std::size_t> line_nodes{};

  [[nodiscard]] static Key quantize(const std::array<svmp::FE::Real, 3>& p, int dim_in) noexcept
  {
    constexpr double scale = 1.0e12;
    auto q = [](svmp::FE::Real value) {
      return static_cast<std::int64_t>(
          std::llround(static_cast<double>(value) * scale));
    };
    return Key{
        .x = q(p[0]),
        .y = dim_in >= 2 ? q(p[1]) : 0,
        .z = dim_in >= 3 ? q(p[2]) : 0,
    };
  }

  [[nodiscard]] svmp::FE::Real sample(std::size_t node, int time_index) const
  {
    const auto idx =
        node * static_cast<std::size_t>(num_time_points) + static_cast<std::size_t>(time_index);
    if (idx >= values.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Internal error: level-set temporal/spatial inflow index out of range.");
    }
    return values[idx];
  }

  void buildInterpolationMetadata()
  {
    line_axis = -1;
    line_coords.clear();
    line_nodes.clear();
    if (coords.size() < 2) {
      return;
    }

    std::array<double, 3> min_coord{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> max_coord{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    for (const auto& p : coords) {
      for (int d = 0; d < dim; ++d) {
        const auto value = static_cast<double>(p[static_cast<std::size_t>(d)]);
        min_coord[static_cast<std::size_t>(d)] =
            std::min(min_coord[static_cast<std::size_t>(d)], value);
        max_coord[static_cast<std::size_t>(d)] =
            std::max(max_coord[static_cast<std::size_t>(d)], value);
      }
    }

    double best_range = 0.0;
    for (int d = 0; d < dim; ++d) {
      const auto range = max_coord[static_cast<std::size_t>(d)] -
                         min_coord[static_cast<std::size_t>(d)];
      if (range > best_range) {
        best_range = range;
        line_axis = d;
      }
    }
    if (line_axis < 0 || best_range <= 1.0e-14) {
      return;
    }

    line_nodes.resize(coords.size());
    for (std::size_t i = 0; i < line_nodes.size(); ++i) {
      line_nodes[i] = i;
    }
    std::sort(line_nodes.begin(), line_nodes.end(), [this](std::size_t a, std::size_t b) {
      return coords[a][static_cast<std::size_t>(line_axis)] <
             coords[b][static_cast<std::size_t>(line_axis)];
    });
    line_coords.reserve(line_nodes.size());
    for (const auto node : line_nodes) {
      line_coords.push_back(coords[node][static_cast<std::size_t>(line_axis)]);
    }
  }

  [[nodiscard]] double wrapTime(double time) const noexcept
  {
    if (!(period > 0.0) || !std::isfinite(period) || num_time_points < 2) {
      return time;
    }
    auto wrapped = std::fmod(time, period);
    if (wrapped < 0.0) {
      wrapped += period;
    }
    return wrapped;
  }

  [[nodiscard]] svmp::FE::Real interpolateTime(std::size_t node,
                                               svmp::FE::Real time) const
  {
    if (num_time_points <= 0) {
      return svmp::FE::Real{0.0};
    }
    if (num_time_points == 1) {
      return sample(node, 0);
    }

    const auto tt = wrapTime(static_cast<double>(time));
    int i0 = 0;
    for (int i = 0; i < num_time_points - 1; ++i) {
      if (t[static_cast<std::size_t>(i + 1)] >= tt) {
        i0 = i;
        break;
      }
    }
    const auto t0 = t[static_cast<std::size_t>(i0)];
    const auto t1 = t[static_cast<std::size_t>(i0 + 1)];
    const auto alpha = (t1 > t0) ? ((tt - t0) / (t1 - t0)) : 0.0;
    const auto v0 = static_cast<double>(sample(node, i0));
    const auto v1 = static_cast<double>(sample(node, i0 + 1));
    return static_cast<svmp::FE::Real>((1.0 - alpha) * v0 + alpha * v1);
  }

  [[nodiscard]] svmp::FE::Real interpolateSpatial(const std::array<svmp::FE::Real, 3>& p,
                                                  svmp::FE::Real time) const
  {
    if (coords.empty()) {
      return svmp::FE::Real{0.0};
    }
    if (const auto it = node_index_by_key.find(quantize(p, dim)); it != node_index_by_key.end()) {
      return interpolateTime(it->second, time);
    }
    if (line_axis >= 0 && !line_coords.empty()) {
      const auto x = p[static_cast<std::size_t>(line_axis)];
      if (x <= line_coords.front()) {
        return interpolateTime(line_nodes.front(), time);
      }
      if (x >= line_coords.back()) {
        return interpolateTime(line_nodes.back(), time);
      }
      const auto upper = std::upper_bound(line_coords.begin(), line_coords.end(), x);
      const auto hi = static_cast<std::size_t>(std::distance(line_coords.begin(), upper));
      const auto lo = hi - 1u;
      const auto x0 = static_cast<double>(line_coords[lo]);
      const auto x1 = static_cast<double>(line_coords[hi]);
      const auto alpha = (x1 > x0) ? ((static_cast<double>(x) - x0) / (x1 - x0)) : 0.0;
      const auto v0 = static_cast<double>(interpolateTime(line_nodes[lo], time));
      const auto v1 = static_cast<double>(interpolateTime(line_nodes[hi], time));
      return static_cast<svmp::FE::Real>((1.0 - alpha) * v0 + alpha * v1);
    }

    std::size_t best = 0;
    auto best_d2 = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < coords.size(); ++i) {
      const auto dx = static_cast<double>(p[0] - coords[i][0]);
      const auto dy = static_cast<double>(p[1] - coords[i][1]);
      const auto dz = static_cast<double>(p[2] - coords[i][2]);
      const auto d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = i;
      }
    }
    return interpolateTime(best, time);
  }
};

std::shared_ptr<LevelSetTemporalSpatialInflowValues>
read_level_set_temporal_spatial_inflow_file(const svmp::MeshBase& mesh,
                                            const std::string& file_path)
{
  std::ifstream input(file_path);
  if (!input.is_open()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to open level-set temporal/spatial inflow file '" +
        file_path + "'.");
  }

  int ndof = 0;
  int num_times = 0;
  int num_nodes = 0;
  input >> ndof >> num_times >> num_nodes;
  if (ndof < 1 || num_times <= 0 || num_nodes <= 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Invalid header in level-set temporal/spatial inflow file '" +
        file_path + "' (expected: <ndof> <num_ts> <num_nodes>).");
  }

  const int dim = mesh.dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Invalid mesh dimension for level-set temporal/spatial inflow parsing.");
  }

  auto out = std::make_shared<LevelSetTemporalSpatialInflowValues>();
  out->dim = dim;
  out->num_time_points = num_times;
  out->file_path = file_path;
  out->t.resize(static_cast<std::size_t>(num_times));

  for (int i = 0; i < num_times; ++i) {
    double time = 0.0;
    input >> time;
    out->t[static_cast<std::size_t>(i)] = time;
    if (i == 0) {
      if (std::abs(time) > 1.0e-14) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set temporal/spatial inflow file '" +
            file_path + "': first time value must be 0.");
      }
    } else if (!(time > out->t[static_cast<std::size_t>(i - 1)])) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set temporal/spatial inflow file '" +
          file_path + "': time values must be increasing.");
    }
  }
  out->period = out->t.back();

  for (int node = 0; node < num_nodes; ++node) {
    long long node_id_1based = 0;
    input >> node_id_1based;
    if (node_id_1based <= 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set temporal/spatial inflow file '" +
          file_path + "': node ids must be one-based positive ids.");
    }
    const auto local = mesh.global_to_local_vertex(static_cast<svmp::gid_t>(node_id_1based - 1));
    const bool keep = local != svmp::INVALID_INDEX;
    std::array<svmp::FE::Real, 3> p{0.0, 0.0, 0.0};
    if (keep) {
      const auto& X = mesh.X_ref();
      const auto base = static_cast<std::size_t>(local) * static_cast<std::size_t>(dim);
      p[0] = static_cast<svmp::FE::Real>(X.at(base + 0u));
      if (dim >= 2) {
        p[1] = static_cast<svmp::FE::Real>(X.at(base + 1u));
      }
      if (dim >= 3) {
        p[2] = static_cast<svmp::FE::Real>(X.at(base + 2u));
      }
      const auto stored = out->coords.size();
      out->coords.push_back(p);
      out->node_index_by_key.emplace(
          LevelSetTemporalSpatialInflowValues::quantize(p, dim),
          stored);
    }

    for (int time = 0; time < num_times; ++time) {
      for (int comp = 0; comp < ndof; ++comp) {
        double value = 0.0;
        input >> value;
        if (keep && comp == 0) {
          out->values.push_back(static_cast<svmp::FE::Real>(value));
        }
      }
    }
  }

  if (out->coords.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set temporal/spatial inflow file '" +
        file_path + "' did not match any local mesh vertices.");
  }
  out->buildInterpolationMetadata();
  return out;
}

class LevelSetTransportInputAdapter final : public svmp::Physics::PhysicsModule {
public:
  LevelSetTransportInputAdapter(
      std::shared_ptr<const svmp::FE::spaces::FunctionSpace> level_set_space,
      ls::LevelSetTransportOptions options,
      svmp::FE::systems::FormInstallOptions install_options)
      : level_set_space_(std::move(level_set_space)),
        options_(std::move(options)),
        install_options_(std::move(install_options))
  {
  }

  void registerOn(svmp::FE::systems::FESystem& system) const override
  {
    (void)ls::installLevelSetTransport(
        system,
        level_set_space_,
        options_,
        install_options_);
  }

  void applyInitialConditions(const svmp::FE::systems::FESystem& system,
                              svmp::FE::backends::GenericVector& u0) const override
  {
    if (options_.level_set.source != ls::LevelSetFieldSource::PrescribedData) {
      return;
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto phi_id = system.findFieldByName(options_.level_set.field_name);
    if (phi_id == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set initial condition could not find field '" +
          options_.level_set.field_name + "'.");
    }

    const auto& rec = system.fieldRecord(phi_id);
    if (rec.components != 1) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set initial condition field '" +
          rec.name + "' must be scalar.");
    }

    const auto& mesh = system.singleMesh("Level-set initial condition");
    const auto& local_mesh = mesh.local_mesh();
    const auto mesh_field = svmp::MeshFields::get_field_handle(
        local_mesh,
        svmp::EntityKind::Vertex,
        options_.level_set.field_name);
    if (mesh_field.id == 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set initial condition requires vertex field '" +
          options_.level_set.field_name + "' in the input mesh.");
    }

    const auto mesh_components = svmp::MeshFields::field_components(local_mesh, mesh_field);
    if (mesh_components < 1u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set input mesh field '" +
          options_.level_set.field_name + "' has no components.");
    }

    const auto* mesh_values = svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, mesh_field);
    if (mesh_values == nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set input mesh field '" +
          options_.level_set.field_name + "' has no data.");
    }

    const auto mesh_entity_count =
        svmp::MeshFields::field_entity_count(local_mesh, mesh_field);
    if (mesh_entity_count < mesh.n_vertices()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set input mesh field '" +
          options_.level_set.field_name + "' has fewer entries than mesh vertices.");
    }

    const auto& field_dofs = system.fieldDofHandler(phi_id);
    const auto phi_offset = system.fieldDofOffset(phi_id);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    std::vector<svmp::FE::Real> coefficients(n_field_dofs, 0.0);
    std::vector<std::uint8_t> assigned(n_field_dofs, 0u);
    const auto projection =
        system.projectMeshVertexValuesToFieldCoefficients(
            phi_id,
            std::span<const svmp::FE::Real>(
                mesh_values, mesh_entity_count * mesh_components),
            mesh_components,
            std::span<svmp::FE::Real>(coefficients.data(), coefficients.size()),
            std::span<std::uint8_t>(assigned.data(), assigned.size()),
            "Level-set initial condition");
    if (projection.unassigned_dofs != 0u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set initial condition field '" +
          rec.name + "' left " +
          std::to_string(projection.unassigned_dofs) +
          " coefficient(s) without a safe mesh-vertex projection.");
    }

    std::vector<svmp::FE::GlobalIndex> dofs;
    std::vector<svmp::FE::Real> values;
    dofs.reserve(n_field_dofs);
    values.reserve(n_field_dofs);

    for (std::size_t local_dof = 0; local_dof < n_field_dofs; ++local_dof) {
      if (assigned[local_dof] == 0u) {
        continue;
      }
      dofs.push_back(phi_offset +
                     static_cast<svmp::FE::GlobalIndex>(local_dof));
      values.push_back(coefficients[local_dof]);
    }

    if (dofs.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set initial condition found no field DOFs.");
    }

    auto view = u0.createAssemblyView();
    if (!view) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set initial condition could not create a vector view.");
    }
    view->beginAssemblyPhase();
    view->setVectorEntries(dofs, values);
    view->endAssemblyPhase();
    view->finalizeAssembly();
#else
    (void)system;
    (void)u0;
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set prescribed-data initialization requires mesh support.");
#endif
  }

private:
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> level_set_space_{};
  ls::LevelSetTransportOptions options_{};
  svmp::FE::systems::FormInstallOptions install_options_{};
};

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

std::string normalized_token(std::string s)
{
  s = lower_copy(trim_copy(std::move(s)));
  s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch) {
            return ch == '_' || ch == '-' || std::isspace(ch);
          }),
          s.end());
  return s;
}

const svmp::Physics::ParameterValue* find_param(const svmp::Physics::ParameterMap& params,
                                                std::string_view key)
{
  const auto it = params.find(std::string(key));
  return (it == params.end()) ? nullptr : &it->second;
}

std::optional<std::string> get_defined_string(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    const auto* p = find_param(params, key);
    if (!p || !p->defined) {
      continue;
    }
    auto value = trim_copy(p->value);
    if (!value.empty()) {
      return value;
    }
  }
  return std::nullopt;
}

bool parse_bool_relaxed(std::string_view raw)
{
  const auto value = lower_copy(trim_copy(std::string(raw)));
  return value == "true" || value == "1" || value == "yes" || value == "on";
}

std::optional<bool> get_defined_bool(const svmp::Physics::ParameterMap& params,
                                     std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    const auto* p = find_param(params, key);
    if (!p || !p->defined) {
      continue;
    }
    const auto value = trim_copy(p->value);
    if (!value.empty()) {
      return parse_bool_relaxed(value);
    }
  }
  return std::nullopt;
}

double parse_double(std::string_view raw, std::string_view context)
{
  const auto s = trim_copy(std::string(raw));
  try {
    size_t pos = 0;
    const double v = std::stod(s, &pos);
    if (pos != s.size() || !std::isfinite(v)) {
      throw std::runtime_error("");
    }
    return v;
  } catch (...) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to parse numeric value '" + std::string(raw) +
                             "' for " + std::string(context) + ".");
  }
}

int parse_int(std::string_view raw, std::string_view context)
{
  const auto s = trim_copy(std::string(raw));
  try {
    size_t pos = 0;
    const int v = std::stoi(s, &pos);
    if (pos != s.size()) {
      throw std::runtime_error("");
    }
    return v;
  } catch (...) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to parse integer value '" + std::string(raw) +
                             "' for " + std::string(context) + ".");
  }
}

std::optional<svmp::FE::Real> get_defined_real(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  for (const auto key : keys) {
    const auto* p = find_param(params, key);
    if (!p || !p->defined) {
      continue;
    }
    const auto value = trim_copy(p->value);
    if (!value.empty()) {
      return static_cast<svmp::FE::Real>(parse_double(value, context));
    }
  }
  return std::nullopt;
}

std::optional<int> get_defined_int(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  for (const auto key : keys) {
    const auto* p = find_param(params, key);
    if (!p || !p->defined) {
      continue;
    }
    const auto value = trim_copy(p->value);
    if (!value.empty()) {
      return parse_int(value, context);
    }
  }
  return std::nullopt;
}

std::array<svmp::FE::Real, 3> parse_real_vector3(std::string_view raw, std::string_view context)
{
  std::istringstream in{std::string(raw)};
  std::array<svmp::FE::Real, 3> out{0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < out.size(); ++i) {
    double value = 0.0;
    if (!(in >> value) || !std::isfinite(value)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Failed to parse three numeric components for " +
          std::string(context) + ".");
    }
    out[i] = static_cast<svmp::FE::Real>(value);
  }
  std::string extra;
  if (in >> extra) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to parse three numeric components for " +
        std::string(context) + ".");
  }
  return out;
}

std::optional<std::array<svmp::FE::Real, 3>> get_defined_vector3(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  for (const auto key : keys) {
    const auto* p = find_param(params, key);
    if (!p || !p->defined) {
      continue;
    }
    const auto value = trim_copy(p->value);
    if (!value.empty()) {
      return parse_real_vector3(value, context);
    }
  }
  return std::nullopt;
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
    throw std::runtime_error("[svMultiPhysics::Application] Failed to parse positive integer value '" +
                             std::string(raw) + "' for " + std::string(context) + ".");
  }
}

std::optional<int> get_defined_positive_int(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  for (const auto key : keys) {
    const auto* p = find_param(params, key);
    if (!p || !p->defined) {
      continue;
    }
    const auto value = trim_copy(p->value);
    if (!value.empty()) {
      return parse_positive_int(value, context);
    }
  }
  return std::nullopt;
}

svmp::FE::systems::FormInstallOptions level_set_install_options(
    const svmp::Physics::core::PhysicsJITPolicy& policy)
{
  svmp::FE::systems::FormInstallOptions options{};
  options.compiler_options.jit.enable =
      policy.enable && svmp::FE::forms::jit::llvmJITEnabled();
  options.compiler_options.jit.optimization_level = policy.optimization_level;
  options.compiler_options.jit.specialization.enable = policy.specialization;
  options.compiler_options.jit.specialization.specialize_n_qpts = policy.specialize_n_qpts;
  options.compiler_options.jit.specialization.specialize_dofs = policy.specialize_dofs;
  return options;
}

svmp::FE::ElementType infer_base_element_type(const svmp::MeshBase& mesh)
{
  if (mesh.n_cells() == 0) {
    throw std::runtime_error("[svMultiPhysics::Application] Mesh has no cells; cannot infer FE element type.");
  }

  const auto& shapes = mesh.cell_shapes();
  if (shapes.empty()) {
    throw std::runtime_error("[svMultiPhysics::Application] Mesh has no cell shapes; cannot infer FE element type.");
  }

  if (shapes.front().is_mixed_order) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Mixed-order meshes are not supported by the level-set transport module.");
  }

  const auto family = shapes.front().family;
  for (const auto& s : shapes) {
    if (s.family != family) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Mixed cell families are not supported by the level-set transport module.");
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

  throw std::runtime_error("[svMultiPhysics::Application] Unsupported mesh cell family for level-set transport.");
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
          "[svMultiPhysics::Application] Mixed polynomial orders are not supported by the level-set transport module.");
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

svmp::FE::BasisType resolve_basis_type(const svmp::Physics::EquationModuleInput& input)
{
  const auto basis = get_defined_string(
      input.equation_params,
      {"Basis_type",
       "BasisType",
       "Level_set_basis_type",
       "LevelSetBasisType",
       "Element_basis_type",
       "ElementBasisType"});
  if (!basis.has_value()) {
    return svmp::FE::BasisType::Lagrange;
  }

  const auto value = normalized_token(*basis);
  if (value == "lagrange" || value == "nodal") {
    return svmp::FE::BasisType::Lagrange;
  }
  if (value == "serendipity") {
    return svmp::FE::BasisType::Serendipity;
  }
  if (value == "hierarchical" || value == "modal") {
    return svmp::FE::BasisType::Hierarchical;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Unsupported level-set Basis_type '" +
      *basis + "'. Supported values are Lagrange, Serendipity, and Hierarchical.");
}

ls::LevelSetFieldSource parse_level_set_source(std::string_view raw)
{
  const auto value = normalized_token(std::string(raw));
  if (value == "unknown" || value == "solution") {
    return ls::LevelSetFieldSource::Unknown;
  }
  if (value == "prescribed" || value == "prescribeddata" || value == "data") {
    return ls::LevelSetFieldSource::PrescribedData;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Level_set_source must be one of 'unknown' or 'prescribed_data'.");
}

ls::LevelSetVelocitySource parse_velocity_source(std::string_view raw)
{
  const auto value = normalized_token(std::string(raw));
  if (value == "coupled" || value == "coupledfield" || value == "unknown" || value == "navierstokes") {
    return ls::LevelSetVelocitySource::CoupledField;
  }
  if (value == "prescribed" || value == "prescribeddata" || value == "data") {
    return ls::LevelSetVelocitySource::PrescribedData;
  }
  if (value == "constant" || value == "constantvector") {
    return ls::LevelSetVelocitySource::ConstantVector;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Velocity_source must be one of 'coupled_field', 'prescribed_data', or 'constant'.");
}

ls::LevelSetTransportForm parse_transport_form(std::string_view raw)
{
  const auto value = normalized_token(std::string(raw));
  if (value == "advective" || value == "classical" || value == "standard") {
    return ls::LevelSetTransportForm::Advective;
  }
  if (value == "conservative" ||
      value == "conservativedivergence" ||
      value == "divergence" ||
      value == "divergenceform") {
    return ls::LevelSetTransportForm::ConservativeDivergence;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Level-set Transport_form must be one of 'advective' or 'conservative_divergence'.");
}

ls::LevelSetReinitializationMethod parse_reinitialization_method(std::string_view raw)
{
  const auto value = normalized_token(std::string(raw));
  if (value == "hamiltonjacobi" || value == "hamiltonjacobipde" || value == "pde") {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Reinitialization_method=HamiltonJacobiPDE "
        "is reserved until runtime Hamilton-Jacobi reinitialization is implemented; "
        "use 'Projection'.");
  }
  if (value == "fastmarching" || value == "fastmarchingmethod" || value == "fmm") {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Reinitialization_method=FastMarching "
        "is reserved until runtime fast-marching reinitialization is implemented; "
        "use 'Projection'.");
  }
  if (value == "projection" || value == "signeddistanceprojection" || value == "repairprojection") {
    return ls::LevelSetReinitializationMethod::Projection;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Reinitialization_method currently supports 'Projection' only.");
}

void apply_level_set_params(const svmp::Physics::ParameterMap& params,
                            ls::LevelSetTransportOptions& options)
{
  if (const auto value = get_defined_string(params, {"Operator_tag", "OperatorTag"})) {
    options.operator_tag = *value;
  }
  if (const auto value = get_defined_bool(params, {"Coupled"})) {
    if (*value && !get_defined_string(params, {"Operator_tag", "OperatorTag"})) {
      options.operator_tag = "equations";
    }
  }
  if (const auto value = get_defined_string(
          params,
          {"Transport_form", "TransportForm", "Advection_form", "AdvectionForm",
           "Level_set_transport_form", "LevelSetTransportForm"})) {
    options.transport_form = parse_transport_form(*value);
  }

  if (const auto value = get_defined_string(
          params,
          {"Level_set_field_name", "LevelSetFieldName", "Level_set_field", "LevelSetField", "Field_name"})) {
    options.level_set.field_name = *value;
  }
  if (const auto value = get_defined_string(params, {"Level_set_source", "LevelSetSource"})) {
    options.level_set.source = parse_level_set_source(*value);
  }
  if (const auto value = get_defined_bool(
          params,
          {"Auto_register_level_set_field", "AutoRegisterLevelSetField"})) {
    options.level_set.auto_register_field = *value;
  }

  if (const auto value = get_defined_string(
          params,
          {"Velocity_field_name", "VelocityFieldName", "Advection_velocity_field", "AdvectionVelocityField"})) {
    options.velocity.field_name = *value;
  }
  if (const auto value = get_defined_string(params, {"Velocity_source", "VelocitySource"})) {
    options.velocity.source = parse_velocity_source(*value);
    if (options.velocity.source == ls::LevelSetVelocitySource::PrescribedData) {
      options.velocity.auto_register_field = true;
    } else if (options.velocity.source == ls::LevelSetVelocitySource::ConstantVector) {
      options.velocity.auto_register_field = false;
    }
  }
  if (const auto value = get_defined_bool(
          params,
          {"Auto_register_velocity_field", "AutoRegisterVelocityField"})) {
    options.velocity.auto_register_field = *value;
  }
  if (const auto value = get_defined_vector3(
          params,
          {"Constant_velocity", "ConstantVelocity", "Velocity_value", "VelocityValue"},
          "Constant_velocity")) {
    options.velocity.source = ls::LevelSetVelocitySource::ConstantVector;
    options.velocity.auto_register_field = false;
    options.velocity.constant_value = *value;
  }

  if (const auto value = get_defined_bool(params, {"Enable_SUPG", "SUPG", "SUPG_enabled"})) {
    options.supg.enabled = *value;
  }
  if (const auto value = get_defined_real(params, {"SUPG_tau_scale", "SUPGScale"}, "SUPG_tau_scale")) {
    options.supg.tau_scale = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"SUPG_velocity_epsilon", "SUPGVelocityEpsilon"},
          "SUPG_velocity_epsilon")) {
    options.supg.velocity_epsilon = *value;
  }

  if (const auto value = get_defined_bool(
          params,
          {"Enable_interface_kinematic", "EnableInterfaceKinematic",
           "Enable_free_surface_kinematic_interface", "EnableFreeSurfaceKinematicInterface"})) {
    options.interface_kinematic.enabled = *value;
  }
  if (const auto value = get_defined_int(
          params,
          {"Interface_kinematic_marker", "InterfaceKinematicMarker",
           "Level_set_interface_marker", "LevelSetInterfaceMarker"},
          "Interface_kinematic_marker")) {
    options.interface_kinematic.enabled = true;
    options.interface_kinematic.interface_marker = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Interface_kinematic_weight_scale", "InterfaceKinematicWeightScale",
           "Free_surface_kinematic_weight_scale", "FreeSurfaceKinematicWeightScale"},
          "Interface_kinematic_weight_scale")) {
    options.interface_kinematic.weight_scale = *value;
  }

  if (const auto value = get_defined_bool(
          params,
          {"Enable_reinitialization", "Enable_level_set_reinitialization",
           "Reinitialization", "Reinitialization_enabled", "Reinitialize_level_set"})) {
    options.reinitialization.enabled = *value;
  }
  if (const auto value = get_defined_string(
          params,
          {"Reinitialization_method", "Level_set_reinitialization_method", "ReinitializationMethod"})) {
    options.reinitialization.method = parse_reinitialization_method(*value);
  }
  if (const auto value = get_defined_positive_int(
          params,
          {"Reinitialization_cadence_steps", "Reinitialization_cadence",
           "Level_set_reinitialization_cadence_steps", "ReinitializationCadenceSteps"},
          "Reinitialization_cadence_steps")) {
    options.reinitialization.cadence_steps = *value;
  }
  if (const auto value = get_defined_positive_int(
          params,
          {"Reinitialization_max_iterations", "Reinitialization_iterations",
           "ReinitializationMaxIterations"},
          "Reinitialization_max_iterations")) {
    options.reinitialization.max_iterations = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Reinitialization_pseudo_time_step_scale", "ReinitializationPseudoTimeStepScale"},
          "Reinitialization_pseudo_time_step_scale")) {
    options.reinitialization.pseudo_time_step_scale = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Reinitialization_interface_band_width", "ReinitializationInterfaceBandWidth"},
          "Reinitialization_interface_band_width")) {
    options.reinitialization.interface_band_width = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Reinitialization_signed_distance_tolerance", "ReinitializationSignedDistanceTolerance"},
          "Reinitialization_signed_distance_tolerance")) {
    options.reinitialization.signed_distance_tolerance = *value;
  }

  if (const auto value = get_defined_bool(
          params,
          {"Enable_volume_correction", "Enable_level_set_volume_correction",
           "Volume_correction", "VolumeCorrection", "Correct_level_set_volume"})) {
    options.volume_correction.enabled = *value;
  }
  if (const auto value = get_defined_positive_int(
          params,
          {"Volume_correction_cadence_steps", "Volume_correction_cadence",
           "Level_set_volume_correction_cadence_steps", "VolumeCorrectionCadenceSteps"},
          "Volume_correction_cadence_steps")) {
    options.volume_correction.cadence_steps = *value;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Volume_correction_use_initial_volume", "Use_initial_level_set_volume_as_target",
           "VolumeCorrectionUseInitialVolume"})) {
    options.volume_correction.use_initial_negative_volume_as_target = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Volume_correction_target_negative_volume",
           "Level_set_volume_correction_target_negative_volume",
           "VolumeCorrectionTargetNegativeVolume"},
          "Volume_correction_target_negative_volume")) {
    options.volume_correction.target_negative_volume = *value;
    options.volume_correction.use_initial_negative_volume_as_target = false;
  }
  if (const auto value = get_defined_real(
          params,
          {"Volume_correction_tolerance", "Volume_correction_volume_tolerance",
           "Level_set_volume_correction_tolerance", "VolumeCorrectionTolerance"},
          "Volume_correction_tolerance")) {
    options.volume_correction.volume_tolerance = *value;
  }
  if (const auto value = get_defined_positive_int(
          params,
          {"Volume_correction_max_iterations", "VolumeCorrectionMaxIterations"},
          "Volume_correction_max_iterations")) {
    options.volume_correction.max_iterations = *value;
  }
}

std::optional<std::string> projected_curvature_field_name(
    const svmp::Physics::ParameterMap& params)
{
  return get_defined_string(
      params,
      {"Curvature_field_name",
       "CurvatureFieldName",
       "Curvature_field",
       "CurvatureField",
       "Projected_curvature_field",
       "ProjectedCurvatureField",
       "Free_surface_curvature_field",
       "FreeSurfaceCurvatureField"});
}

void ensure_projected_curvature_field(
    svmp::FE::systems::FESystem& system,
    const std::string& field_name,
    const std::shared_ptr<const svmp::FE::spaces::FunctionSpace>& space)
{
  if (field_name.empty()) {
    return;
  }
  const auto existing = system.findFieldByName(field_name);
  if (existing != svmp::FE::INVALID_FIELD_ID) {
    const auto& rec = system.fieldRecord(existing);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Projected curvature field '" +
          field_name + "' must be scalar.");
    }
    if (rec.source_kind != svmp::FE::systems::FieldSourceKind::PrescribedData) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Projected curvature field '" +
          field_name + "' must be registered as PrescribedData.");
    }
    return;
  }
  if (!space) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Projected curvature field auto-registration requires a scalar function space.");
  }
  system.addField(svmp::FE::systems::FieldSpec{
      .name = field_name,
      .space = space,
      .components = 1,
      .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
  });
}

void ensure_projected_curvature_fields_from_input(
    svmp::FE::systems::FESystem& system,
    const svmp::Physics::EquationModuleInput& input,
    const std::shared_ptr<const svmp::FE::spaces::FunctionSpace>& space)
{
  if (const auto field = projected_curvature_field_name(input.equation_params)) {
    ensure_projected_curvature_field(system, *field, space);
  }
  if (const auto field = projected_curvature_field_name(input.default_domain.params)) {
    ensure_projected_curvature_field(system, *field, space);
  }
  for (const auto& domain : input.domains) {
    if (const auto field = projected_curvature_field_name(domain.params)) {
      ensure_projected_curvature_field(system, *field, space);
    }
  }
}

void apply_level_set_bcs(const svmp::Physics::EquationModuleInput& input,
                         ls::LevelSetTransportOptions& options)
{
  for (const auto& bc : input.boundary_conditions) {
    if (bc.boundary_marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto* type_param = find_param(bc.params, "Type");
    const std::string type = type_param ? normalized_token(type_param->value) : std::string{};

    if (type == "levelsetinflow" || type == "inflow" || type == "levelsetdirichlet") {
      ls::LevelSetInflowBoundary inflow{};
      inflow.boundary_marker = bc.boundary_marker;
      const auto* temporal_spatial_file =
          find_param(bc.params, "Temporal_and_spatial_values_file_path");
      const bool has_temporal_spatial_file =
          temporal_spatial_file != nullptr &&
          temporal_spatial_file->defined &&
          !trim_copy(temporal_spatial_file->value).empty();
      const auto value = get_defined_real(
          bc.params,
          {"Value", "Level_set_value"},
          "Add_BC/Value");
      if (has_temporal_spatial_file && value.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set inflow boundary '" + bc.name +
            "' must define either Value/Level_set_value or Temporal_and_spatial_values_file_path, not both.");
      }
      if (has_temporal_spatial_file) {
        if (!input.mesh) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set temporal/spatial inflow boundary '" + bc.name +
              "' requires an input mesh.");
        }
        auto data = read_level_set_temporal_spatial_inflow_file(
            *input.mesh,
            trim_copy(temporal_spatial_file->value));
        inflow.value = svmp::FE::forms::TimeScalarCoefficient(
            [data](svmp::FE::Real x,
                   svmp::FE::Real y,
                   svmp::FE::Real z,
                   svmp::FE::Real t) -> svmp::FE::Real {
              return data->interpolateSpatial({x, y, z}, t);
            });
      } else if (value.has_value()) {
        inflow.value = *value;
      }
      if (const auto penalty = get_defined_real(
              bc.params,
              {"Penalty_scale", "Penalty", "Inflow_penalty_scale"},
              "Add_BC/Penalty_scale")) {
        inflow.penalty_scale = *penalty;
      }
      options.boundaries.inflow.push_back(std::move(inflow));
      continue;
    }

    if (type == "levelsetoutflow" || type == "outflow") {
      options.boundaries.outflow.push_back(ls::LevelSetOutflowBoundary{.boundary_marker = bc.boundary_marker});
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Application] Boundary condition type '" +
        (type_param ? trim_copy(type_param->value) : std::string{}) +
        "' is not supported for the level-set transport module. Supported types: LevelSetInflow, LevelSetOutflow.");
  }
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_level_set_transport_from_input(const svmp::Physics::EquationModuleInput& input,
                                      svmp::FE::systems::FESystem& system)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Application] Level-set transport module factory received null mesh.");
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int order = resolve_element_order(input, infer_polynomial_order(*input.mesh));
  const auto basis_type = resolve_basis_type(input);
  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error("[svMultiPhysics::Application] Level-set transport requires a mesh dimension in [1, 3].");
  }

  auto level_set_space = svmp::FE::spaces::SpaceFactory::create_h1(
      element_type, order, basis_type);
  auto velocity_scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
      element_type, order, basis_type);
  auto velocity_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(
          velocity_scalar_space, dim);

  ls::LevelSetTransportOptions options{};
  const auto install_options =
      level_set_install_options(svmp::Physics::core::resolveOopJitPolicy(input));

  apply_level_set_params(input.equation_params, options);
  apply_level_set_params(input.default_domain.params, options);
  if (!input.domains.empty()) {
    if (input.domains.size() != 1u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Multiple <Domain> blocks are not supported by the level-set transport module.");
    }
    apply_level_set_params(input.domains.front().params, options);
  }
  apply_level_set_bcs(input, options);
  ensure_projected_curvature_fields_from_input(
      system, input, level_set_space);

  options.velocity.space = velocity_space;

  auto module = std::make_unique<LevelSetTransportInputAdapter>(
      std::move(level_set_space),
      std::move(options),
      install_options);
  module->registerOn(system);
  return module;
}

} // namespace

namespace application {
namespace translators {
namespace level_set {

bool isEquationType(std::string_view type)
{
  return type == "level_set" ||
         type == "levelSet" ||
         type == "level_set_transport";
}

std::vector<std::string> equationTypes()
{
  return {"level_set", "levelSet", "level_set_transport"};
}

std::unique_ptr<svmp::Physics::PhysicsModule>
createModule(const svmp::Physics::EquationModuleInput& input,
             svmp::FE::systems::FESystem& system)
{
  return create_level_set_transport_from_input(input, system);
}

} // namespace level_set
} // namespace translators
} // namespace application

#include "Application/Translators/LevelSetEquationTranslator.h"

#include "Application/Core/LevelSetEquationInputSnapshot.h"
#include "Application/Core/LevelSetPdeVelocityExtension.h"
#include "Application/Translators/LevelSetConfigurationParsing.h"
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
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <locale>
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
namespace parsing = application::translators::level_set::configuration;
namespace level_set_aliases = parsing::aliases;

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

std::string json_string(std::string_view value)
{
  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  out.reserve(value.size() + 2u);
  out.push_back('"');
  for (const unsigned char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20u) {
          out += "\\u00";
          out.push_back(hex[(c >> 4u) & 0x0fu]);
          out.push_back(hex[c & 0x0fu]);
        } else {
          out.push_back(static_cast<char>(c));
        }
        break;
    }
  }
  out.push_back('"');
  return out;
}

std::string json_real(svmp::FE::Real value)
{
  if (!std::isfinite(value)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Effective level-set configuration contains a non-finite scalar.");
  }
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(
             std::numeric_limits<svmp::FE::Real>::max_digits10)
      << value;
  return out.str();
}

constexpr const char* json_bool(bool value) noexcept
{
  return value ? "true" : "false";
}

std::string phase_region_boxes_json(
    const std::vector<ls::LevelSetPhaseRegionBox>& boxes)
{
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << '[';
  for (std::size_t index = 0u; index < boxes.size(); ++index) {
    if (index != 0u) {
      out << ',';
    }
    const auto& box = boxes[index];
    out << "{\"name\":" << json_string(box.name)
        << ",\"kind\":"
        << json_string(ls::levelSetPhaseRegionKindName(box.kind))
        << ",\"minimum\":["
        << json_real(box.minimum[0]) << ','
        << json_real(box.minimum[1]) << ','
        << json_real(box.minimum[2]) << ']'
        << ",\"maximum\":["
        << json_real(box.maximum[0]) << ','
        << json_real(box.maximum[1]) << ','
        << json_real(box.maximum[2]) << "]}";
  }
  out << ']';
  return out.str();
}

std::string json_scalar_value(const ls::ScalarValue& value)
{
  if (const auto* literal = std::get_if<svmp::FE::Real>(&value)) {
    return json_real(*literal);
  }
  if (std::holds_alternative<svmp::FE::forms::ScalarCoefficient>(value)) {
    return R"({"kind":"spatial_coefficient"})";
  }
  if (std::holds_alternative<svmp::FE::forms::TimeScalarCoefficient>(value)) {
    return R"({"kind":"time_coefficient"})";
  }
  return R"({"kind":"form_expression"})";
}

const char* level_set_field_source_name(ls::LevelSetFieldSource source) noexcept
{
  return source == ls::LevelSetFieldSource::Unknown
             ? "Unknown"
             : "PrescribedData";
}

const char* level_set_velocity_source_name(
    ls::LevelSetVelocitySource source) noexcept
{
  switch (source) {
    case ls::LevelSetVelocitySource::CoupledField: return "CoupledField";
    case ls::LevelSetVelocitySource::PrescribedData: return "PrescribedData";
    case ls::LevelSetVelocitySource::ConstantVector: return "ConstantVector";
    case ls::LevelSetVelocitySource::MaterialInterfacePhasePair:
      return "MaterialInterfacePhasePair";
  }
  return "Unknown";
}

const char* level_set_transport_form_name(ls::LevelSetTransportForm form) noexcept
{
  return form == ls::LevelSetTransportForm::Advective
             ? "Advective"
             : "ConservativeDivergence";
}

const char* level_set_phase_side_name(ls::LevelSetPhaseSide side) noexcept
{
  return side == ls::LevelSetPhaseSide::Negative ? "Negative" : "Positive";
}

svmp::Physics::EffectiveConfigurationArtifact
make_level_set_effective_configuration(const ls::LevelSetTransportOptions& options)
{
  auto inflow = options.boundaries.inflow;
  auto outflow = options.boundaries.outflow;
  std::sort(inflow.begin(), inflow.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.boundary_marker < rhs.boundary_marker;
  });
  std::sort(outflow.begin(), outflow.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.boundary_marker < rhs.boundary_marker;
  });

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "{\"artifact_schema_version\":3"
      << ",\"component\":\"level_set_transport\""
      << ",\"capability_label\":"
      << json_string(
             options.velocity.source ==
                     ls::LevelSetVelocitySource::MaterialInterfacePhasePair
                 ? "two_phase_material_interface_transport"
             : options.conservative_phase.enabled
                 ? "one_phase_locally_conservative_p1_indicator_transport"
                 : "one_phase_interface_transport_nonlocal_conservation")
      << ",\"units\":{\"system\":\"consistent_solver_units\",\"length\":\"solver_length\",\"time\":\"solver_time\",\"volume\":\"solver_volume\"}"
      << ",\"operator\":" << json_string(options.operator_tag)
      << ",\"transport_form\":"
      << json_string(level_set_transport_form_name(options.transport_form))
      << ",\"conservation_diagnostic\":"
      << json_string(ls::levelSetConservationDiagnosticName(
             ls::levelSetConservationDiagnostic(options)))
      << ",\"level_set\":{\"field\":"
      << json_string(options.level_set.field_name)
      << ",\"source\":"
      << json_string(level_set_field_source_name(options.level_set.source))
      << ",\"auto_register\":"
      << json_bool(options.level_set.auto_register_field) << '}'
      << ",\"conservative_phase\":{\"enabled\":"
      << json_bool(options.conservative_phase.enabled)
      << ",\"field\":"
      << json_string(
             options.conservative_phase.liquid_indicator.field_name)
      << ",\"source\":"
      << json_string(level_set_field_source_name(
             options.conservative_phase.liquid_indicator.source))
      << ",\"auto_register\":"
      << json_bool(
             options.conservative_phase.liquid_indicator.auto_register_field)
      << ",\"liquid_side\":"
      << json_string(level_set_phase_side_name(
             options.conservative_phase.liquid_side))
      << ",\"invariant_tolerance\":"
      << json_real(options.conservative_phase.invariant_tolerance)
      << ",\"component_activity_tolerance\":"
      << json_real(options.conservative_phase.component_activity_tolerance)
      << ",\"maximum_courant\":"
      << json_real(options.conservative_phase.maximum_courant)
      << ",\"enforce_courant_limit\":"
      << json_bool(options.conservative_phase.enforce_courant_limit)
      << ",\"require_constant_preservation\":"
      << json_bool(
             options.conservative_phase.require_constant_preservation)
      << ",\"momentum_relative_tolerance\":"
      << json_real(options.conservative_phase.momentum_relative_tolerance)
      << ",\"momentum_reconciliation_policy\":\"phasewise_fail_closed_no_hidden_velocity_update\""
      << ",\"write_flux_artifacts\":"
      << json_bool(options.conservative_phase.write_flux_artifacts)
      << ",\"flux_artifact_cadence_steps\":"
      << options.conservative_phase.flux_artifact_cadence_steps
      << ",\"classify_nonprimary_components_as_satellites\":"
      << json_bool(options.conservative_phase
                       .classify_nonprimary_components_as_satellites)
      << ",\"fixed_flux_regions\":"
      << phase_region_boxes_json(
             options.conservative_phase.fixed_flux_regions)
      << ",\"impermeable_normal_velocity_tolerance\":"
      << json_real(options.conservative_phase
                       .impermeable_normal_velocity_tolerance)
      << ",\"pointwise_impermeable_velocity_tolerance_explicitly_requested\":"
      << json_bool(options.conservative_phase
                       .pointwise_impermeable_velocity_tolerance_explicitly_requested)
      << ",\"pointwise_impermeable_velocity_contract\":\"unsupported_fail_closed_when_explicit\""
      << ",\"reconcile_geometry\":"
      << json_bool(options.conservative_phase.reconcile_geometry)
      << ",\"geometry_measure_tolerance\":"
      << json_real(options.conservative_phase.geometry_measure_tolerance)
      << ",\"geometry_correction_max_iterations\":"
      << options.conservative_phase.geometry_correction_max_iterations
      << ",\"maximum_geometry_displacement_fraction\":"
      << json_real(options.conservative_phase
                       .maximum_geometry_displacement_fraction)
      << ",\"boundary_flux_policy\":"
      << json_string(
             ls::levelSetConservativePhaseBoundaryFluxPolicyName(
                 options.conservative_phase.boundary_flux_policy))
      << ",\"boundary_flux_tolerance_source\":\"invariant_tolerance\""
      << ",\"boundary_flux_blind_where_q_zero\":true"
      << ",\"newton_policy\":\"held_at_previous_accepted_endpoint\"}"
      << ",\"advection_velocity\":{\"field\":"
      << json_string(options.velocity.field_name)
      << ",\"source\":"
      << json_string(level_set_velocity_source_name(options.velocity.source))
      << ",\"material_interface_marker\":"
      << options.velocity.material_interface_marker
      << ",\"auto_register\":"
      << json_bool(options.velocity.auto_register_field)
      << ",\"constant_value\":["
      << json_real(options.velocity.constant_value[0]) << ','
      << json_real(options.velocity.constant_value[1]) << ','
      << json_real(options.velocity.constant_value[2]) << ']'
      << ",\"algebraic_extension_source_field\":"
      << json_string(options.velocity.algebraic_extension_source_field_name)
      << ",\"dependency_direction\":\"physical_velocity_to_extension_to_level_set\""
      << ",\"physical_momentum_coupling_allowed\":false"
      << ",\"map_guard_policy\":\"fixed_bounded_application_policy\"}"
      << ",\"supg\":{\"enabled\":" << json_bool(options.supg.enabled)
      << ",\"tau_scale\":" << json_real(options.supg.tau_scale)
      << ",\"velocity_epsilon\":"
      << json_real(options.supg.velocity_epsilon)
      << ",\"transient_scale\":"
      << json_real(options.supg.transient_scale) << '}'
      << ",\"interface_kinematic\":{\"enabled\":"
      << json_bool(options.interface_kinematic.enabled)
      << ",\"interface_marker\":"
      << options.interface_kinematic.interface_marker
      << ",\"weight_scale\":"
      << json_real(options.interface_kinematic.weight_scale) << '}'
      << ",\"maintenance_transaction\":{\"ordering\":"
      << json_string(
             options.conservative_phase.enabled
                 ? "conservative_phase_transport_then_raw_geometry_rebuild_then_local_geometry_reconciliation_then_validation_then_commit"
                 : "transport_then_geometry_refresh");
  // Present only when enabled, so existing configurations keep their
  // artifact unchanged.
  if (options.kinematic_reconciliation.enabled) {
    out << ",\"kinematic_reconciliation\":{\"enabled\":true}";
  }
  if (options.sign_definite_patch_bounds.enabled) {
    out << ",\"sign_definite_patch_bounds\":{\"enabled\":true}";
  }
  if (options.interface_preserving_redistancing.enabled) {
    out << ",\"interface_preserving_redistancing\":{\"enabled\":true}";
  }
  out << '}'
      << ",\"boundaries\":{\"inflow\":[";
  for (std::size_t i = 0; i < inflow.size(); ++i) {
    if (i != 0u) out << ',';
    out << "{\"marker\":" << inflow[i].boundary_marker
        << ",\"value\":" << json_scalar_value(inflow[i].value)
        << ",\"penalty_scale\":" << json_real(inflow[i].penalty_scale)
        << '}';
  }
  out << "],\"outflow\":[";
  for (std::size_t i = 0; i < outflow.size(); ++i) {
    if (i != 0u) out << ',';
    out << "{\"marker\":" << outflow[i].boundary_marker << '}';
  }
  out << "]}}";

  return svmp::Physics::EffectiveConfigurationArtifact{
      .component = "level_set_transport",
      .json = out.str(),
  };
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
    effective_configuration_artifact_.reset();
    (void)ls::installLevelSetTransport(
        system,
        level_set_space_,
        options_,
        install_options_);
    effective_configuration_artifact_ =
        make_level_set_effective_configuration(options_);
  }

  [[nodiscard]] std::optional<svmp::Physics::EffectiveConfigurationArtifact>
  effectiveConfigurationArtifact() const override
  {
    return effective_configuration_artifact_;
  }

  void applyInitialConditions(const svmp::FE::systems::FESystem& system,
                              svmp::FE::backends::GenericVector& u0) const override
  {
    const bool initialize_from_mesh =
        options_.level_set.source == ls::LevelSetFieldSource::PrescribedData ||
        options_.velocity.source ==
            ls::LevelSetVelocitySource::MaterialInterfacePhasePair;
    if (!initialize_from_mesh) {
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
        "[svMultiPhysics::Application] Level-set mesh initialization requires mesh support.");
#endif
  }

private:
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> level_set_space_{};
  ls::LevelSetTransportOptions options_{};
  svmp::FE::systems::FormInstallOptions install_options_{};
  mutable std::optional<svmp::Physics::EffectiveConfigurationArtifact>
      effective_configuration_artifact_{};
};

std::string trim_copy(std::string s)
{
  return application::translators::level_set::configuration::trimCopy(
      std::move(s));
}

std::string normalized_token(std::string s)
{
  return application::translators::level_set::configuration::normalizedToken(
      std::move(s));
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
  using namespace application::translators::level_set::configuration;
  const LevelSetConfigurationReader reader(
      params, LevelSetInputPolicy::Installation);
  const auto value = reader.selected(keys);
  return value ? std::optional<std::string>{value->text} : std::nullopt;
}

std::optional<bool> get_defined_bool(const svmp::Physics::ParameterMap& params,
                                     std::initializer_list<std::string_view> keys)
{
  using namespace application::translators::level_set::configuration;
  LevelSetConfigurationReader reader(params, LevelSetInputPolicy::Installation);
  return reader.boolean(keys);
}

std::optional<svmp::FE::Real> get_defined_real(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  using namespace application::translators::level_set::configuration;
  LevelSetConfigurationReader reader(params, LevelSetInputPolicy::Installation);
  return reader.real(keys, context);
}

std::optional<int> get_defined_int(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  using namespace application::translators::level_set::configuration;
  LevelSetConfigurationReader reader(params, LevelSetInputPolicy::Installation);
  return reader.integer(keys, context);
}

std::optional<std::array<svmp::FE::Real, 3>> get_defined_vector3(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  using namespace application::translators::level_set::configuration;
  LevelSetConfigurationReader reader(params, LevelSetInputPolicy::Installation);
  return reader.vector3(keys, context);
}

std::optional<int> get_defined_positive_int(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  using namespace application::translators::level_set::configuration;
  LevelSetConfigurationReader reader(params, LevelSetInputPolicy::Installation);
  return reader.integer(keys, context, true);
}

template <std::size_t N>
std::optional<std::string> get_defined_string(
    const svmp::Physics::ParameterMap& params,
    const std::array<std::string_view, N>& keys)
{
  parsing::LevelSetConfigurationReader reader(
      params, parsing::LevelSetInputPolicy::Installation);
  const auto value = reader.selected(keys);
  return value ? std::optional<std::string>{value->text} : std::nullopt;
}

template <std::size_t N>
std::optional<bool> get_defined_bool(
    const svmp::Physics::ParameterMap& params,
    const std::array<std::string_view, N>& keys)
{
  parsing::LevelSetConfigurationReader reader(
      params, parsing::LevelSetInputPolicy::Installation);
  return reader.boolean(keys);
}

template <std::size_t N>
std::optional<svmp::FE::Real> get_defined_real(
    const svmp::Physics::ParameterMap& params,
    const std::array<std::string_view, N>& keys,
    std::string_view context)
{
  parsing::LevelSetConfigurationReader reader(
      params, parsing::LevelSetInputPolicy::Installation);
  return reader.real(keys, context);
}

template <std::size_t N>
std::optional<int> get_defined_int(
    const svmp::Physics::ParameterMap& params,
    const std::array<std::string_view, N>& keys,
    std::string_view context)
{
  parsing::LevelSetConfigurationReader reader(
      params, parsing::LevelSetInputPolicy::Installation);
  return reader.integer(keys, context);
}

template <std::size_t N>
std::optional<std::array<svmp::FE::Real, 3>> get_defined_vector3(
    const svmp::Physics::ParameterMap& params,
    const std::array<std::string_view, N>& keys,
    std::string_view context)
{
  parsing::LevelSetConfigurationReader reader(
      params, parsing::LevelSetInputPolicy::Installation);
  return reader.vector3(keys, context);
}

template <std::size_t N>
std::optional<int> get_defined_positive_int(
    const svmp::Physics::ParameterMap& params,
    const std::array<std::string_view, N>& keys,
    std::string_view context)
{
  parsing::LevelSetConfigurationReader reader(
      params, parsing::LevelSetInputPolicy::Installation);
  return reader.integer(keys, context, true);
}

svmp::FE::systems::FormInstallOptions level_set_install_options(
    const svmp::Physics::core::PhysicsJITPolicy& policy)
{
  svmp::FE::systems::FormInstallOptions options{};
  options.compiler_options.jit.enable =
      svmp::Physics::core::effectivePhysicsJITEnable(policy);
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
    return parsing::parseStrictPositiveInteger(p->value, "Element_order");
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
  using namespace application::translators::level_set::configuration;
  return parseVelocitySource(raw, LevelSetInputPolicy::Installation);
}

ls::LevelSetTransportForm parse_transport_form(std::string_view raw)
{
  return application::translators::level_set::configuration::parseTransportForm(
      raw);
}

ls::LevelSetPhaseSide parse_phase_side(std::string_view raw)
{
  return application::translators::level_set::configuration::parsePhaseSide(raw);
}

ls::LevelSetConservativePhaseBoundaryFluxPolicy
parse_conservative_phase_boundary_flux_policy(std::string_view raw)
{
  return application::translators::level_set::configuration::
      parseConservativePhaseBoundaryFluxPolicy(raw);
}

void append_installation_selection(
    std::vector<application::core::LevelSetInputObservation>& observations,
    const parsing::LevelSetSelectedParameter& selection,
    std::string_view source_layer)
{
  observations.push_back(application::core::LevelSetInputObservation{
      .canonical_key = selection.canonical_key,
      .selected_spelling = selection.selected_spelling,
      .source_layer = std::string(source_layer),
      .supplied = selection.supplied,
      .representation = "installation_snapshot",
      .compatibility_fallback = false,
      .ordered_overrides = {},
  });
}

void append_installation_selections(
    std::vector<application::core::LevelSetInputObservation>& observations,
    const parsing::LevelSetConfigurationReader& reader,
    std::string_view source_layer)
{
  for (const auto& selection : reader.selections()) {
    append_installation_selection(observations, selection, source_layer);
  }
}

void append_installation_derived(
    std::vector<application::core::LevelSetInputObservation>& observations,
    std::string_view canonical_key, std::string_view source_layer,
    std::string_view originating_layer = {})
{
  observations.push_back(application::core::LevelSetInputObservation{
      .canonical_key = std::string(canonical_key),
      .selected_spelling = {},
      .source_layer = std::string(source_layer),
      .supplied = false,
      .representation = "installation_derived",
      .compatibility_fallback = false,
      .ordered_overrides = originating_layer.empty()
                               ? std::vector<std::string>{}
                               : std::vector<std::string>{
                                     std::string(originating_layer)},
  });
}

void append_installation_default(
    std::vector<application::core::LevelSetInputObservation>& observations,
    std::string_view canonical_key)
{
  observations.push_back(application::core::LevelSetInputObservation{
      .canonical_key = std::string(canonical_key),
      .selected_spelling = {},
      .source_layer = "fe_default",
      .supplied = false,
      .representation = "installation_default",
      .compatibility_fallback = false,
      .ordered_overrides = {},
  });
}

bool has_installation_observation(
    const std::vector<application::core::LevelSetInputObservation>& observations,
    std::string_view canonical_key)
{
  return std::any_of(observations.begin(), observations.end(),
                     [&](const auto& observation) {
                       return observation.canonical_key == canonical_key;
                     });
}

void apply_level_set_params(const svmp::Physics::ParameterMap& params,
                            ls::LevelSetTransportOptions& options,
                            std::vector<application::core::LevelSetInputObservation>&
                                observations,
                            std::string_view source_layer)
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
          params, level_set_aliases::transport_form)) {
    options.transport_form = parse_transport_form(*value);
  }

  if (const auto value = get_defined_string(
          params, level_set_aliases::level_set_field_name)) {
    options.level_set.field_name = *value;
  }
  if (const auto value = get_defined_string(params, {"Level_set_source"})) {
    options.level_set.source = parse_level_set_source(*value);
  }
  if (const auto value = get_defined_bool(
          params,
          {"Auto_register_level_set_field"})) {
    options.level_set.auto_register_field = *value;
  }

  if (const auto value = get_defined_bool(
          params, level_set_aliases::conservative_phase_enable)) {
    options.conservative_phase.enabled = *value;
  }
  if (const auto value = get_defined_string(
          params, level_set_aliases::conservative_phase_field)) {
    options.conservative_phase.liquid_indicator.field_name = *value;
  }
  if (const auto value = get_defined_bool(
          params, level_set_aliases::conservative_phase_auto_register)) {
    options.conservative_phase.liquid_indicator.auto_register_field = *value;
  }
  if (const auto value = get_defined_string(
          params, level_set_aliases::conservative_phase_side)) {
    options.conservative_phase.liquid_side = parse_phase_side(*value);
  }
  if (const auto value = get_defined_real(
          params,
          {"Conservative_phase_invariant_tolerance"},
          "Conservative_phase_invariant_tolerance")) {
    options.conservative_phase.invariant_tolerance = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Conservative_phase_component_activity_tolerance"},
          "Conservative_phase_component_activity_tolerance")) {
    options.conservative_phase.component_activity_tolerance = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Conservative_phase_maximum_courant"},
          "Conservative_phase_maximum_courant")) {
    options.conservative_phase.maximum_courant = *value;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Conservative_phase_enforce_courant_limit"})) {
    options.conservative_phase.enforce_courant_limit = *value;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Conservative_phase_require_constant_preservation"})) {
    options.conservative_phase.require_constant_preservation = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Conservative_phase_momentum_relative_tolerance"},
          "Conservative_phase_momentum_relative_tolerance")) {
    options.conservative_phase.momentum_relative_tolerance = *value;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Conservative_phase_write_flux_artifacts"})) {
    options.conservative_phase.write_flux_artifacts = *value;
  }
  if (const auto value = get_defined_int(
          params,
          {"Conservative_phase_flux_artifact_cadence_steps"},
          "Conservative_phase_flux_artifact_cadence_steps")) {
    options.conservative_phase.flux_artifact_cadence_steps = *value;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Conservative_phase_classify_nonprimary_components_as_satellites"})) {
    options.conservative_phase
        .classify_nonprimary_components_as_satellites = *value;
  }
  if (const auto value = get_defined_string(
          params,
          {"Conservative_phase_boundary_flux_policy"})) {
    options.conservative_phase.boundary_flux_policy =
        parse_conservative_phase_boundary_flux_policy(*value);
  }
  if (const auto value = get_defined_string(
          params,
          {"Conservative_phase_fixed_flux_regions"})) {
    options.conservative_phase.fixed_flux_regions =
        ls::parseLevelSetPhaseRegionBoxes(*value);
  }
  if (const auto value = get_defined_real(
          params,
          {"Conservative_phase_impermeable_normal_velocity_tolerance"},
          "Conservative_phase_impermeable_normal_velocity_tolerance")) {
    options.conservative_phase.impermeable_normal_velocity_tolerance = *value;
    options.conservative_phase
        .pointwise_impermeable_velocity_tolerance_explicitly_requested = true;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Conservative_phase_reconcile_geometry"})) {
    options.conservative_phase.reconcile_geometry = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Conservative_phase_geometry_measure_tolerance"},
          "Conservative_phase_geometry_measure_tolerance")) {
    options.conservative_phase.geometry_measure_tolerance = *value;
  }
  if (const auto value = get_defined_positive_int(
          params,
          {"Conservative_phase_geometry_correction_max_iterations"},
          "Conservative_phase_geometry_correction_max_iterations")) {
    options.conservative_phase.geometry_correction_max_iterations = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Conservative_phase_maximum_geometry_displacement_fraction"},
          "Conservative_phase_maximum_geometry_displacement_fraction")) {
    options.conservative_phase.maximum_geometry_displacement_fraction = *value;
  }

  if (const auto value = get_defined_string(
          params, level_set_aliases::velocity_field_name)) {
    options.velocity.field_name = *value;
  }
  parsing::LevelSetConfigurationReader velocity_source_reader(
      params, parsing::LevelSetInputPolicy::Installation);
  if (const auto selected = velocity_source_reader.string(
          level_set_aliases::velocity_source, "velocity_source")) {
    options.velocity.source = parse_velocity_source(selected->text);
    append_installation_selections(observations, velocity_source_reader,
                                   source_layer);
    if (options.velocity.source == ls::LevelSetVelocitySource::PrescribedData) {
      options.velocity.auto_register_field = true;
      append_installation_derived(observations,
                                  "velocity_auto_register_field",
                                  "derived:velocity_source", source_layer);
    } else if (options.velocity.source == ls::LevelSetVelocitySource::ConstantVector) {
      options.velocity.auto_register_field = false;
      append_installation_derived(observations,
                                  "velocity_auto_register_field",
                                  "derived:velocity_source", source_layer);
    } else if (options.velocity.source ==
               ls::LevelSetVelocitySource::MaterialInterfacePhasePair) {
      options.velocity.auto_register_field = false;
      append_installation_derived(observations,
                                  "velocity_auto_register_field",
                                  "derived:velocity_source", source_layer);
    }
  }
  if (const auto value = get_defined_int(
          params, level_set_aliases::material_interface_marker,
          "Material_interface_marker")) {
    options.velocity.material_interface_marker = *value;
  }
  parsing::LevelSetConfigurationReader velocity_metadata_reader(
      params, parsing::LevelSetInputPolicy::Installation);
  if (const auto value = velocity_metadata_reader.boolean(
          level_set_aliases::velocity_auto_register,
          "velocity_auto_register_field")) {
    options.velocity.auto_register_field = *value;
  }
  append_installation_selections(observations, velocity_metadata_reader,
                                 source_layer);
  parsing::LevelSetConfigurationReader constant_velocity_reader(
      params, parsing::LevelSetInputPolicy::Installation);
  if (const auto value = constant_velocity_reader.vector3(
          level_set_aliases::constant_velocity, "Constant_velocity",
          "constant_velocity")) {
    options.velocity.source = ls::LevelSetVelocitySource::ConstantVector;
    options.velocity.auto_register_field = false;
    options.velocity.constant_value = *value;
    append_installation_selections(observations, constant_velocity_reader,
                                   source_layer);
    append_installation_derived(observations, "velocity_source",
                                "derived:constant_velocity", source_layer);
    append_installation_derived(observations,
                                "velocity_auto_register_field",
                                "derived:constant_velocity", source_layer);
  }

  if (const auto value = get_defined_bool(params, {"Enable_SUPG"})) {
    options.supg.enabled = *value;
  }
  if (const auto value = get_defined_real(params, {"SUPG_tau_scale"}, "SUPG_tau_scale")) {
    options.supg.tau_scale = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"SUPG_transient_scale"},
          "SUPG_transient_scale")) {
    options.supg.transient_scale = *value;
  }
  // The SUPG discontinuity capturing of the level-set transport was retired
  // (decision D47).  Decks may still state it as disabled.
  if (get_defined_bool(params, {"Enable_discontinuity_capturing"})
          .value_or(false) ||
      get_defined_string(params,
                         {"Discontinuity_capturing_scale",
                          "Discontinuity_capturing_gradient_epsilon",
                          "Discontinuity_capturing_max_courant"})
          .has_value()) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] Level-set discontinuity capturing "
        "(Enable_discontinuity_capturing and the Discontinuity_capturing_* "
        "settings) was retired (decision D47); no accepted benchmark used "
        "it.  Remove these keys or set Enable_discontinuity_capturing=false.");
  }
  // The bound-preserving limiter of the level-set transport was retired
  // (decision D51).  Decks may still state it as disabled.
  if (get_defined_bool(params, {"Enable_bound_preserving_limiter"})
          .value_or(false) ||
      get_defined_string(
          params,
          {"Bound_preserving_bound_tolerance",
           "Bound_preserving_sign_tolerance",
           "Bound_preserving_maximum_courant",
           "Bound_preserving_enforce_courant_limit",
           "Bound_preserving_enforce_impermeable_boundaries",
           "Bound_preserving_impermeable_normal_velocity_tolerance"})
          .has_value()) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] The level-set bound-preserving "
        "limiter (Enable_bound_preserving_limiter and the "
        "Bound_preserving_* settings) was retired (decision D51); with it "
        "on, the SPHERIC reruns failed at their first step, while the runs "
        "without it and the sign-definite patch bounds were clean.  Remove "
        "these keys or set Enable_bound_preserving_limiter=false.");
  }

  if (const auto value = get_defined_bool(
          params,
          {"Enable_interface_kinematic"})) {
    options.interface_kinematic.enabled = *value;
  }
  if (const auto value = get_defined_int(
          params,
          {"Interface_kinematic_marker"},
          "Interface_kinematic_marker")) {
    options.interface_kinematic.enabled = true;
    options.interface_kinematic.interface_marker = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"Interface_kinematic_weight_scale"},
          "Interface_kinematic_weight_scale")) {
    options.interface_kinematic.weight_scale = *value;
  }

  // Projection reinitialization was retired (decision D47).  Decks may
  // still state it as disabled.
  if (get_defined_bool(params, {"Enable_reinitialization"}).value_or(false) ||
      get_defined_string(
          params,
          {"Reinitialization_method",
           "Reinitialization_cadence_steps",
           "Reinitialization_max_iterations",
           "Reinitialization_pseudo_time_step_scale",
           "Reinitialization_interface_band_width",
           "Reinitialization_signed_distance_tolerance",
           "Reinitialization_max_zero_set_displacement"})
          .has_value()) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] Level-set projection "
        "reinitialization (Enable_reinitialization and the "
        "Reinitialization_* settings) was retired (decision D47); no "
        "accepted benchmark used it, and the interface-preserving "
        "redistancing is the accepted-step redistancing.  Remove these keys "
        "or set Enable_reinitialization=false.");
  }

  // The global level-set volume correction was retired (decision D47).
  // Decks may still state it as disabled.
  if (get_defined_bool(params, {"Enable_volume_correction"}).value_or(false) ||
      get_defined_string(
          params,
          {"Volume_correction_cadence_steps",
           "Volume_correction_use_initial_volume",
           "Volume_correction_target_negative_volume",
           "Volume_correction_tolerance",
           "Volume_correction_max_iterations",
           "Volume_correction_minimum_relative_error",
           "Volume_correction_maximum_interface_displacement_fraction",
           "Volume_correction_maximum_cumulative_interface_displacement_fraction"})
          .has_value()) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] The global level-set volume "
        "correction (Enable_volume_correction and the Volume_correction_* "
        "settings) was retired (decision D47); no accepted benchmark used "
        "it, and the kinematic reconciliation keeps the liquid volume "
        "consistent with the transport velocity.  Remove these keys or set "
        "Enable_volume_correction=false.");
  }
  if (const auto value = get_defined_bool(
          params,
          {"Enable_kinematic_reconciliation"})) {
    options.kinematic_reconciliation.enabled = *value;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Enable_sign_definite_patch_bounds"})) {
    options.sign_definite_patch_bounds.enabled = *value;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Enable_interface_preserving_redistancing"})) {
    options.interface_preserving_redistancing.enabled = *value;
  }
}

// The projected curvature field of the retired level-set curvature
// projection (decision D48): a field name at the equation or domain level
// fails at translation.
void reject_retired_curvature_field(
    const svmp::Physics::EquationModuleInput& input)
{
  const auto reject = [](const svmp::Physics::ParameterMap& params) {
    const auto field =
        get_defined_string(params, level_set_aliases::curvature_field);
    if (field.has_value() && !trim_copy(*field).empty()) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Curvature_field_name '" +
          trim_copy(*field) +
          "': The level-set curvature projection (Enable_curvature_projection, "
          "Curvature_field_name and the Curvature_projection_* settings) was "
          "retired (decision D48; its least-squares recovery modes by decision "
          "D47): SurfaceStress, the only capillary route, reads no curvature "
          "field.  Remove these keys.");
    }
  };
  reject(input.equation_params);
  reject(input.default_domain.params);
  for (const auto& domain : input.domains) {
    reject(domain.params);
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
              {"Penalty_scale", "Penalty"},
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

application::core::ResolvedLevelSetEquationConfiguration
translate_level_set_transport_input(
    const svmp::Physics::EquationModuleInput& input)
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
  std::vector<application::core::LevelSetInputObservation> input_observations;
  const auto install_options =
      level_set_install_options(svmp::Physics::core::resolveOopJitPolicy(input));

  apply_level_set_params(input.equation_params, options, input_observations,
                         "equation");
  apply_level_set_params(input.default_domain.params, options,
                         input_observations, "default_domain");
  if (!input.domains.empty()) {
    if (input.domains.size() != 1u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Multiple <Domain> blocks are not supported by the level-set transport module.");
    }
    apply_level_set_params(input.domains.front().params, options,
                           input_observations,
                           "domain:" + input.domains.front().id);
  }
  apply_level_set_bcs(input, options);

  const bool material_interface_velocity =
      options.velocity.source ==
      ls::LevelSetVelocitySource::MaterialInterfacePhasePair;
  if (material_interface_velocity) {
    if (options.velocity.material_interface_marker < 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Material_interface_marker is required for material-interface phase-pair velocity.");
    }
    const auto reject_explicitly_disabled_kinematic = [](const auto& params) {
      const auto enabled = get_defined_bool(
          params,
          {"Enable_interface_kinematic"});
      if (enabled.has_value() && !*enabled) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Material-interface phase-pair velocity requires interface kinematic enforcement.");
      }
    };
    reject_explicitly_disabled_kinematic(input.equation_params);
    reject_explicitly_disabled_kinematic(input.default_domain.params);
    for (const auto& domain : input.domains) {
      reject_explicitly_disabled_kinematic(domain.params);
    }
    if (options.interface_kinematic.interface_marker >= 0 &&
        options.interface_kinematic.interface_marker !=
            options.velocity.material_interface_marker) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Material-interface velocity and kinematic markers must match.");
    }
    options.interface_kinematic.enabled = true;
    options.interface_kinematic.interface_marker =
        options.velocity.material_interface_marker;
    if (!options.conservative_phase.enabled ||
        !options.conservative_phase.reconcile_geometry ||
        options.transport_form != ls::LevelSetTransportForm::Advective ||
        options.velocity.auto_register_field ||
        !options.velocity.algebraic_extension_source_field_name.empty() ||
        !options.boundaries.inflow.empty() ||
        !options.boundaries.outflow.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Material-interface phase-pair velocity requires advective-form conservative phase transport with geometry reconciliation, interface kinematic enforcement, no separate velocity registration or extension, and no declared inflow or outflow boundary.");
    }
  }

  if (options.conservative_phase.enabled &&
      options.conservative_phase
          .pointwise_impermeable_velocity_tolerance_explicitly_requested) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative_phase_impermeable_normal_velocity_tolerance requests a pointwise velocity-normal wall contract that the current conservative-phase split does not implement; the configured boundary policy enforces only discrete q-flux at invariant_tolerance and can be blind where q=0.");
  }

  parsing::LevelSetConfigurationReader wet_reader(
      input.equation_params, parsing::LevelSetInputPolicy::Installation);
  const auto wet_enabled = wet_reader.boolean(
      level_set_aliases::wet_extension_enable,
      "use_wet_extension_advection_velocity");
  const auto extension_method = wet_reader.string(
      level_set_aliases::advection_velocity_extension_method,
      "advection_velocity_extension_method");
  const bool pde_velocity_extension =
      extension_method.has_value() &&
      application::core::pdeVelocityExtensionOperatorFromToken(
          extension_method->text)
          .has_value();
  // The algebraic wall-compatible wet extension was retired (decisions D47
  // and D55): the velocity extension needs an explicit PDE method.  Decks may
  // still state Use_wet_extension_advection_velocity and
  // Write_velocity_extension_maps as false.
  if (wet_enabled.value_or(false) ||
      get_defined_bool(input.equation_params,
                       {"Write_velocity_extension_maps"})
          .value_or(false) ||
      ((extension_method.has_value() ||
        wet_reader
            .selected(level_set_aliases::wet_extension_source,
                      "advection_velocity_from_field")
            .has_value()) &&
       !pde_velocity_extension)) {
    throw std::runtime_error(
        std::string(application::core::kRetiredWetExtensionMessage));
  }
  std::optional<parsing::LevelSetSelectedParameter> wet_source;
  if (pde_velocity_extension) {
    // The PDE extension is recomputed from the physical velocity at every
    // geometry refresh.  Its coupling is explicit: "prescribed" writes it into
    // a prescribed advection field (the transport sees it lagged by one outer
    // pass); "monolithic" installs the same discrete problem as frozen rows of
    // an algebraic extension unknown, so Newton sees dR_phi/du.
    if (options.velocity.source !=
        ls::LevelSetVelocitySource::PrescribedData) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] The PDE velocity extension requires "
          "Velocity_source=prescribed_data.");
    }
    wet_source = wet_reader.string(level_set_aliases::wet_extension_source,
                                   "advection_velocity_from_field");
    const auto coupling = wet_reader.string(
        level_set_aliases::advection_velocity_extension_coupling,
        "advection_velocity_extension_coupling");
    const auto coupling_token =
        coupling.has_value() ? normalized_token(coupling->text) : std::string{};
    if (coupling_token == "monolithic") {
      options.velocity.algebraic_extension_source_field_name =
          wet_source.has_value() ? trim_copy(wet_source->text)
                                 : std::string{"Velocity"};
      if (options.velocity.algebraic_extension_source_field_name.empty() ||
          options.velocity.algebraic_extension_source_field_name ==
              options.velocity.field_name) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The PDE velocity extension source "
            "must name a distinct physical velocity field.");
      }
      options.velocity.source = ls::LevelSetVelocitySource::CoupledField;
      append_installation_derived(input_observations, "velocity_source",
                                  "derived:pde_velocity_extension_monolithic",
                                  "equation");
    } else if (coupling_token != "prescribed") {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Advection_velocity_extension_method=" +
          trim_copy(extension_method->text) +
          " requires Advection_velocity_extension_coupling=monolithic or "
          "prescribed" +
          (coupling.has_value() ? " (got '" + trim_copy(coupling->text) + "')"
                                : std::string{}) +
          ".");
    }
    options.velocity.auto_register_field = true;
    append_installation_selections(input_observations, wet_reader, "equation");
    append_installation_derived(input_observations,
                                "velocity_auto_register_field",
                                "derived:pde_velocity_extension", "equation");
  } else {
    append_installation_selections(input_observations, wet_reader, "equation");
  }

  if (!has_installation_observation(input_observations,
                                    "velocity_auto_register_field")) {
    append_installation_default(input_observations,
                                "velocity_auto_register_field");
  }
  if (!has_installation_observation(input_observations,
                                    "velocity_source_field_name")) {
    append_installation_default(input_observations,
                                "velocity_source_field_name");
  }

  if (!material_interface_velocity) {
    options.velocity.space = velocity_space;
    append_installation_derived(input_observations, "velocity_space_present",
                                "derived:resolved_velocity_space");
  } else {
    append_installation_derived(input_observations, "velocity_space_present",
                                "derived:material_interface_phase_pair");
  }
  reject_retired_curvature_field(input);
  return application::core::ResolvedLevelSetEquationConfiguration{
      .source_mesh = input.mesh,
      .source_mesh_name = input.mesh_name,
      .level_set_space = std::move(level_set_space),
      .options = std::move(options),
      .install_options = install_options,
      .input_observations = std::move(input_observations),
  };
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_level_set_transport_from_configuration(
    const application::core::ResolvedLevelSetEquationConfiguration&
        configuration,
    svmp::FE::systems::FESystem& system)
{

  auto module = std::make_unique<LevelSetTransportInputAdapter>(
      configuration.level_set_space,
      configuration.options,
      configuration.install_options);
  module->registerOn(system);
  return module;
}

[[nodiscard]] svmp::FE::systems::FieldSourceKind
translatedFieldSource(ls::LevelSetFieldSource source) noexcept
{
  return source == ls::LevelSetFieldSource::Unknown
             ? svmp::FE::systems::FieldSourceKind::Unknown
             : svmp::FE::systems::FieldSourceKind::PrescribedData;
}

[[nodiscard]] bool compatibleScalarTransportSpace(
    const svmp::FE::spaces::FunctionSpace& lhs,
    const svmp::FE::spaces::FunctionSpace& rhs) noexcept
{
  return lhs.space_type() == rhs.space_type() &&
         lhs.field_type() == rhs.field_type() &&
         lhs.value_dimension() == rhs.value_dimension() &&
         lhs.topological_dimension() == rhs.topological_dimension() &&
         lhs.polynomial_order() == rhs.polynomial_order() &&
         lhs.element_type() == rhs.element_type();
}

void preflightScalarTransportField(
    const svmp::FE::systems::FESystem& system,
    const ls::LevelSetFieldOptions& requested,
    const std::shared_ptr<const svmp::FE::spaces::FunctionSpace>& space,
    std::string_view role)
{
  if (requested.field_name.empty() || !space ||
      space->value_dimension() != 1) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] material-interface " +
        std::string(role) + " field requires a named scalar space");
  }
  if (requested.source != ls::LevelSetFieldSource::Unknown) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] material-interface " +
        std::string(role) + " field must be an unknown");
  }
  const auto existing = system.findFieldByName(requested.field_name);
  if (existing == svmp::FE::INVALID_FIELD_ID) {
    if (!requested.auto_register_field) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] material-interface " +
          std::string(role) + " field '" + requested.field_name +
          "' is missing and auto-registration is disabled");
    }
    return;
  }
  const auto& record = system.fieldRecord(existing);
  if (record.source_kind != translatedFieldSource(requested.source) ||
      record.components != 1 || !record.space ||
      !compatibleScalarTransportSpace(*record.space, *space)) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] material-interface " +
        std::string(role) + " field '" + requested.field_name +
        "' is incompatible with the requested scalar unknown");
  }
}

[[nodiscard]] svmp::FE::FieldId ensureScalarTransportField(
    svmp::FE::systems::FESystem& system,
    const ls::LevelSetFieldOptions& requested,
    const std::shared_ptr<const svmp::FE::spaces::FunctionSpace>& space)
{
  const auto existing = system.findFieldByName(requested.field_name);
  if (existing != svmp::FE::INVALID_FIELD_ID) {
    return existing;
  }
  return system.addField(svmp::FE::systems::FieldSpec{
      .name = requested.field_name,
      .space = space,
      .components = 1,
      .source_kind = translatedFieldSource(requested.source),
  });
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

std::shared_ptr<
    const application::core::ResolvedLevelSetEquationConfiguration>
resolveConfiguration(const svmp::Physics::EquationModuleInput& input)
{
  auto snapshot =
      std::make_shared<application::core::LevelSetEquationInputSnapshot>();
  snapshot->installation_input = input;
  return resolveConfiguration(std::move(snapshot));
}

application::core::ResolvedLevelSetEquationHandle
resolveConfiguration(application::core::LevelSetEquationInputHandle input)
{
  if (!input) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] Cannot resolve a null level-set input snapshot.");
  }
  auto configuration =
      translate_level_set_transport_input(input->installation_input);
  configuration.input_snapshot = std::move(input);
  return std::make_shared<
      application::core::ResolvedLevelSetEquationConfiguration>(
      std::move(configuration));
}

std::optional<MaterialInterfaceTransportDependency>
materialInterfaceTransportDependency(
    const application::core::ResolvedLevelSetEquationConfiguration&
        configuration)
{
  const auto& options = configuration.options;
  if (options.velocity.source !=
      ls::LevelSetVelocitySource::MaterialInterfacePhasePair) {
    return std::nullopt;
  }
  if (!configuration.source_mesh || options.operator_tag.empty() ||
      options.level_set.field_name.empty() ||
      options.conservative_phase.liquid_indicator.field_name.empty()) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] incomplete material-interface transport dependency");
  }
  return MaterialInterfaceTransportDependency{
      .mesh = configuration.source_mesh.get(),
      .mesh_name = configuration.source_mesh_name,
      .dimension = configuration.source_mesh->dim(),
      .interface_marker = options.velocity.material_interface_marker,
      .level_set_field_name = options.level_set.field_name,
      .conservative_phase_field_name =
          options.conservative_phase.liquid_indicator.field_name,
      .operator_tag = options.operator_tag,
  };
}

void preRegisterMaterialInterfaceTransportFields(
    const application::core::ResolvedLevelSetEquationConfiguration&
        configuration,
    svmp::FE::systems::FESystem& system)
{
  const auto& options = configuration.options;
  if (options.velocity.source !=
      ls::LevelSetVelocitySource::MaterialInterfacePhasePair) {
    throw std::invalid_argument(
        "preRegisterMaterialInterfaceTransportFields requires material-interface phase-pair velocity");
  }
  if (options.level_set.field_name ==
      options.conservative_phase.liquid_indicator.field_name) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] material-interface level-set and phase field names must be distinct");
  }

  preflightScalarTransportField(
      system,
      options.level_set,
      configuration.level_set_space,
      "level-set");
  preflightScalarTransportField(
      system,
      options.conservative_phase.liquid_indicator,
      configuration.level_set_space,
      "conservative-phase");

  (void)ensureScalarTransportField(
      system, options.level_set, configuration.level_set_space);
  (void)ensureScalarTransportField(
      system,
      options.conservative_phase.liquid_indicator,
      configuration.level_set_space);
}

std::unique_ptr<svmp::Physics::PhysicsModule>
createModule(
    const application::core::ResolvedLevelSetEquationConfiguration&
        configuration,
    svmp::FE::systems::FESystem& system)
{
  return create_level_set_transport_from_configuration(
      configuration, system);
}

std::optional<MaterialInterfaceTransportDependency>
materialInterfaceTransportDependency(
    const svmp::Physics::EquationModuleInput& input)
{
  const auto configuration = resolveConfiguration(input);
  return materialInterfaceTransportDependency(*configuration);
}

void preRegisterMaterialInterfaceTransportFields(
    const svmp::Physics::EquationModuleInput& input,
    svmp::FE::systems::FESystem& system)
{
  const auto configuration = resolveConfiguration(input);
  preRegisterMaterialInterfaceTransportFields(*configuration, system);
}

std::unique_ptr<svmp::Physics::PhysicsModule>
createModule(const svmp::Physics::EquationModuleInput& input,
             svmp::FE::systems::FESystem& system)
{
  const auto configuration = resolveConfiguration(input);
  return createModule(*configuration, system);
}

} // namespace level_set
} // namespace translators
} // namespace application

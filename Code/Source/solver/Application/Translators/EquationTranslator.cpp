#include "Application/Translators/EquationTranslator.h"

#include "Application/Core/OopMpiLog.h"
#include "Application/Translators/LevelSetEquationTranslator.h"
#include "Mesh/Core/MeshBase.h"
#include "Parameters.h"
#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

template <typename T>
std::string snapshot_value_string(const Parameter<T>& p)
{
  if constexpr (std::is_floating_point_v<T>) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<T>::max_digits10) << p.value();
    return out.str();
  } else {
    return p.svalue();
  }
}

template <typename T>
std::string snapshot_value_string(const VectorParameter<T>& p)
{
  if constexpr (std::is_floating_point_v<T>) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<T>::max_digits10);
    for (const auto v : p.value()) {
      out << ' ' << v;
    }
    return out.str();
  } else {
    return p.svalue();
  }
}

svmp::Physics::ParameterMap snapshot_params(const ParameterLists& list)
{
  svmp::Physics::ParameterMap out;
  for (const auto& [_, v] : list.params_map) {
    std::visit(
        [&](const auto* p) {
          if (!p) {
            return;
          }
          out[p->name()] = svmp::Physics::ParameterValue{p->defined(), snapshot_value_string(*p)};
        },
        v);
  }
  return out;
}

svmp::Physics::ParameterMap snapshot_legacy_params(
    const ParameterLists& list)
{
  svmp::Physics::ParameterMap out;
  for (const auto& [_, value] : list.params_map) {
    std::visit(
        [&](const auto* parameter) {
          if (parameter == nullptr) {
            return;
          }
          out[parameter->name()] = svmp::Physics::ParameterValue{
              parameter->defined(), parameter->svalue()};
        },
        value);
  }
  return out;
}

std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string normalized_token(std::string raw)
{
  raw = trim_copy(std::move(raw));
  std::transform(raw.begin(), raw.end(), raw.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  raw.erase(std::remove_if(raw.begin(), raw.end(), [](unsigned char ch) {
              return ch == '_' || ch == '-' || std::isspace(ch);
            }),
            raw.end());
  return raw;
}

std::string defined_param_token(const svmp::Physics::ParameterMap& params,
                                std::string_view key)
{
  const auto it = params.find(std::string(key));
  if (it == params.end() || !it->second.defined) {
    return {};
  }
  return normalized_token(it->second.value);
}

bool is_unfitted_free_surface_bc(const svmp::Physics::ParameterMap& params)
{
  if (defined_param_token(params, "Type") != "freesurface") {
    return false;
  }
  const auto implementation = defined_param_token(params, "Implementation");
  return implementation == "unfitted" ||
         implementation == "unfittedlevelset" ||
         implementation == "levelset" ||
         implementation == "embeddedlevelset";
}

const svmp::Physics::ParameterValue* first_defined_param(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    const auto it = params.find(std::string(key));
    if (it != params.end() && it->second.defined &&
        !trim_copy(it->second.value).empty()) {
      return &it->second;
    }
  }
  return nullptr;
}

std::vector<std::string> split_face_entries(std::string_view raw)
{
  std::vector<std::string> entries;
  std::string current;
  for (const char ch : raw) {
    if (ch == ';' || ch == ',') {
      auto entry = trim_copy(std::move(current));
      if (!entry.empty()) {
        entries.push_back(std::move(entry));
      }
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  auto entry = trim_copy(std::move(current));
  if (!entry.empty()) {
    entries.push_back(std::move(entry));
  }
  return entries;
}

void resolve_contact_line_wall_faces(svmp::Physics::ParameterMap& params,
                                     const svmp::Mesh& mesh)
{
  const auto* faces = first_defined_param(
      params,
      {"Contact_line_wall_faces", "ContactLineWallFaces",
       "Contact_line_wall_face", "ContactLineWallFace",
       "Wall_boundary_faces", "WallBoundaryFaces",
       "Wall_boundary_face", "WallBoundaryFace"});
  if (faces == nullptr) {
    return;
  }
  if (first_defined_param(
          params,
          {"Contact_line_wall_markers", "ContactLineWallMarkers",
           "Contact_line_wall_marker", "ContactLineWallMarker",
           "Wall_boundary_markers", "WallBoundaryMarkers",
           "Wall_boundary_marker", "WallBoundaryMarker"}) != nullptr) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface contact-line wall input must use either wall face names or wall markers, not both.");
  }

  const auto entries = split_face_entries(faces->value);
  if (entries.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Contact_line_wall_face(s) is defined but empty.");
  }

  std::ostringstream markers;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto label = mesh.label_from_name(entries[i]);
    if (label == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Contact-line wall face '" +
          entries[i] +
          "' is not registered in the mesh labels.");
    }
    if (i != 0u) {
      markers << ';';
    }
    markers << static_cast<int>(label);
  }
  params["Contact_line_wall_markers"] =
      svmp::Physics::ParameterValue{true, markers.str()};

  // Face names are an application-level convenience.  The physics module
  // intentionally accepts only resolved marker ids, so remove every face
  // alias after translating it instead of forwarding both representations.
  for (const std::string_view key : {
           "Contact_line_wall_faces", "ContactLineWallFaces",
           "Contact_line_wall_face", "ContactLineWallFace",
           "Wall_boundary_faces", "WallBoundaryFaces",
           "Wall_boundary_face", "WallBoundaryFace"}) {
    params.erase(std::string(key));
  }
}

svmp::Physics::DomainInput snapshot_domain(const DomainParameters& domain)
{
  svmp::Physics::DomainInput out{};
  out.id = domain.id.value();
  out.params = snapshot_params(domain);

  if (domain.constitutive_model.defined()) {
    out.nested_configuration_blocks.push_back(
        ConstitutiveModelParameters::xml_element_name_);
  }
  if (domain.fiber_reinforcement_stress.defined()) {
    out.nested_configuration_blocks.push_back(
        FiberReinforcementStressParameters::xml_element_name_);
  }
  if (domain.stimulus.defined()) {
    out.nested_configuration_blocks.push_back(
        StimulusParameters::xml_element_name_);
  }

  if (domain.fluid_viscosity.model.defined()) {
    out.params["Viscosity.model"] =
        svmp::Physics::ParameterValue{domain.fluid_viscosity.model.defined(), domain.fluid_viscosity.model.value()};

    const auto append = [&](const ParameterLists& list, const std::string& prefix) {
      const auto block = snapshot_params(list);
      for (const auto& [k, v] : block) {
        out.params[prefix + k] = v;
      }
    };

    append(domain.fluid_viscosity.newtonian_model, "Viscosity.");
    append(domain.fluid_viscosity.carreau_yasuda_model, "Viscosity.");
    append(domain.fluid_viscosity.cassons_model, "Viscosity.");
  }

  if (domain.constitutive_model.defined() && domain.constitutive_model.type.defined()) {
    out.params["Constitutive_model.type"] =
        svmp::Physics::ParameterValue{domain.constitutive_model.type.defined(),
                                      domain.constitutive_model.type.value()};
  }

  return out;
}

} // namespace

namespace application {
namespace translators {

svmp::Physics::EquationModuleInput EquationTranslator::buildInput(
    const EquationParameters& eq_params,
    const std::map<std::string, std::shared_ptr<svmp::Mesh>>& meshes)
{
  const std::string eq_type = eq_params.type.value();

  application::core::oopCout() << "[svMultiPhysics::Application] EquationTranslator: buildInput(type='" << eq_type
                               << "')" << std::endl;

  if (meshes.empty()) {
    throw std::runtime_error("[svMultiPhysics::Application] No meshes are available for equation translation.");
  }

  if (meshes.size() != 1) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Multiple <Add_mesh> blocks are not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto mesh_name = meshes.begin()->first;
  auto mesh = meshes.begin()->second;
  if (!mesh) {
    throw std::runtime_error("[svMultiPhysics::Application] Null mesh encountered during equation translation.");
  }

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = eq_type;
  input.equation_params = snapshot_params(eq_params);
  input.mesh_name = mesh_name;
  input.mesh = mesh->local_mesh_ptr();

  const auto append_nested_equation_block =
      [&](bool defined, const std::string& name) {
        if (defined) {
          input.nested_configuration_blocks.push_back(name);
        }
      };
  append_nested_equation_block(
      eq_params.couple_to_cplBC.defined(),
      CoupleCplBCParameters::xml_element_name_);
  append_nested_equation_block(
      eq_params.couple_to_genBC.defined(),
      CoupleGenBCParameters::xml_element_name_);
  append_nested_equation_block(
      eq_params.svzerodsolver_interface_parameters.defined(),
      svZeroDSolverInterfaceParameters::xml_element_name_);
  append_nested_equation_block(
      eq_params.remesher.defined(),
      RemesherParameters::xml_element_name_);
  append_nested_equation_block(
      eq_params.variable_wall_properties.defined(),
      VariableWallPropsParameters::xml_element_name_);
  append_nested_equation_block(
      eq_params.ecg_leads.defined(),
      ECGLeadsParameters::xml_element_name_);
  input.body_force_block_count = static_cast<std::size_t>(std::count_if(
      eq_params.body_forces.begin(),
      eq_params.body_forces.end(),
      [](const auto* body_force) { return body_force != nullptr; }));

  application::core::oopCout() << "[svMultiPhysics::Application] EquationTranslator: mesh='" << mesh_name << "'"
                               << " domains=" << static_cast<int>(eq_params.domains.size())
                               << " boundary_conditions=" << static_cast<int>(eq_params.boundary_conditions.size())
                               << std::endl;

  // Generic module-specific options hook (added to EquationParameters once).
  // If unset, these remain empty and the formulation uses its defaults.
  if (eq_params.module_options.defined()) {
    input.module_options = eq_params.module_options.value();
    application::core::oopCout() << "[svMultiPhysics::Application] EquationTranslator: module_options='"
                                 << input.module_options << "'" << std::endl;
  }
  if (eq_params.module_options_file_path.defined()) {
    input.module_options_file_path = eq_params.module_options_file_path.value();
    application::core::oopCout() << "[svMultiPhysics::Application] EquationTranslator: module_options_file_path='"
                                 << input.module_options_file_path << "'" << std::endl;
  }

  if (eq_params.node_pressure_constraints.value_set) {
    const auto id_type = eq_params.node_pressure_constraints.id_type.value();
    const auto values_file_path = eq_params.node_pressure_constraints.values_file_path.value();
    if (id_type != "Global_vertex_gid") {
      throw std::runtime_error(
          "[svMultiPhysics::Application] <Node_pressure_constraints><Id_type> must be 'Global_vertex_gid' "
          "for the new OOP solver Darcy/Poisson/Navier-Stokes node pressure constraint path.");
    }
    if (values_file_path.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] <Node_pressure_constraints><Values_file_path> must be non-empty.");
    }

    svmp::Physics::NodePressureConstraintInput node_pressure_constraints;
    node_pressure_constraints.id_type = id_type;
    node_pressure_constraints.values_file_path = values_file_path;
    input.node_pressure_constraints = std::move(node_pressure_constraints);
    application::core::oopCout()
        << "[svMultiPhysics::Application] EquationTranslator: node_pressure_constraints id_type='"
        << id_type << "' values_file_path='" << values_file_path << "'" << std::endl;
  }

  if (eq_params.default_domain) {
    input.default_domain = snapshot_domain(*eq_params.default_domain);
  }

  if (!eq_params.domains.empty()) {
    input.domains.reserve(eq_params.domains.size());
    for (const auto* d : eq_params.domains) {
      if (!d) {
        continue;
      }
      input.domains.push_back(snapshot_domain(*d));
    }
  }

  if (!eq_params.outputs.empty()) {
    input.outputs.reserve(eq_params.outputs.size());
    for (const auto* output : eq_params.outputs) {
      if (!output) {
        continue;
      }

      svmp::Physics::OutputRequestInput output_in{};
      output_in.type = output->type.value();
      for (const auto& param : output->output_list) {
        output_in.params[param.name()] =
            svmp::Physics::ParameterValue{param.defined(), snapshot_value_string(param)};
      }
      input.outputs.push_back(std::move(output_in));
    }
  }

  if (!eq_params.boundary_conditions.empty()) {
    input.boundary_conditions.reserve(eq_params.boundary_conditions.size());
    for (const auto* bc : eq_params.boundary_conditions) {
      if (!bc) {
        continue;
      }

      svmp::Physics::BoundaryConditionInput bc_in{};
      bc_in.name = bc->name.value();
      bc_in.boundary_marker = mesh->label_from_name(bc_in.name);
      bc_in.params = snapshot_params(*bc);
      if (is_unfitted_free_surface_bc(bc_in.params)) {
        resolve_contact_line_wall_faces(bc_in.params, *mesh);
      }
      application::core::oopCout() << "[svMultiPhysics::Application]   BC '" << bc_in.name
                                   << "': boundary_marker=" << bc_in.boundary_marker << std::endl;
      if (bc_in.boundary_marker == svmp::INVALID_LABEL &&
          !is_unfitted_free_surface_bc(bc_in.params)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Boundary condition references face '" + bc_in.name +
            "', but that face is not registered. Ensure <Add_face name=\"" + bc_in.name +
            "\"> exists under the mesh and <Add_BC name=\"" + bc_in.name +
            "\"> references it, or set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }

      if (bc->rcr.value_set) {
        bc_in.nested_configuration_blocks.push_back(
            BoundaryConditionRCRParameters::xml_element_name_);
        const auto rcr = snapshot_params(bc->rcr);
        for (const auto& [k, v] : rcr) {
          bc_in.params["RCR." + k] = v;
        }
      }
      if (bc->rcrcr.value_set) {
        bc_in.nested_configuration_blocks.push_back(
            BoundaryConditionRCRCRParameters::xml_element_name_);
        const auto rcrcr = snapshot_params(bc->rcrcr);
        for (const auto& [k, v] : rcrcr) {
          bc_in.params["RCRCR." + k] = v;
        }
      }
      input.boundary_conditions.push_back(std::move(bc_in));
    }
  }

  return input;
}

application::core::LegacyLevelSetMaintenanceInput
EquationTranslator::snapshotLegacyLevelSetMaintenanceInput(
    const EquationParameters& equation)
{
  application::core::LegacyLevelSetMaintenanceInput input{};
  input.equation_type_defined = equation.type.defined();
  input.equation_type = equation.type.value();
  input.equation_parameters = snapshot_legacy_params(equation);
  input.boundaries.reserve(equation.boundary_conditions.size());
  for (const auto* boundary : equation.boundary_conditions) {
    if (boundary == nullptr) {
      continue;
    }
    input.boundaries.push_back(
        application::core::LegacyLevelSetBoundaryInput{
            .type_defined = boundary->type.defined(),
            .name_defined = boundary->name.defined(),
            .type = boundary->type.value(),
            .name = boundary->name.value(),
            .parameters = snapshot_legacy_params(*boundary),
        });
  }
  return input;
}

application::core::LevelSetEquationInputHandle
EquationTranslator::snapshotLevelSetEquationInput(
    const EquationParameters& equation,
    svmp::Physics::EquationModuleInput installation_input)
{
  auto snapshot =
      std::make_shared<application::core::LevelSetEquationInputSnapshot>();
  snapshot->installation_input = std::move(installation_input);
  snapshot->legacy_maintenance_input =
      snapshotLegacyLevelSetMaintenanceInput(equation);
  return snapshot;
}

std::unique_ptr<svmp::Physics::PhysicsModule> EquationTranslator::createModule(
    const EquationParameters& eq_params, svmp::FE::systems::FESystem& system,
    const std::map<std::string, std::shared_ptr<svmp::Mesh>>& meshes)
{
  const auto input = buildInput(eq_params, meshes);
  const std::string eq_type = input.equation_type;

  application::core::oopCout() << "[svMultiPhysics::Application] EquationTranslator: createModule(type='" << eq_type
                               << "')" << std::endl;

  if (level_set::isEquationType(eq_type)) {
    return level_set::createModule(input, system);
  }

  auto& registry = svmp::Physics::EquationModuleRegistry::instance();
  auto types = registry.registeredTypes();
  for (auto type : level_set::equationTypes()) {
    if (std::find(types.begin(), types.end(), type) == types.end()) {
      types.push_back(std::move(type));
    }
  }
  const auto supported = std::find(types.begin(), types.end(), eq_type) != types.end();
  if (!supported) {
    std::string supported_list = types.empty() ? "(none)" : types.front();
    for (std::size_t i = 1; i < types.size(); ++i) {
      supported_list += ", " + types[i];
    }

    throw std::runtime_error(
        "[svMultiPhysics::Application] Equation type '" + eq_type +
        "' is not registered for the new OOP solver. Registered types: " + supported_list +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  return registry.create(eq_type, input, system);
}

} // namespace translators
} // namespace application

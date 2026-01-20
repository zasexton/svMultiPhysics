#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
class MeshBase;
}

namespace svmp::Physics {

struct ParameterValue {
  bool defined{false};
  std::string value{};
};

using ParameterMap = std::map<std::string, ParameterValue>;

struct BoundaryConditionInput {
  std::string name{};
  int boundary_marker{-1};
  ParameterMap params{};
};

struct DomainInput {
  std::string id{};
  ParameterMap params{};
};

struct EquationModuleInput {
  std::string equation_type{};

  ParameterMap equation_params{};

  std::string module_options{};
  std::string module_options_file_path{};

  std::string mesh_name{};
  std::shared_ptr<svmp::MeshBase> mesh{};

  DomainInput default_domain{};
  std::vector<DomainInput> domains{};

  std::vector<BoundaryConditionInput> boundary_conditions{};
};

} // namespace svmp::Physics

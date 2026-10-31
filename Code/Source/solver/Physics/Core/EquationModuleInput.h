#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
class MeshBase;
}

namespace svmp::Physics {

struct ParameterValue {
  bool defined{false};
  std::string value{};
  /// The method profile that stated the value on behalf of the input (for
  /// example "Capillary@1" for a key of a free-surface protocol); empty when
  /// the input gave it.  Consumers resolve the value like any other and
  /// record the profile as its source.
  std::string origin{};
};

using ParameterMap = std::map<std::string, ParameterValue>;

struct BoundaryConditionInput {
  std::string name{};
  int boundary_marker{-1};
  ParameterMap params{};
  std::vector<std::string> nested_configuration_blocks{};
};

struct DomainInput {
  std::string id{};
  ParameterMap params{};
  std::vector<std::string> nested_configuration_blocks{};
};

struct NodePressureConstraintInput {
  std::string id_type{};
  std::string values_file_path{};
};

struct OutputRequestInput {
  std::string type{};
  ParameterMap params{};
};

/// Requests that the Application derives from the other equations of the
/// same input.  They are never input keys.
struct DerivedModuleRequests {
  /// Install the conservative-balance diagnostic operators of the flow
  /// module: a level-set equation enables the static-capillary
  /// initialization, whose startup certificate evaluates them.
  bool conservative_balance_diagnostics{false};
};

struct EquationModuleInput {
  std::string equation_type{};

  DerivedModuleRequests derived_requests{};

  ParameterMap equation_params{};
  std::vector<std::string> nested_configuration_blocks{};

  std::string module_options{};

  std::string mesh_name{};
  std::shared_ptr<svmp::MeshBase> mesh{};

  DomainInput default_domain{};
  std::vector<DomainInput> domains{};

  std::vector<BoundaryConditionInput> boundary_conditions{};

  std::size_t body_force_block_count{0u};
  std::vector<OutputRequestInput> outputs{};

  std::optional<NodePressureConstraintInput> node_pressure_constraints{};
};

} // namespace svmp::Physics

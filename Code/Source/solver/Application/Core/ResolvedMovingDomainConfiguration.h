#pragma once

#include "Application/Core/LevelSetEquationInputSnapshot.h"
#include "FE/LevelSet/LevelSetOptions.h"
#include "FE/Systems/FormsInstaller.h"

#include <memory>
#include <string>
#include <vector>

namespace svmp {
class MeshBase;

namespace FE::spaces {
class FunctionSpace;
} // namespace FE::spaces
} // namespace svmp

namespace application::core {

struct LevelSetInputObservation {
  std::string canonical_key{};
  std::string selected_spelling{};
  std::string source_layer{};
  bool supplied{false};
  std::string representation{};
  bool compatibility_fallback{false};
  std::vector<std::string> ordered_overrides{};
};

struct ResolvedLevelSetEquationConfiguration {
  std::shared_ptr<svmp::MeshBase> source_mesh{};
  std::string source_mesh_name{};
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> level_set_space{};
  svmp::FE::level_set::LevelSetTransportOptions options{};
  svmp::FE::systems::FormInstallOptions install_options{};
  std::vector<std::string> projected_curvature_fields{};
  std::vector<LevelSetInputObservation> input_observations{};
  LevelSetEquationInputHandle input_snapshot{};
};

using ResolvedLevelSetEquationHandle =
    std::shared_ptr<const ResolvedLevelSetEquationConfiguration>;

} // namespace application::core

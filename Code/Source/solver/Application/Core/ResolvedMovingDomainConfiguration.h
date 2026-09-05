#pragma once

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

struct ResolvedLevelSetEquationConfiguration {
  std::shared_ptr<svmp::MeshBase> source_mesh{};
  std::string source_mesh_name{};
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> level_set_space{};
  svmp::FE::level_set::LevelSetTransportOptions options{};
  svmp::FE::systems::FormInstallOptions install_options{};
  std::vector<std::string> projected_curvature_fields{};
};

using ResolvedLevelSetEquationHandle =
    std::shared_ptr<const ResolvedLevelSetEquationConfiguration>;

} // namespace application::core

#pragma once

#include "Application/Core/LevelSetEquationInputSnapshot.h"
#include "FE/LevelSet/LevelSetOptions.h"
#include "FE/Systems/FormsInstaller.h"

#include <memory>
#include <string>

namespace svmp {
class MeshBase;

namespace FE::spaces {
class FunctionSpace;
} // namespace FE::spaces
} // namespace svmp

namespace application::core {

// The keys of a level-set equation as one value, as levelSetEquationSchema()
// (Application/Translators/LevelSetConfigurationParsing.h) resolves them
// (decision D47, step P3): the transport options and the keys that select a
// transport value without being one.
struct LevelSetEquationKeys {
  svmp::FE::level_set::LevelSetTransportOptions transport{};
  // Coupled: the transport joins the monolithic operator ("equations").
  bool coupled{false};
  // Level_set_isovalue: the interface of the maintained field.
  double level_set_isovalue{0.0};
};

struct ResolvedLevelSetEquationConfiguration {
  std::shared_ptr<svmp::MeshBase> source_mesh{};
  std::string source_mesh_name{};
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> level_set_space{};
  svmp::FE::level_set::LevelSetTransportOptions options{};
  svmp::FE::systems::FormInstallOptions install_options{};
  // The keys of the equation element alone, before its domain elements and
  // the extension: what the maintenance of the field reads.
  LevelSetEquationKeys equation_keys{};
  // The "resolved" section of the transport's effective-configuration
  // artifact: every key of the schema with its value and source.
  std::string resolved_input_json{};
  LevelSetEquationInputHandle input_snapshot{};
};

using ResolvedLevelSetEquationHandle =
    std::shared_ptr<const ResolvedLevelSetEquationConfiguration>;

} // namespace application::core

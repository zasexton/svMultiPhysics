#pragma once

#include <map>
#include <memory>
#include <string>

#include "Application/Core/LevelSetEquationInputSnapshot.h"
#include "Mesh/Mesh.h"
#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/PhysicsModule.h"

class EquationParameters;
class Parameters;

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
}
namespace spaces {
class FunctionSpace;
}
} // namespace FE
} // namespace svmp

namespace application {
namespace translators {

class EquationTranslator {
public:
  static svmp::Physics::EquationModuleInput buildInput(
      const EquationParameters& eq_params,
      const std::map<std::string, std::shared_ptr<svmp::Mesh>>& meshes);

  static application::core::LegacyLevelSetMaintenanceInput
  snapshotLegacyLevelSetMaintenanceInput(
      const EquationParameters& equation);

  static application::core::LevelSetEquationInputHandle
  snapshotLevelSetEquationInput(
      const EquationParameters& equation,
      svmp::Physics::EquationModuleInput installation_input);

  /// States the values of the method profile that a Free_surface boundary
  /// condition names (Free_surface_profile, decision D47 P4) on that
  /// boundary and on the level-set equation of its field, for every key the
  /// element does not state itself; each such parameter records the profile
  /// as its origin.  Every reader of the input then sees the deck the
  /// profile stands for.  Called once, right after the input file is read.
  static void applyFreeSurfaceProfiles(Parameters& params);

  /// `derived_requests` carries what the Application derived from the other
  /// equations of the input (see svmp::Physics::DerivedModuleRequests).
  static std::unique_ptr<svmp::Physics::PhysicsModule> createModule(
      const EquationParameters& eq_params, svmp::FE::systems::FESystem& system,
      const std::map<std::string, std::shared_ptr<svmp::Mesh>>& meshes,
      const svmp::Physics::DerivedModuleRequests& derived_requests = {});
};

} // namespace translators
} // namespace application

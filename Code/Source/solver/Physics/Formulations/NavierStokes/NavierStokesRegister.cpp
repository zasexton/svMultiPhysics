// The equation factory of the incompressible Navier-Stokes equations (types
// "fluid" and "stokes").  Physics/Options resolves the input: the physical
// model, the option schema and readers, the boundary conditions and the
// two-fluid translation (decisions D47, D50 and D52).
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Options/NavierStokesRegistration.h"

#include "FE/Systems/FESystem.h"

#include <memory>
#include <utility>

namespace {

// resolveNavierStokesInput (Physics/Options, decision D52).
using namespace svmp::Physics::options;

// The equation factory: resolve the input through Physics/Options, then
// construct the module and register it (decision D50).  A two-fluid input
// constructs the module in its two-fluid mode.
std::unique_ptr<svmp::Physics::PhysicsModule>
create_navier_stokes_from_input(const svmp::Physics::EquationModuleInput& input,
                                svmp::FE::systems::FESystem& system)
{
  using Module = svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSModule;
  auto resolved = resolveNavierStokesInput(input, system);
  if (resolved.two_fluid) {
    auto& translated = *resolved.two_fluid;
    auto module = std::make_unique<Module>(
        translated.velocity_space,
        translated.pressure_space,
        translated.velocity_space,
        translated.pressure_space,
        std::move(translated.options));
    module->registerOn(system);
    return module;
  }
  auto& one_phase = *resolved.one_phase;
  auto module = std::make_unique<Module>(
      std::move(one_phase.velocity_space), std::move(one_phase.pressure_space),
      std::move(one_phase.options));
  module->registerOn(system);
  return module;
}

} // namespace

SVMP_REGISTER_EQUATION("fluid", &create_navier_stokes_from_input);
SVMP_REGISTER_EQUATION("stokes", &create_navier_stokes_from_input);

namespace svmp::Physics::formulations::navier_stokes {

void forceLink_NavierStokesRegister() {}

} // namespace svmp::Physics::formulations::navier_stokes

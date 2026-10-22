#pragma once

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyAccount.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

class Parameters;

namespace application {
namespace core {

struct SimulationComponents;
struct VtkTimeSeriesCollection;

// The level-set maintenance work ledger and the accepted-step energy account
// are Navier-Stokes free-surface physics
// (Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyAccount.h).
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceWorkStatus;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceWorkSubstage;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceDeclaredStage;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetAuthoritativeFunctionalValue;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceWorkTransaction;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceWorkRow;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceModeledEnergySubstage;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceModeledEnergyBreakdown;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceWorkAttempt;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceAcceptedStepEnergyAccount;
using svmp::Physics::formulations::navier_stokes::free_surface::aggregateLevelSetMaintenanceAcceptedStepEnergy;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenancePhysicalEndpointChannels;
using svmp::Physics::formulations::navier_stokes::free_surface::FreeSurfaceResidualWorkPairing;
using svmp::Physics::formulations::navier_stokes::free_surface::FreeSurfaceResidualWorkChannels;
using svmp::Physics::formulations::navier_stokes::free_surface::evaluateFreeSurfaceResidualWorkChannels;
using svmp::Physics::formulations::navier_stokes::free_surface::evaluateLevelSetMaintenancePhysicalEndpointChannels;
using svmp::Physics::formulations::navier_stokes::free_surface::LevelSetMaintenanceWorkLedger;

class ApplicationDriver {
public:
  // How far ApplicationDriver::run goes.  Full runs the simulation.
  // ResolveOnly (svMultiPhysics --resolve-only XML_FILE) builds the
  // simulation as a full run does, writes effective_configuration.json and
  // stops before the first solve: the resolved configuration of a deck can
  // be compared without running it.
  enum class RunMode {
    Full,
    ResolveOnly,
  };

  static bool shouldUseNewSolver(const std::string& xml_file);
  static void run(const std::string& xml_file, RunMode mode = RunMode::Full);

private:
  static void runWithParameters(const Parameters& params, RunMode mode);
  static void runSteadyState(SimulationComponents& sim, const Parameters& params, VtkTimeSeriesCollection* pvd);
  static void runTransient(SimulationComponents& sim, const Parameters& params, VtkTimeSeriesCollection* pvd);
  static void outputResults(const SimulationComponents& sim, const Parameters& params, int step,
                            double time, VtkTimeSeriesCollection* pvd);
};

} // namespace core
} // namespace application

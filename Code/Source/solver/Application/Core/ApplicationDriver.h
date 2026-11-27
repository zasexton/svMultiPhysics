#pragma once

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyBalance.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/LevelSetMaintenanceWorkLedger.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::timestepping { enum class SchemeKind : std::uint8_t; }

class Parameters;

namespace application {
namespace core {

struct SimulationComponents;
struct VtkTimeSeriesCollection;

// The free-surface formulation owns the maintenance energy ledger and endpoint channels.
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
                            double time, VtkTimeSeriesCollection* pvd,
                            std::optional<svmp::FE::timestepping::SchemeKind> transient_scheme = std::nullopt);
};

} // namespace core
} // namespace application

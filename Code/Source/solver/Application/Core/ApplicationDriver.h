#pragma once

#include "FE/LevelSet/LevelSetEnergyBalance.h"

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

// The level-set maintenance work ledger and the accepted-step maintenance
// account are FE services
// (FE/LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h); the physical
// endpoint channels and the residual-work channels are part of the energy
// record of a level-set run (FE/LevelSet/LevelSetEnergyBalance.h).
using svmp::FE::level_set::LevelSetMaintenanceWorkStatus;
using svmp::FE::level_set::LevelSetMaintenanceWorkSubstage;
using svmp::FE::level_set::LevelSetMaintenanceDeclaredStage;
using svmp::FE::level_set::LevelSetAuthoritativeFunctionalValue;
using svmp::FE::level_set::LevelSetMaintenanceWorkTransaction;
using svmp::FE::level_set::LevelSetMaintenanceWorkRow;
using svmp::FE::level_set::LevelSetMaintenanceModeledEnergySubstage;
using svmp::FE::level_set::LevelSetMaintenanceModeledEnergyBreakdown;
using svmp::FE::level_set::LevelSetMaintenanceWorkAttempt;
using svmp::FE::level_set::LevelSetMaintenanceAcceptedStepEnergyAccount;
using svmp::FE::level_set::aggregateLevelSetMaintenanceAcceptedStepEnergy;
using svmp::FE::level_set::LevelSetMaintenancePhysicalEndpointChannels;
using svmp::FE::level_set::FreeSurfaceResidualWorkChannels;
using svmp::FE::level_set::evaluateFreeSurfaceResidualWorkChannels;
using svmp::FE::level_set::evaluateLevelSetMaintenancePhysicalEndpointChannels;
using svmp::FE::level_set::LevelSetMaintenanceWorkLedger;

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

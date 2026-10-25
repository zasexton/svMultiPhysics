#include "Application/Core/ApplicationDriver.h"
#include "FE/Core/LogRecord.h"

#include "Application/Core/ActiveDomainOutput.h"
#include "Application/Core/FreeSurfaceSemiImplicitReference.h"
#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/LevelSetMaintenanceConfiguration.h"
#include "Application/Core/LevelSetMaintenanceTransactionConsensus.h"
#include "Application/Core/LevelSetPdeVelocityExtension.h"
#include "Application/Core/OopMpiLog.h"
#include "Application/Core/SimulationBuilder.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/CutGeometryMemoryReport.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Basis/BasisCache.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Backends/MUMPS/MumpsDistributedSolver.h"
#include "FE/Constraints/AffineConstraints.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/LevelSet/LevelSetCutTopology.h"
#include "FE/LevelSet/LevelSetContactStageParticipant.h"
#include "FE/LevelSet/LevelSetGeometryService.h"
#include "FE/LevelSet/LevelSetCellEvaluator.h"
#include "FE/LevelSet/LevelSetConservativePhaseOperator.h"
#include "FE/LevelSet/LevelSetConservativePhaseArtifact.h"
#include "FE/LevelSet/LevelSetConservativePhaseRegions.h"
#include "FE/LevelSet/LevelSetConservativePhaseState.h"
#include "FE/LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "FE/LevelSet/LevelSetMaintenanceSchedule.h"
#include "FE/LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h"
#include "FE/LevelSet/Maintenance/LevelSetMaintenancePipeline.h"
#include "FE/LevelSet/Maintenance/LevelSetMaintenanceRequestSchedule.h"
#include "FE/LevelSet/Maintenance/LevelSetWallAwareMaintenance.h"
#include "FE/LevelSet/LevelSetImplicitCutQuadratureBackend.h"
#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"
#include "FE/LevelSet/LevelSetKinematicReconciliation.h"
#include "FE/LevelSet/LevelSetSignDefinitePatchBounds.h"
#include "FE/LevelSet/LevelSetInterfacePreservingRedistancing.h"
#include "FE/LevelSet/LevelSetInterfacePreservingRedistancing3D.h"
#include "FE/LevelSet/LevelSetTransport.h"
#include "FE/LevelSet/LevelSetVelocityExtensionConstraint.h"
#include "FE/LevelSet/Extension/LevelSetVelocityExtensionRefresh.h"
#include "FE/LevelSet/LevelSetVolume.h"
#include "FE/LevelSet/LevelSetWallContactConstraint.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/Geometry/CutQuadratureMapping.h"
#include "FE/Interfaces/GeneratedActiveBoundaryDomain.h"
#include "FE/Core/DeterministicParallel.h"
#include "FE/Core/AggregationGuardDiagnostics.h"
#include "FE/Core/HaloDiagnostics.h"
#include "FE/Interfaces/ContactLineStageObservables.h"
#include "FE/Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "FE/Interfaces/IncompressibleTwoFluidDiagnostics.h"
#include "FE/Interfaces/MaterialInterfaceTransportVelocity.h"
#include "FE/Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "FE/PostProcessing/DerivedResultTypes.h"
#include "FE/PostProcessing/DerivedResultEvaluator.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/CutIntegrationInvalidation.h"
#include "FE/Systems/SystemCollectives.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/TimeStepping/FieldRepairPublication.h"
#include "FE/TimeStepping/GeneratedState.h"
#include "FE/TimeStepping/MeshNormalStageParticipant.h"
#include "FE/TimeStepping/OperatorStageCapture.h"
#include "FE/TimeStepping/NewtonSolver.h"
#include "FE/TimeStepping/StepContext.h"
#include "FE/TimeStepping/StepParticipants.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"
#include "FE/TimeStepping/TimeSteppingUtils.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Core/MeshComm.h"
#include "Mesh/Topology/CellTopology.h"
#include "Mesh/Topology/DistributedTopology.h"
#include "Physics/Core/PhysicsModule.h"
#include "FE/LevelSet/AcceptedFunctionalRecorder.h"
#include "FE/LevelSet/LevelSetEnergyParticipant.h"
#include "FE/LevelSet/LevelSetVolumeConstrainedMinimizer.h"
#include "FE/LevelSet/TwoFluidAcceptedStage.h"
#include "FE/LevelSet/TwoFluidStageParticipant.h"
#include "Parameters.h"
#include "tinyxml2.h"
#include "FE/Systems/FunctionalRecords.h"
#include "FE/Systems/TwoFluidStageRecords.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace application::core {

// The complete energy record of a level-set run
// (FE/LevelSet/LevelSetEnergyBalance.h) on the FE energy-balance ledger
// (FE/Systems/EnergyBalanceLedger.h).
using svmp::FE::level_set::FreeSurfaceAcceptedEnergyGroup;
using svmp::FE::level_set::freeSurfaceEnergyBalanceCatalogue;
using svmp::FE::level_set::summarizeAcceptedFreeSurfaceEnergyGroup;
using svmp::FE::level_set::FreeSurfaceCompleteEnergyCandidate;
using svmp::FE::level_set::FreeSurfaceCompleteEnergyBuildResult;
using svmp::FE::level_set::buildFixedTopologyOnePhaseEnergyCandidate;
using svmp::FE::level_set::buildUnstagedFreeSurfaceEnergyAttemptMetadata;
using svmp::FE::level_set::freeSurfaceEnergyRejectionReasonForStepReject;
using svmp::FE::level_set::FreeSurfaceAggregationEnergyClassification;
using svmp::FE::level_set::classifyAcceptedAggregationEnergy;
using svmp::FE::level_set::logLevelSetMaintenanceWorkPublishedSince;
using svmp::FE::level_set::logLevelSetMaintenancePhysicalEndpointChannels;

} // namespace application::core

namespace {

using Clock = std::chrono::steady_clock;
using application::core::ActiveCutVolumeRequest;
using application::core::LevelSetMaintenanceConfigurationHandle;
using application::core::LevelSetActiveSide;
using application::core::activeCutVolumeRequests;

double secondsSince(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// A deprecated input that the run accepted (decision D47): a legacy
// spelling of an XML key that frozen decks still use, a legacy environment
// variable that a key replaced, or a retired setting restated with its
// no-op value (empty replacement).
struct InputDeprecation {
  std::string input;
  std::string replacement;
  std::string location;
};

std::vector<InputDeprecation> collectInputDeprecations(const Parameters& params)
{
  std::vector<InputDeprecation> deprecations;
  if (std::getenv("SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT") != nullptr) {
    deprecations.push_back({"SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT",
                            "Generalized_alpha_initial_rate", "environment"});
  }
  for (const auto* equation : params.equation_parameters) {
    if (equation == nullptr) {
      continue;
    }
    const std::string equation_location =
        "Add_equation type=" +
        (equation->type.defined() ? equation->type.value() : std::string{});
    for (const auto& [spelling, replacement] : equation->deprecated_spellings) {
      deprecations.push_back({spelling, replacement, equation_location});
    }
    for (const auto* bc : equation->boundary_conditions) {
      if (bc == nullptr) {
        continue;
      }
      const std::string bc_location =
          equation_location + " Add_BC name=" +
          (bc->name.defined() ? bc->name.value() : std::string{});
      for (const auto& [spelling, replacement] : bc->deprecated_spellings) {
        deprecations.push_back({spelling, replacement, bc_location});
      }
    }
  }
  return deprecations;
}

void warnInputDeprecations(const std::vector<InputDeprecation>& deprecations)
{
  for (const auto& deprecation : deprecations) {
    const bool retired = deprecation.replacement.empty();
    svmp::FE::LogRecord record("ApplicationDriver",
                               retired ? "retired input setting"
                                       : "deprecated input setting",
                               svmp::FE::LogLevel::WARNING);
    record.diagnostic("input_deprecation")
        .quoted("setting", deprecation.input)
        .quoted("location", deprecation.location)
        .text("decision", "D47");
    if (retired) {
      record.text("effect", "none");
    } else {
      record.quoted("replacement", deprecation.replacement);
    }
    svmp::FE::logRecord(record.rootOnly());
  }
}

std::string jsonQuoted(std::string_view text)
{
  std::string quoted;
  quoted.reserve(text.size() + 2u);
  quoted.push_back('"');
  for (const char ch : text) {
    if (ch == '"' || ch == '\\') {
      quoted.push_back('\\');
    }
    quoted.push_back(ch);
  }
  quoted.push_back('"');
  return quoted;
}

void writeEffectiveConfigurationArtifact(
    const application::core::SimulationComponents& sim,
    const Parameters& params,
    const svmp::MeshComm& comm)
{
  if (comm.rank() != 0) {
    return;
  }

  std::vector<svmp::Physics::EffectiveConfigurationArtifact> artifacts;
  artifacts.reserve(sim.physics_modules.size());
  for (const auto& module : sim.physics_modules) {
    if (!module) {
      continue;
    }
    if (auto artifact = module->effectiveConfigurationArtifact()) {
      if (artifact->component.empty() || artifact->json.empty()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] A physics module returned an incomplete effective-configuration artifact.");
      }
      artifacts.push_back(std::move(*artifact));
    }
  }
  if (artifacts.empty()) {
    return;
  }

  std::sort(artifacts.begin(), artifacts.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.component, lhs.json) < std::tie(rhs.component, rhs.json);
  });

  std::filesystem::path output_directory = ".";
  if (params.general_simulation_parameters.save_results_in_folder.defined() &&
      !params.general_simulation_parameters.save_results_in_folder.value().empty()) {
    output_directory =
        params.general_simulation_parameters.save_results_in_folder.value();
  }
  std::filesystem::create_directories(output_directory);

  const auto output_path = output_directory / "effective_configuration.json";
  const auto temporary_path =
      output_directory / "effective_configuration.json.tmp";
  {
    std::ofstream output(temporary_path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Failed to open effective-configuration artifact '" +
          temporary_path.string() + "'.");
    }
    output << "{\"artifact_schema_version\":1,\"modules\":[";
    for (std::size_t i = 0; i < artifacts.size(); ++i) {
      if (i != 0u) {
        output << ',';
      }
      output << artifacts[i].json;
    }
    output << "]";
    // Written only when the run accepted a deprecated input, so the
    // artifact of every other run is unchanged.
    const auto deprecations = collectInputDeprecations(params);
    if (!deprecations.empty()) {
      output << ",\"input_deprecations\":[";
      for (std::size_t i = 0; i < deprecations.size(); ++i) {
        if (i != 0u) {
          output << ',';
        }
        output << "{\"input\":" << jsonQuoted(deprecations[i].input)
               << ",\"replacement\":"
               << jsonQuoted(deprecations[i].replacement)
               << ",\"location\":" << jsonQuoted(deprecations[i].location)
               << ",\"decision\":\"D47\"}";
      }
      output << "]";
    }
    output << "}\n";
    output.flush();
    if (!output.good()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Failed to write effective-configuration artifact '" +
          temporary_path.string() + "'.");
    }
  }

  std::error_code rename_error;
  std::filesystem::rename(temporary_path, output_path, rename_error);
  if (rename_error) {
    std::error_code remove_error;
    std::filesystem::remove(output_path, remove_error);
    rename_error.clear();
    std::filesystem::rename(temporary_path, output_path, rename_error);
  }
  if (rename_error) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to publish effective-configuration artifact '" +
        output_path.string() + "': " + rename_error.message());
  }

  svmp::FE::logRecord(
      svmp::FE::LogRecord("ApplicationDriver", "wrote effective configuration")
          .diagnostic("effective_configuration")
          .quoted("path", output_path.string())
          .rootOnly());
}

svmp::MeshComm activeFESystemCommunicator(
    const svmp::FE::systems::FESystem& system)
{
#if defined(MESH_HAS_MPI) && FE_HAS_MPI
  return svmp::MeshComm(system.activeMpiCommunicator());
#else
  (void)system;
  return svmp::MeshComm::self();
#endif
}

svmp::FE::systems::CollectiveCommunicator feCollectiveCommunicator(
    const svmp::MeshComm& comm)
{
#if defined(MESH_HAS_MPI) && FE_HAS_MPI
  return svmp::FE::systems::CollectiveCommunicator(
      comm.native(), static_cast<int>(comm.rank()), comm.size());
#else
  (void)comm;
  return {};
#endif
}

// The generated cut-context lifecycle lives in
// FE/LevelSet/LevelSetGeometryService.h; the driver keeps its own spellings.
std::vector<int> communicatorWideBoundaryMarkers(
    const svmp::FE::assembly::IMeshAccess& mesh,
    const svmp::MeshComm& comm)
{
  return svmp::FE::level_set::communicatorWideBoundaryMarkers(
      mesh, feCollectiveCommunicator(comm));
}

std::vector<std::size_t> communicatorWideIndexUnion(
    std::vector<std::size_t> local_indices,
    std::size_t upper_bound,
    const svmp::MeshComm& comm,
    std::string_view purpose)
{
  return svmp::FE::systems::collectiveIndexUnion(
      std::move(local_indices), upper_bound, feCollectiveCommunicator(comm),
      purpose);
}



using OutputTimingStats = svmp::FE::systems::CollectiveTimingStatistics;

using svmp::FE::level_set::cutGeometryMemoryReportEnabled;
using svmp::FE::level_set::logCutGeometryMemory;

OutputTimingStats reduceOutputTiming(double local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveTimingStatistics(
      local, feCollectiveCommunicator(comm));
}

void printOutputTimingLine(const char* label,
                           const OutputTimingStats& stats,
                           bool mpi_parallel,
                           double parent_seconds = 0.0)
{
  const double pct = parent_seconds > 0.0 ? 100.0 * stats.local / parent_seconds : 0.0;
  if (mpi_parallel) {
    std::fprintf(stderr,
                 "  %-24s %10.6f s  (%5.1f%%)  rank min/mean/max %10.6f / %10.6f / %10.6f s\n",
                 label,
                 stats.local,
                 pct,
                 stats.min,
                 stats.mean,
                 stats.max);
  } else {
    std::fprintf(stderr,
                 "  %-24s %10.6f s  (%5.1f%%)\n",
                 label,
                 stats.local,
                 pct);
  }
}

/// @brief Detect the number of physical CPU cores (excluding hyperthreads).
/// Falls back to std::thread::hardware_concurrency() if detection fails.
int detectPhysicalCores()
{
  int physical = 0;
#if defined(__linux__)
  // Count unique physical cores from sysfs topology.
  // Each /sys/devices/system/cpu/cpuN/topology/thread_siblings_list contains
  // the list of sibling logical CPUs sharing one physical core. We count
  // unique "first sibling" entries to get the physical core count.
  std::set<int> seen_first_siblings;
  for (int cpu = 0; cpu < 4096; ++cpu) {
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    std::ifstream f(path);
    if (!f.is_open()) break;
    int first_sibling = -1;
    // thread_siblings_list format: "0-1" or "0,1" or "0" — first integer is
    // the lowest-numbered sibling.
    f >> first_sibling;
    if (first_sibling >= 0) {
      seen_first_siblings.insert(first_sibling);
    }
  }
  physical = static_cast<int>(seen_first_siblings.size());
#endif
  if (physical <= 0) {
    physical = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  }
  return physical;
}

/// @brief Automatically configure OpenMP thread count based on physical cores
/// and number of MPI ranks sharing this node. If OMP_NUM_THREADS is already set
/// by the user, that value is respected.
///
/// Uses physical cores (not logical/hyperthreaded) because FEM workloads are
/// memory-bandwidth-bound and hyperthreading typically hurts performance.
///
/// Logic: threads_per_rank = floor(physical_cores / ranks_on_this_node)
///        clamped to [1, physical_cores].
void configureOpenMPThreads(const svmp::MeshComm& comm)
{
#ifdef _OPENMP
  // If user explicitly set OMP_NUM_THREADS, respect it.
  if (std::getenv("OMP_NUM_THREADS")) {
    return;
  }

  const int physical_cores = detectPhysicalCores();

  // Determine how many MPI ranks share this physical node.
  int ranks_on_node = 1;
#ifdef MESH_HAS_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm.native(), MPI_COMM_TYPE_SHARED, comm.rank(),
                        MPI_INFO_NULL, &node_comm);
    if (node_comm != MPI_COMM_NULL) {
      MPI_Comm_size(node_comm, &ranks_on_node);
      MPI_Comm_free(&node_comm);
    }
  }
#else
  (void)comm;
#endif

  if (ranks_on_node < 1) ranks_on_node = 1;
  int threads = std::max(1, physical_cores / ranks_on_node);
  omp_set_num_threads(threads);
#else
  (void)comm;
#endif
}

std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

bool parseBoolEnv(const char* name, bool default_value)
{
  const char* env = std::getenv(name);
  if (!env) {
    return default_value;
  }
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (v == "1" || v == "true" || v == "on" || v == "yes") {
    return true;
  }
  if (v == "0" || v == "false" || v == "off" || v == "no") {
    return false;
  }
  return default_value;
}

// The environment variable SVMP_GENERATED_STATE_MAX_DISCONTINUITY_RESTARTS was
// replaced by GeneralSimulationParameters/Max_cut_topology_restarts_per_step
// (decision D35).  A run that still sets it stops, so that the variable does
// not silently stop applying.
void rejectRemovedCutTopologyRestartEnvironment()
{
  if (std::getenv("SVMP_GENERATED_STATE_MAX_DISCONTINUITY_RESTARTS") !=
      nullptr) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] The environment variable "
        "SVMP_GENERATED_STATE_MAX_DISCONTINUITY_RESTARTS was removed; set "
        "<Max_cut_topology_restarts_per_step> in "
        "<GeneralSimulationParameters> instead (default: the outer "
        "fixed-point iteration limit; 0 stops and rejects the step at a "
        "cut-topology change) and unset the variable.");
  }
}

// <Generalized_alpha_initial_rate> chooses the first generalized-alpha rate
// uDot_0: Consistent (default) solves the PDE-consistent rate, Zero starts
// from zero.  It replaces SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT (decision D47),
// which stays a deprecated alias for one release: it applies only when the
// key is absent, and a value that contradicts the key stops the run.
constexpr const char* kLegacyInitialRateEnvironment =
    "SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT";

bool generalizedAlphaInitialRateIsConsistent(
    const GeneralSimulationParameters& general)
{
  const char* legacy = std::getenv(kLegacyInitialRateEnvironment);
  const bool legacy_consistent =
      parseBoolEnv(kLegacyInitialRateEnvironment, true);
  const auto& key = general.generalized_alpha_initial_rate;
  if (!key.defined()) {
    return legacy_consistent;
  }
  std::string token = trim_copy(key.value());
  std::transform(token.begin(), token.end(), token.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  bool consistent = true;
  if (token == "consistent") {
    consistent = true;
  } else if (token == "zero") {
    consistent = false;
  } else {
    throw std::runtime_error(
        "[svMultiPhysics::Application] <Generalized_alpha_initial_rate> must "
        "be Consistent or Zero; got '" + key.value() + "'.");
  }
  if (legacy != nullptr) {
    std::string value(legacy);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool recognized = value == "1" || value == "true" || value == "on" ||
                            value == "yes" || value == "0" || value == "false" ||
                            value == "off" || value == "no";
    if (recognized && legacy_consistent != consistent) {
      throw std::runtime_error(
          std::string("[svMultiPhysics::Application] ") +
          kLegacyInitialRateEnvironment + "=" + legacy +
          " contradicts <Generalized_alpha_initial_rate>" + key.value() +
          "</Generalized_alpha_initial_rate>; unset the deprecated "
          "environment variable.");
    }
  }
  return consistent;
}

bool generalizedAlphaPdeRateInitializationRequested(
    const GeneralSimulationParameters& general,
    bool active_cut_domain_present)
{
  // Active-cut systems used to default this solve off because their dt-only
  // operator contains exact zero rows outside the retained physical domain.
  // TimeLoop now regularizes precisely those rows, so an active cut is not a
  // reason to replace the PDE-consistent initial rate with zero.  Keep the
  // argument explicit so this policy remains covered at the application
  // boundary if active-cut startup handling changes again.
  (void)active_cut_domain_present;
  return generalizedAlphaInitialRateIsConsistent(general);
}

struct TransientTimeIntegrationSelection {
  svmp::FE::timestepping::SchemeKind scheme{
      svmp::FE::timestepping::SchemeKind::GeneralizedAlpha};
  std::string_view canonical_name{"GeneralizedAlpha"};
  std::optional<double> generalized_alpha_rho_inf{0.5};
  svmp::FE::Real stage_alpha_f{svmp::FE::Real{1.0}};
};

constexpr std::array<std::pair<
    std::string_view,
    svmp::FE::timestepping::SchemeKind>, 2>
    kTransientTimeIntegrationSchemes{{
        {"GeneralizedAlpha",
         svmp::FE::timestepping::SchemeKind::GeneralizedAlpha},
        {"BackwardEuler",
         svmp::FE::timestepping::SchemeKind::BackwardEuler},
    }};

svmp::FE::timestepping::SchemeKind parseTransientTimeIntegrationScheme(
    std::string_view name)
{
  const auto entry = std::find_if(
      kTransientTimeIntegrationSchemes.begin(),
      kTransientTimeIntegrationSchemes.end(),
      [name](const auto& candidate) {
        return candidate.first == name;
      });
  if (entry == kTransientTimeIntegrationSchemes.end()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Unsupported "
        "<Transient_time_integration_scheme> '" +
        std::string(name) +
        "'. Supported values are exactly 'GeneralizedAlpha' and "
        "'BackwardEuler'.");
  }
  return entry->second;
}

std::string_view transientTimeIntegrationSchemeName(
    svmp::FE::timestepping::SchemeKind scheme)
{
  const auto entry = std::find_if(
      kTransientTimeIntegrationSchemes.begin(),
      kTransientTimeIntegrationSchemes.end(),
      [scheme](const auto& candidate) {
        return candidate.second == scheme;
      });
  if (entry == kTransientTimeIntegrationSchemes.end()) {
    throw std::logic_error(
        "[svMultiPhysics::Application] Transient scheme is outside the "
        "application-supported set.");
  }
  return entry->first;
}

TransientTimeIntegrationSelection resolveTransientTimeIntegrationSelection(
    const GeneralSimulationParameters& parameters)
{
  const auto scheme =
      parseTransientTimeIntegrationScheme(
          parameters.transient_time_integration_scheme.value());
  TransientTimeIntegrationSelection selection;
  selection.scheme = scheme;
  selection.canonical_name = transientTimeIntegrationSchemeName(scheme);
  if (scheme == svmp::FE::timestepping::SchemeKind::BackwardEuler) {
    // A spectral radius is deliberately inapplicable to backward Euler,
    // which uses the accepted endpoint.
    selection.generalized_alpha_rho_inf = std::nullopt;
    selection.stage_alpha_f = svmp::FE::Real{1.0};
    return selection;
  }

  const double rho_inf =
      parameters.spectral_radius_of_infinite_time_step.value();
  const auto generalized_alpha =
      svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(
          rho_inf);
  selection.generalized_alpha_rho_inf = rho_inf;
  selection.stage_alpha_f =
      static_cast<svmp::FE::Real>(generalized_alpha.alpha_f);
  return selection;
}

double parseDoubleEnv(const char* name, double default_value)
{
  const char* env = std::getenv(name);
  if (!env) {
    return default_value;
  }
  char* end = nullptr;
  const double value = std::strtod(env, &end);
  if (end == env || !std::isfinite(value)) {
    return default_value;
  }
  return value;
}

int parseIntEnv(const char* name, int default_value)
{
  const char* env = std::getenv(name);
  if (!env) {
    return default_value;
  }
  char* end = nullptr;
  const long value = std::strtol(env, &end, 10);
  if (end == env || value < static_cast<long>(std::numeric_limits<int>::min()) ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return default_value;
  }
  return static_cast<int>(value);
}

void applyGeneratedStateOuterRelaxationEnvOptions(
    svmp::FE::timestepping::NewtonOptions& opts)
{
  auto& relaxation =
      opts.external_state_fixed_point.dynamic_relaxation;
  if (std::getenv("SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION") !=
      nullptr) {
    // An explicit switch keeps its original meaning: relax every update
    // (on) or none (off).
    relaxation.enabled = parseBoolEnv(
        "SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION",
        relaxation.enabled);
    relaxation.start_iteration = 0;
  }
  if (std::getenv("SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION_START_PASS") !=
      nullptr) {
    // 1-based pass whose update is the first relaxed one; 0 disables.
    const int start_pass = parseIntEnv(
        "SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION_START_PASS", 0);
    relaxation.enabled = start_pass > 0;
    relaxation.start_iteration = std::max(0, start_pass - 1);
  }
  relaxation.initial_factor = parseDoubleEnv(
      "SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION_INITIAL_FACTOR",
      relaxation.initial_factor);
  relaxation.minimum_factor = parseDoubleEnv(
      "SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION_MINIMUM_FACTOR",
      relaxation.minimum_factor);
  relaxation.maximum_factor = parseDoubleEnv(
      "SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION_MAXIMUM_FACTOR",
      relaxation.maximum_factor);
  relaxation.denominator_relative_tolerance = parseDoubleEnv(
      "SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION_DENOMINATOR_RELATIVE_TOLERANCE",
      relaxation.denominator_relative_tolerance);
}

void applyNewtonLineSearchEnvOptions(svmp::FE::timestepping::NewtonOptions& opts)
{
  opts.line_search_max_iterations =
      std::max(1, parseIntEnv("SVMP_NEWTON_LINE_SEARCH_MAX_ITERATIONS",
                              opts.line_search_max_iterations));
  opts.line_search_alpha_min =
      parseDoubleEnv("SVMP_NEWTON_LINE_SEARCH_ALPHA_MIN",
                     opts.line_search_alpha_min);
  opts.line_search_shrink =
      parseDoubleEnv("SVMP_NEWTON_LINE_SEARCH_SHRINK",
                     opts.line_search_shrink);
  opts.line_search_c1 =
      parseDoubleEnv("SVMP_NEWTON_LINE_SEARCH_C1",
                     opts.line_search_c1);
  opts.line_search_fail_on_no_reduction =
      parseBoolEnv("SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION",
                   opts.line_search_fail_on_no_reduction);
}

void applyNewtonToleranceEnvOptions(svmp::FE::timestepping::NewtonOptions& opts)
{
  if (std::getenv("SVMP_NEWTON_ABS_TOLERANCE")) {
    const double value =
        parseDoubleEnv("SVMP_NEWTON_ABS_TOLERANCE", opts.abs_tolerance);
    if (value >= 0.0) {
      opts.abs_tolerance = value;
    }
  }
  if (std::getenv("SVMP_NEWTON_REL_TOLERANCE")) {
    const double value =
        parseDoubleEnv("SVMP_NEWTON_REL_TOLERANCE", opts.rel_tolerance);
    if (value >= 0.0) {
      opts.rel_tolerance = value;
    }
  }
}

void applyNewtonToleranceXmlOptions(
    const GeneralSimulationParameters& general_params,
    svmp::FE::timestepping::NewtonOptions& opts)
{
  if (general_params.newton_absolute_tolerance.defined()) {
    const double value = general_params.newton_absolute_tolerance.value();
    if (!std::isfinite(value) || value < 0.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Newton_absolute_tolerance must be finite and nonnegative.");
    }
    opts.abs_tolerance = value;
  }
  if (general_params.newton_relative_tolerance.defined()) {
    const double value = general_params.newton_relative_tolerance.value();
    if (!std::isfinite(value) || value < 0.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Newton_relative_tolerance must be finite and nonnegative.");
    }
    opts.rel_tolerance = value;
  }
}

void applyNewtonIterationXmlOptions(
    const GeneralSimulationParameters& general_params,
    svmp::FE::timestepping::NewtonOptions& opts)
{
  if (general_params.newton_max_iterations.defined()) {
    const int value = general_params.newton_max_iterations.value();
    if (value < 1) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Newton_max_iterations must be positive.");
    }
    opts.max_iterations = value;
  }
}

void applyNewtonLineSearchXmlOptions(
    const GeneralSimulationParameters& general_params,
    svmp::FE::timestepping::NewtonOptions& opts)
{
  if (general_params.newton_line_search_max_iterations.defined()) {
    opts.line_search_max_iterations =
        std::max(1, general_params.newton_line_search_max_iterations.value());
  }
  if (general_params.newton_line_search_fail_on_no_reduction.defined()) {
    opts.line_search_fail_on_no_reduction =
        general_params.newton_line_search_fail_on_no_reduction.value();
  }
}

void applyNewtonPseudoTransientEnvOptions(svmp::FE::timestepping::NewtonOptions& opts)
{
  auto& ptc = opts.pseudo_transient;
  ptc.enabled =
      parseBoolEnv("SVMP_NEWTON_PTC", ptc.enabled);
  ptc.activate_on_linear_failure =
      parseBoolEnv("SVMP_NEWTON_PTC_ACTIVATE_ON_LINEAR_FAILURE",
                   ptc.activate_on_linear_failure);
  ptc.gamma_initial =
      std::max(0.0,
               parseDoubleEnv("SVMP_NEWTON_PTC_GAMMA_INITIAL",
                              ptc.gamma_initial));

  const double gamma_growth =
      parseDoubleEnv("SVMP_NEWTON_PTC_GAMMA_GROWTH", ptc.gamma_growth);
  if (gamma_growth > 1.0) {
    ptc.gamma_growth = gamma_growth;
  }

  ptc.gamma_max =
      std::max(0.0,
               parseDoubleEnv("SVMP_NEWTON_PTC_GAMMA_MAX",
                              ptc.gamma_max));
  ptc.gamma_drop_tolerance =
      std::max(0.0,
               parseDoubleEnv("SVMP_NEWTON_PTC_GAMMA_DROP_TOLERANCE",
                              ptc.gamma_drop_tolerance));
  ptc.max_linear_retries =
      std::max(1,
               parseIntEnv("SVMP_NEWTON_PTC_MAX_LINEAR_RETRIES",
                           ptc.max_linear_retries));
  ptc.update_from_residual_ratio =
      parseBoolEnv("SVMP_NEWTON_PTC_UPDATE_FROM_RESIDUAL_RATIO",
                   ptc.update_from_residual_ratio);
}

std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool oopStateTraceEnabled()
{
  const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE");
  if (env == nullptr) {
    return false;
  }
  const auto v = lower_copy(env);
  return !(v == "0" || v == "false" || v == "off" || v == "no");
}

void traceStateVectorFields(const svmp::FE::systems::FESystem& system,
                            svmp::FE::backends::GenericVector& vector,
                            const char* label)
{
  if (!oopStateTraceEnabled()) {
    return;
  }
  const auto view = vector.createAssemblyView();
  if (!view) {
    return;
  }
  const auto& fields = system.fieldMap();
  svmp::FE::LogRecord record("ApplicationDriver", "state vector");
  record.diagnostic("state_vector_fields").quoted("label", label).rootOnly();
  for (std::size_t field = 0; field < fields.numFields(); ++field) {
    const auto& rec = fields.getField(field);
    const auto range = fields.getFieldDofRange(field);
    double sq_norm = 0.0;
    double sum = 0.0;
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    std::uint64_t count = 0;
    for (svmp::FE::GlobalIndex dof = range.first; dof < range.second; ++dof) {
      const double value = static_cast<double>(view->getVectorEntry(dof));
      sq_norm += value * value;
      sum += value;
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
      ++count;
    }
    const double mean = count > 0 ? sum / static_cast<double>(count) : 0.0;
    record.child("state_vector_field", {"label"})
        .quoted("field", rec.name)
        .value("dofs", count)
        .value("norm", std::sqrt(std::max(0.0, sq_norm)))
        .value("mean", mean)
        .value("min", count > 0 ? min_value : 0.0)
        .value("max", count > 0 ? max_value : 0.0);
  }
  svmp::FE::logRecord(record);
}

bool parse_bool_relaxed(const std::string& raw)
{
  const auto v = lower_copy(trim_copy(raw));
  if (v == "true" || v == "1" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "false" || v == "0" || v == "no" || v == "off") {
    return false;
  }
  return false;
}

std::string normalized_token(std::string value)
{
  value = lower_copy(trim_copy(std::move(value)));
  value.erase(std::remove_if(value.begin(), value.end(),
                             [](unsigned char c) {
                               return c == '_' || c == '-' || std::isspace(c);
                             }),
              value.end());
  return value;
}

std::optional<std::string> first_defined_parameter(
    const std::map<std::string, std::string>& params,
    std::initializer_list<const char*> keys)
{
  for (const char* key : keys) {
    const auto it = params.find(key);
    if (it != params.end() && !trim_copy(it->second).empty()) {
      return it->second;
    }
  }
  return std::nullopt;
}

std::optional<double> first_defined_double_parameter(
    const std::map<std::string, std::string>& params,
    std::initializer_list<const char*> keys)
{
  if (const auto value = first_defined_parameter(params, keys)) {
    return std::stod(*value);
  }
  return std::nullopt;
}

std::optional<int> first_defined_int_parameter(
    const std::map<std::string, std::string>& params,
    std::initializer_list<const char*> keys)
{
  if (const auto value = first_defined_parameter(params, keys)) {
    return std::stoi(*value);
  }
  return std::nullopt;
}

std::optional<bool> first_defined_bool_parameter(
    const std::map<std::string, std::string>& params,
    std::initializer_list<const char*> keys)
{
  if (const auto value = first_defined_parameter(params, keys)) {
    return parse_bool_relaxed(*value);
  }
  return std::nullopt;
}

std::string step_reject_reason_to_string(svmp::FE::timestepping::StepRejectReason r)
{
  using svmp::FE::timestepping::StepRejectReason;
  switch (r) {
    case StepRejectReason::NonlinearSolveFailed:
      return "NonlinearSolveFailed";
    case StepRejectReason::ErrorTooLarge:
      return "ErrorTooLarge";
    case StepRejectReason::CutTopologyChanged:
      return "CutTopologyChanged";
  }
  return std::to_string(static_cast<int>(r));
}

const EquationParameters* first_equation(const Parameters& params)
{
  for (const auto* e : params.equation_parameters) {
    if (e) {
      return e;
    }
  }
  return nullptr;
}

const EquationParameters* primary_solver_equation(const Parameters& params)
{
  const auto* first = first_equation(params);
  if (first == nullptr) {
    return nullptr;
  }

  if (first->type.defined() && lower_copy(first->type.value()) == "level_set") {
    for (const auto* e : params.equation_parameters) {
      if (e && e->type.defined() && lower_copy(e->type.value()) == "fluid") {
        return e;
      }
    }
  }

  return first;
}

std::vector<const EquationParameters*>
monolithic_solver_control_equations(const Parameters& params)
{
  std::vector<const EquationParameters*> coupled;
  for (auto* equation : params.equation_parameters) {
    if (equation != nullptr && equation->coupled.defined() &&
        equation->coupled.value()) {
      coupled.push_back(equation);
    }
  }
  if (!coupled.empty()) {
    return coupled;
  }

  if (const auto* primary = primary_solver_equation(params)) {
    return {primary};
  }
  return {};
}

void applyMonolithicEquationNewtonControls(
    const Parameters& params,
    svmp::FE::timestepping::NewtonOptions& options)
{
  const auto equations = monolithic_solver_control_equations(params);
  std::optional<int> maximum_iterations;
  std::optional<int> minimum_iterations;
  std::optional<double> strictest_tolerance;
  std::optional<double> strictest_non_level_set_tolerance;

  for (const auto* equation : equations) {
    if (equation->max_iterations.defined()) {
      maximum_iterations = maximum_iterations.has_value()
          ? std::max(*maximum_iterations, equation->max_iterations.value())
          : equation->max_iterations.value();
    }
    if (equation->min_iterations.defined()) {
      const int value = std::max(0, equation->min_iterations.value());
      minimum_iterations = minimum_iterations.has_value()
          ? std::max(*minimum_iterations, value)
          : value;
    }
    if (equation->tolerance.defined()) {
      const double value = equation->tolerance.value();
      if (value > 0.0) {
        strictest_tolerance = strictest_tolerance.has_value()
            ? std::min(*strictest_tolerance, value)
            : value;
        const auto type = equation->type.defined()
            ? normalized_token(equation->type.value())
            : std::string{};
        if (type != "levelset" && type != "levelsettransport") {
          strictest_non_level_set_tolerance =
              strictest_non_level_set_tolerance.has_value()
              ? std::min(*strictest_non_level_set_tolerance, value)
              : value;
        }
      }
    }
  }

  // Opt-in inexact Newton forcing (<LS><Inexact_Newton_forcing>), read from
  // the equations that control the monolithic solve.
  using ForcingMode = svmp::FE::timestepping::NewtonOptions::LinearForcingOptions::Mode;
  for (const auto* equation : equations) {
    const auto& forcing = equation->linear_solver.inexact_newton_forcing;
    if (!forcing.defined()) {
      continue;
    }
    const auto value = normalized_token(forcing.value());
    if (value.empty() || value == "none" || value == "fixed" || value == "off") {
      continue;
    }
    if (value == "eisenstatwalker" || value == "ew") {
      options.linear_forcing.mode = ForcingMode::EisenstatWalker;
      continue;
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Unsupported <Inexact_Newton_forcing> '" +
        forcing.value() + "'. Supported values: none, eisenstat_walker.");
  }

  if (maximum_iterations.has_value()) {
    options.max_iterations = *maximum_iterations;
  }
  if (minimum_iterations.has_value()) {
    options.min_iterations = *minimum_iterations;
  }
  if (strictest_tolerance.has_value()) {
    // Coupled level-set equations receive an owned/global named-field gate
    // below.  Apply the monolithic relative criterion to the remaining
    // equations so a dimensionally smaller transport block cannot force an
    // unrelated momentum/pressure block to its tolerance (or be masked by
    // it).  If the solve contains only level-set equations, retain their
    // strictest tolerance as the monolithic fallback.
    options.rel_tolerance = strictest_non_level_set_tolerance.value_or(
        *strictest_tolerance);
    // Equation <Tolerance> is a relative nonlinear target, not a raw-residual
    // absolute tolerance.  Preserve Newton's configured absolute tolerance;
    // callers may override it explicitly through the Newton options/env path.
  }
}

// The FE run state of the field (the redistancing trigger), with the
// static-capillary initialization state and the conservative-phase graph.
struct LevelSetMaintenanceRuntimeState
    : svmp::FE::level_set::LevelSetMaintenanceRuntime {
  bool static_capillary_equilibrium_initialized{false};
  std::optional<svmp::FE::Real> measured_static_capillary_target{};
  bool conservative_phase_initialized{false};
  std::optional<svmp::FE::level_set::LevelSetP1PhaseTransportGraph>
      conservative_phase_graph{};
};

struct LevelSetMaintenanceRequest {
  explicit LevelSetMaintenanceRequest(
      LevelSetMaintenanceConfigurationHandle configured)
      : configuration(std::move(configured))
  {
    if (!configuration) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Level-set maintenance requires a nonnull immutable configuration.");
    }
  }

  LevelSetMaintenanceConfigurationHandle configuration;
  LevelSetMaintenanceRuntimeState runtime{};
};

std::vector<LevelSetMaintenanceRequest> makeLevelSetMaintenanceRequests(
    std::span<const LevelSetMaintenanceConfigurationHandle> configurations)
{
  std::vector<LevelSetMaintenanceRequest> requests;
  requests.reserve(configurations.size());
  for (const auto& configuration : configurations) {
    requests.emplace_back(configuration);
  }
  return requests;
}

application::core::StaticCapillaryEquilibriumOptions
effectiveStaticCapillaryEquilibriumOptions(
    const LevelSetMaintenanceRequest& request)
{
  auto options = request.configuration->static_capillary_equilibrium;
  if (request.runtime.measured_static_capillary_target.has_value()) {
    options.target_liquid_volume =
        *request.runtime.measured_static_capillary_target;
  }
  return options;
}

bool hasExplicitUnsupportedConservativePhasePointwiseWallContract(
    std::span<const LevelSetMaintenanceRequest> requests) noexcept
{
  return std::any_of(
      requests.begin(), requests.end(), [](const auto& request) {
        return request.configuration->transport.conservative_phase.enabled &&
               request
                   .configuration->transport.conservative_phase.pointwise_impermeable_velocity_tolerance_explicitly_requested;
      });
}

// The schedule stages, the canonical words of the FE configuration and run
// state of a request, the live-geometry words and their agreement are in
// FE/LevelSet/Maintenance/LevelSetMaintenanceRequestSchedule.h.  The driver
// adds the static capillary equilibrium: action bit 11 and its words.
using svmp::FE::level_set::LevelSetMaintenanceScheduleStage;

// The schedule words and their append helpers are in
// FE/LevelSet/LevelSetMaintenanceSchedule.h.
using svmp::FE::level_set::CanonicalLevelSetMaintenanceSchedule;
using svmp::FE::level_set::appendMaintenanceScheduleBool;
using svmp::FE::level_set::appendMaintenanceScheduleEnum;
using svmp::FE::level_set::appendMaintenanceScheduleReal;
using svmp::FE::level_set::appendMaintenanceScheduleSigned;
using svmp::FE::level_set::appendMaintenanceScheduleString;
using svmp::FE::level_set::appendMaintenanceScheduleWord;

using svmp::FE::level_set::CanonicalLevelSetMaintenanceGeometryState;
using svmp::FE::level_set::appendCanonicalLevelSetMaintenanceGeometrySection;
using svmp::FE::level_set::appendCanonicalLevelSetMaintenanceScheduleSection;
using svmp::FE::level_set::appendLevelSetMaintenanceConfigurationScheduleWords;
using svmp::FE::level_set::appendLevelSetMaintenanceScheduleHeader;
using svmp::FE::level_set::appendMaintenanceScheduleGraphMetadata;

// The FE action bits of the request with bit 11, the static capillary
// equilibrium.
std::uint64_t levelSetMaintenanceRequestActionBits(
    const LevelSetMaintenanceRequest& request,
    int completed_step)
{
  std::uint64_t bits =
      svmp::FE::level_set::levelSetMaintenanceRequestActionBits(
          *request.configuration, completed_step);
  bits |= request.configuration->static_capillary_equilibrium_enabled
              ? (1ull << 11u)
              : 0u;
  return bits;
}

CanonicalLevelSetMaintenanceSchedule
canonicalLevelSetMaintenanceRequestSchedule(
    const std::vector<LevelSetMaintenanceRequest>& requests,
    LevelSetMaintenanceScheduleStage stage,
    int completed_step)
{
  CanonicalLevelSetMaintenanceSchedule schedule;
  auto& words = schedule.words;
  words.reserve(32u + 224u * requests.size());
  std::uint64_t aggregate_action_bits = 0u;
  for (const auto& request : requests) {
    aggregate_action_bits |=
        levelSetMaintenanceRequestActionBits(
            request, completed_step);
  }
  appendLevelSetMaintenanceScheduleHeader(
      words, stage, completed_step, requests.size(), aggregate_action_bits);

  for (const auto& request : requests) {
    const auto capillary_options = effectiveStaticCapillaryEquilibriumOptions(request);
    appendMaintenanceScheduleWord(
        words,
        levelSetMaintenanceRequestActionBits(
            request, completed_step));
    schedule.supported =
        appendLevelSetMaintenanceConfigurationScheduleWords(
            words, *request.configuration) &&
        schedule.supported;

    appendMaintenanceScheduleBool(
        words, request.configuration->static_capillary_equilibrium_enabled);
    appendMaintenanceScheduleReal(
        words,
        capillary_options.target_liquid_volume);
    appendMaintenanceScheduleReal(
        words,
        capillary_options.volume_tolerance);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .projected_gradient_tolerance);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .pressure_representability_max_residual_norm);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .pressure_representability_max_relative_distance);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .physical_equilibrium_max_residual_norm);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .constant_pressure_kkt_max_residual_norm);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .constant_pressure_kkt_max_relative_distance);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .finite_difference_reference_coefficient_scale);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .finite_difference_relative_step);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .minimum_finite_difference_step);
    appendMaintenanceScheduleSigned(
        words,
        capillary_options
            .finite_difference_max_shrinks);
    appendMaintenanceScheduleSigned(
        words,
        capillary_options.max_iterations);
    appendMaintenanceScheduleSigned(
        words,
        capillary_options
            .max_line_search_iterations);
    appendMaintenanceScheduleSigned(
        words,
        capillary_options
            .max_topology_epoch_transitions);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .projected_gradient_inverse_stiffness);
    appendMaintenanceScheduleReal(
        words,
        capillary_options.tangent_trust_radius);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .maximum_coefficient_update_linf);
    appendMaintenanceScheduleReal(
        words,
        capillary_options.line_search_shrink);
    appendMaintenanceScheduleReal(
        words,
        capillary_options.armijo_fraction);
    appendMaintenanceScheduleSigned(
        words,
        capillary_options
            .limited_memory_history_size);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .limited_memory_curvature_tolerance);
    appendMaintenanceScheduleReal(
        words,
        capillary_options
            .minimum_volume_merit_penalty);
    appendMaintenanceScheduleBool(
        words, request.runtime.static_capillary_equilibrium_initialized);

    appendMaintenanceScheduleBool(
        words, request.runtime.conservative_phase_initialized);
    appendMaintenanceScheduleGraphMetadata(
        words, request.runtime.conservative_phase_graph);
  }
  return schedule;
}

// The FE part of the driver's requests: the configuration handle and a copy
// of the FE run state.
std::vector<svmp::FE::level_set::LevelSetMaintenanceRequest>
feLevelSetMaintenanceRequests(
    std::span<const LevelSetMaintenanceRequest> requests)
{
  std::vector<svmp::FE::level_set::LevelSetMaintenanceRequest> fe_requests;
  fe_requests.reserve(requests.size());
  for (const auto& request : requests) {
    fe_requests.push_back(svmp::FE::level_set::LevelSetMaintenanceRequest{
        .configuration = request.configuration,
        .runtime = static_cast<
            const svmp::FE::level_set::LevelSetMaintenanceRuntime&>(
            request.runtime),
    });
  }
  return fe_requests;
}

CanonicalLevelSetMaintenanceGeometryState
canonicalLevelSetMaintenanceGeometryState(
    const application::core::SimulationComponents& sim,
    const std::vector<LevelSetMaintenanceRequest>& requests)
{
  return svmp::FE::level_set::canonicalLevelSetMaintenanceGeometryState(
      sim.fe_system.get(), feLevelSetMaintenanceRequests(requests));
}

void requireCollectiveLevelSetMaintenanceRequestSchedule(
    const std::vector<LevelSetMaintenanceRequest>& requests,
    LevelSetMaintenanceScheduleStage stage,
    int completed_step,
    const svmp::MeshComm& comm)
{
  svmp::FE::level_set::requireCollectiveLevelSetMaintenanceRequestSchedule(
      canonicalLevelSetMaintenanceRequestSchedule(
          requests, stage, completed_step),
      feCollectiveCommunicator(comm));
}

// One advection-velocity extension request of a level-set equation
// (FE/LevelSet/Extension/LevelSetVelocityExtensionRefresh.h).
using LevelSetAdvectionVelocityRequest =
    svmp::FE::level_set::LevelSetVelocityExtensionRequest;

std::vector<std::string> splitFaceNameList(std::string_view raw)
{
  std::vector<std::string> names;
  std::string token;
  const auto append = [&]() {
    const auto value = trim_copy(token);
    if (!value.empty() &&
        std::find(names.begin(), names.end(), value) == names.end()) {
      names.push_back(value);
    }
    token.clear();
  };
  for (const char character : raw) {
    if (character == ';' || character == ',') {
      append();
    } else {
      token.push_back(character);
    }
  }
  append();
  return names;
}

// The refresh report, the cut-topology attempt tracker and the topology
// fingerprints live in FE/LevelSet/LevelSetCutTopology.h.
using svmp::FE::level_set::ActiveCutContextRefreshReport;
using svmp::FE::level_set::ActiveCutTopologySnapshotBinding;
using svmp::FE::level_set::CutTopologyFingerprintMultiset;
using svmp::FE::level_set::TransientCutTopologyAttemptTracker;
using svmp::FE::level_set::activeCutTopologyRequestIdentity;
using svmp::FE::level_set::avalancheCutTopologyFingerprint;
using svmp::FE::level_set::cutContextTopologyKey;
using svmp::FE::level_set::finalizeCutTopologyFingerprintMultiset;
using svmp::FE::level_set::freeSurfaceRuleSemanticTopologyFingerprint;
using svmp::FE::level_set::kCutContextHashOffset;
using svmp::FE::level_set::kCutContextHashPrime;
using svmp::FE::level_set::mixCutContextHash;
using svmp::FE::level_set::transientCutTopologyFingerprintSupportsRequests;

struct WetVolumeDiagnostic {
  std::string level_set_field_name{};
  std::string domain_id{};
  int marker{-1};
  std::uint64_t free_surface_snapshot_revision_key{0};
  std::uint64_t source_value_revision{0};
  LevelSetActiveSide active_side{LevelSetActiveSide::Negative};
  double isovalue{0.0};
  svmp::FE::Real wet_volume{0.0};
  svmp::FE::Real reference_wet_volume{0.0};
  svmp::FE::Real physical_wet_volume{0.0};
  std::string wet_volume_frame{"physical"};
  std::size_t volume_rule_count{0};
  std::size_t physical_volume_rule_count{0};
  std::size_t skipped_physical_volume_rule_count{0};
  std::size_t cut_cell_count{0};
  std::size_t full_wet_cell_count{0};
  std::size_t full_dry_cell_count{0};
};

struct ActiveFluidReport {
  std::size_t total_vertices{0};
  std::size_t active_vertices{0};
  std::size_t dry_vertices{0};
};

double globalSumDouble(double local, const svmp::MeshComm& comm);
std::size_t globalSumSize(std::size_t local, const svmp::MeshComm& comm);
std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::MeshComm& comm);
bool globalAnyBool(bool local, const svmp::MeshComm& comm);
void requireCollectivePhasePreparation(
    const std::exception_ptr& local_failure,
    const svmp::MeshComm& comm,
    std::string_view phase);

std::vector<LevelSetAdvectionVelocityRequest>
levelSetAdvectionVelocityRequests(const Parameters& params)
{
  std::vector<LevelSetAdvectionVelocityRequest> requests;
  const auto active_requests = activeCutVolumeRequests(params);

  for (auto* eq : params.equation_parameters) {
    if (eq == nullptr || !eq->type.defined()) {
      continue;
    }
    const auto type = normalized_token(eq->type.value());
    if (type != "levelset" && type != "levelsettransport") {
      continue;
    }

    const auto eq_params = eq->get_parameter_list();
    const auto source_field =
        first_defined_parameter(
            eq_params,
            {"Advection_velocity_from_field",
             "Source_velocity_field_name"});
    const auto method_parameter =
        first_defined_parameter(eq_params, {"Wet_extension_advection_velocity_method",
                                           "Advection_velocity_extension_method"});
    const auto pde_operator =
        method_parameter.has_value()
            ? application::core::pdeVelocityExtensionOperatorFromToken(
                  *method_parameter)
            : std::nullopt;
    // The algebraic wall-compatible wet extension was retired (decisions
    // D47 and D55): the extension needs an explicit PDE method.  The
    // translator rejects these settings first.
    if (first_defined_bool_parameter(
            eq_params, {"Use_wet_extension_advection_velocity"})
            .value_or(false) ||
        first_defined_bool_parameter(
            eq_params, {"Write_velocity_extension_maps"})
            .value_or(false) ||
        ((method_parameter.has_value() || source_field.has_value()) &&
         !pde_operator.has_value())) {
      throw std::runtime_error(
          std::string(application::core::kRetiredWetExtensionMessage));
    }
    if (!pde_operator.has_value()) {
      continue;
    }

    LevelSetAdvectionVelocityRequest request{};
    if (const auto field =
            first_defined_parameter(eq_params, {"Level_set_field_name",
                                               "Field_name"})) {
      request.level_set_field_name = trim_copy(*field);
    }
    if (source_field.has_value()) {
      request.source_velocity_field_name = trim_copy(*source_field);
    }
    if (const auto target =
            first_defined_parameter(eq_params, {"Velocity_field_name"})) {
      request.target_velocity_field_name = trim_copy(*target);
    }
    if (request.target_velocity_field_name.empty()) {
      request.target_velocity_field_name = "LevelSetAdvectionVelocity";
    }
    if (const auto operator_tag = first_defined_parameter(
            eq_params, {"Operator_tag", "OperatorTag"})) {
      request.operator_tag = trim_copy(*operator_tag);
    } else if (first_defined_bool_parameter(eq_params, {"Coupled"})
                   .value_or(false)) {
      request.operator_tag = "equations";
    }
    if (request.operator_tag.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wet-extension operator tag must be non-empty.");
    }
    if (request.target_velocity_field_name == request.source_velocity_field_name) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wet-extension level-set advection requires "
          "Velocity_field_name to be distinct from the physical source velocity field.");
    }

    const auto velocity_source =
        first_defined_parameter(eq_params, {"Velocity_source"});
    if (!velocity_source.has_value() ||
        normalized_token(*velocity_source) != "prescribeddata") {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wet-extension level-set advection requires "
          "<Velocity_source>prescribed_data</Velocity_source> so the transport equation uses "
          "the generated prescribed advection field.");
    }

    request.extension_method = std::string(
        application::core::pdeVelocityExtensionOperatorName(*pde_operator));
    // Wet_extension_band_layers and Wet_extension_enforce_wall_impermeability,
    // which no deck, runner or test set, were removed (decision D47): the
    // extension covers every dry vertex and enforces wall impermeability.
    bool explicit_wall_faces = false;
    if (const auto wall_faces = first_defined_parameter(
            eq_params,
            {"Wet_extension_wall_faces"})) {
      request.wall_face_names = splitFaceNameList(*wall_faces);
      explicit_wall_faces = true;
      if (request.enforce_wall_impermeability &&
          request.wall_face_names.empty()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wet-extension wall-face list is empty.");
      }
    }
    if (request.enforce_wall_impermeability) {
      const auto append_wall_faces = [&](std::string_view raw) {
        for (auto name : splitFaceNameList(raw)) {
          if (std::find(request.wall_face_names.begin(),
                        request.wall_face_names.end(),
                        name) == request.wall_face_names.end()) {
            request.wall_face_names.push_back(std::move(name));
          }
        }
      };
      const auto incompatibility = [](
          const BoundaryConditionParameters& boundary)
          -> std::optional<std::string> {
        if (!boundary.type.defined()) {
          return std::string{"has no boundary-condition type"};
        }
        const auto boundary_type = normalized_token(boundary.type.value());
        if (boundary_type != "dir" && boundary_type != "dirichlet") {
          return std::string{"is not a velocity Dirichlet condition"};
        }
        if (boundary.weakly_applied.value()) {
          return std::string{"is weakly applied, not a strong constraint"};
        }
        const auto time_dependence =
            normalized_token(boundary.time_dependence.value());
        if (!time_dependence.empty() && time_dependence != "steady") {
          return std::string{"is time/spatially varying and cannot be proven homogeneous"};
        }
        const auto value = boundary.value.value();
        if (!std::isfinite(value) || std::abs(value) > 1.0e-14) {
          return std::string{"prescribes a nonzero or non-finite velocity"};
        }
        if ((boundary.temporal_and_spatial_values_file_path.defined() &&
             !trim_copy(boundary.temporal_and_spatial_values_file_path.value())
                  .empty()) ||
            (boundary.temporal_values_file_path.defined() &&
             !trim_copy(boundary.temporal_values_file_path.value()).empty()) ||
            (boundary.spatial_values_file_path.defined() &&
             !trim_copy(boundary.spatial_values_file_path.value()).empty()) ||
            (boundary.bct_file_path.defined() &&
             !trim_copy(boundary.bct_file_path.value()).empty())) {
          return std::string{"uses prescribed data whose zero component mask is not statically known"};
        }
        return std::nullopt;
      };

      const EquationParameters* velocity_equation = nullptr;
      for (const auto* candidate_equation : params.equation_parameters) {
        if (candidate_equation == nullptr ||
            !candidate_equation->type.defined() ||
            lower_copy(candidate_equation->type.value()) != "fluid") {
          continue;
        }
        if (velocity_equation != nullptr) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Wet-extension wall constraint "
              "discovery found multiple fluid equations. The owner of source "
              "velocity field '" + request.source_velocity_field_name +
              "' is ambiguous.");
        }
        velocity_equation = candidate_equation;
      }
      if (velocity_equation == nullptr) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wet-extension wall constraint "
            "discovery requires exactly one fluid equation owning source "
            "velocity field '" + request.source_velocity_field_name + "'.");
      }

      for (auto* boundary : velocity_equation->boundary_conditions) {
        if (boundary == nullptr) {
          continue;
        }
        if (!explicit_wall_faces && boundary->name.defined() &&
            !incompatibility(*boundary).has_value()) {
            append_wall_faces(boundary->name.value());
        }
        // The plural contact-wall face spellings read here were removed
        // (decision D47); the singular Contact_line_wall_face was never read
        // by this discovery, so no accepted deck changes.
      }

      for (const auto& wall_face_name : request.wall_face_names) {
        const BoundaryConditionParameters* matching_dirichlet = nullptr;
        for (auto* boundary : velocity_equation->boundary_conditions) {
          if (boundary == nullptr || !boundary->name.defined() ||
              trim_copy(boundary->name.value()) != wall_face_name ||
              !boundary->type.defined()) {
            continue;
          }
          const auto boundary_type = normalized_token(boundary->type.value());
          if (boundary_type != "dir" && boundary_type != "dirichlet") {
            continue;
          }
          if (matching_dirichlet != nullptr) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Wet-extension wall face '" +
                wall_face_name +
                "' has multiple velocity Dirichlet records in its owning "
                "fluid equation; its strong zero-component mask is ambiguous.");
          }
          matching_dirichlet = boundary;
        }
        if (matching_dirichlet == nullptr) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Wet-extension wall face '" +
              wall_face_name +
              "' has no matching velocity Dirichlet condition. Wall-compatible "
              "extension requires an actual strong homogeneous velocity constraint.");
        }
        if (const auto reason = incompatibility(*matching_dirichlet)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Wet-extension wall face '" +
              wall_face_name + "' " + *reason + ".");
        }
        request.wall_constraints.push_back(
            LevelSetAdvectionVelocityRequest::WallConstraint{
                .face_name = wall_face_name,
                .effective_direction =
                    matching_dirichlet->effective_direction.defined()
                        ? matching_dirichlet->effective_direction.value()
                        : std::vector<int>{}});
      }
    }

    if (const auto isovalue =
            first_defined_double_parameter(eq_params, {"Level_set_isovalue"})) {
      request.isovalue = *isovalue;
    }

    for (std::size_t active_request_index = 0u;
         active_request_index < active_requests.size();
         ++active_request_index) {
      const auto& active_request = active_requests[active_request_index];
      if (active_request.level_set_field_name != request.level_set_field_name) {
        continue;
      }
      request.domain_id = active_request.domain_id;
      request.isovalue = active_request.isovalue;
      request.requested_interface_marker =
          active_request.requested_interface_marker;
      request.active_cut_request_index = active_request_index;
      request.active_side = active_request.active_side;
      break;
    }

    requests.push_back(std::move(request));
  }

  return requests;
}

bool activeSideContains(double phi, const ActiveCutVolumeRequest& request)
{
  return request.active_side == LevelSetActiveSide::Negative
             ? phi <= request.isovalue
             : phi >= request.isovalue;
}

using svmp::FE::level_set::activeSideName;

using svmp::FE::level_set::cutIntegrationSide;

// Moving-quadrature consistency probe (SVMP_CUT_RULE_DUMP=1): emit every
// retained generated volume rule with its physical quadrature points and
// weights so an offline tool can integrate known monomials over the active
// domain and compare against the analytic moving-domain integrals (exact in
// the MMS family). One block per accepted step.
void dumpActiveCutVolumeRulesForProbe(const svmp::FE::systems::FESystem& system,
                                      int step)
{
  const auto* context = system.cutIntegrationContext();
  if (context == nullptr) {
    return;
  }
  context->assertAllFreeSurfaceGeometrySnapshotsCurrent(system.meshAccess());
  const auto& rules = context->volumeRules();
  const auto& metadata = context->metadata();
  svmp::FE::LogRecord dump("ApplicationDriver", "cut rule dump");
  dump.diagnostic("cut_rule_dump")
      .value("step", step)
      .value("n_rules", rules.size())
      .rootOnly();
  for (std::size_t i = 0; i < rules.size() && i < metadata.size(); ++i) {
    std::optional<svmp::FE::geometry::CutQuadratureRule> materialized;
    if (context->volumeRuleIsClassificationOnly(i)) {
      materialized.emplace(context->materializedVolumeRule(i));
    }
    const auto& rule = materialized ? *materialized : rules[i];
    if (rule.kind != svmp::FE::geometry::CutQuadratureKind::Volume) {
      continue;
    }
    const auto& meta = metadata[i];
    const auto cell = meta.cell >= 0 ? meta.cell : meta.parent_entity;
    const auto side_name = [](svmp::FE::geometry::CutIntegrationSide s) {
      return s == svmp::FE::geometry::CutIntegrationSide::Negative   ? "neg"
             : s == svmp::FE::geometry::CutIntegrationSide::Positive ? "pos"
                                                                     : "ifc";
    };
    const char* side = side_name(rule.side);
    auto& rule_line = dump.child("cut_rule", {"step"});
    rule_line.value("idx", i)
        .value("cell", cell)
        .value("side", side)
        .value("mside", side_name(meta.side))
        .value("full", rule.full_cell_equivalent ? 1 : 0)
        .value("measure", rule.measure)
        .value("parent_measure", rule.parent_measure)
        .value("vf", rule.volume_fraction)
        .value("mvf", meta.volume_fraction)
        .value("n_qp", rule.points.size());
    for (const auto& q : rule.points) {
      rule_line.child("cut_rule_point", {"step", "cell", "side"})
          .value("x", q.point[0])
          .value("y", q.point[1])
          .value("z", q.point[2])
          .value("w", q.weight);
    }
  }
  svmp::FE::logRecord(dump);
}

std::string fieldNameToken(std::string value)
{
  value = trim_copy(std::move(value));
  for (auto& c : value) {
    const auto uc = static_cast<unsigned char>(c);
    if (!std::isalnum(uc)) {
      c = '_';
    }
  }
  value.erase(std::unique(value.begin(), value.end(),
                          [](char a, char b) {
                            return a == '_' && b == '_';
                          }),
              value.end());
  while (!value.empty() && value.front() == '_') {
    value.erase(value.begin());
  }
  while (!value.empty() && value.back() == '_') {
    value.pop_back();
  }
  return value.empty() ? std::string{"free_surface"} : value;
}

std::string wetVolumeFractionFieldName(
    const ActiveCutVolumeRequest& request,
    std::size_t request_index)
{
  if (request_index == 0u) {
    return "WetVolumeFraction";
  }
  return "WetVolumeFraction_" + fieldNameToken(request.domain_id);
}

std::string wetVolumeMeasureFieldName(
    const ActiveCutVolumeRequest& request,
    std::size_t request_index)
{
  if (request_index == 0u) {
    return "WetVolumeMeasure";
  }
  return "WetVolumeMeasure_" + fieldNameToken(request.domain_id);
}

std::optional<int> generatedVolumeMarkerForRequest(
    const svmp::FE::assembly::CutIntegrationContext& cut_context,
    const ActiveCutVolumeRequest& request,
    std::size_t request_index)
{
  if (request.requested_interface_marker >= 0) {
    return request.requested_interface_marker;
  }
  const auto& markers = cut_context.generatedVolumeMarkers();
  if (request_index < markers.size()) {
    return markers[request_index];
  }
  return std::nullopt;
}

// Driver spellings of the FE topology collectives (svmp::MeshComm).
CutTopologyFingerprintMultiset collectiveCutTopologyFingerprintMultiset(
    const CutTopologyFingerprintMultiset& local,
    const svmp::MeshComm& comm)
{
  return svmp::FE::level_set::collectiveCutTopologyFingerprintMultiset(
      local, feCollectiveCommunicator(comm));
}

std::uint64_t collectivePartitionIndependentCutTopologyFingerprint(
    std::span<const std::uint64_t> request_identities,
    std::span<const std::uint64_t> local_owned_rule_fingerprints,
    const svmp::MeshComm& comm)
{
  return svmp::FE::level_set::
      collectivePartitionIndependentCutTopologyFingerprint(
          request_identities,
          local_owned_rule_fingerprints,
          feCollectiveCommunicator(comm));
}

std::uint64_t activeCutContextTopologyFingerprint(
    std::span<const ActiveCutTopologySnapshotBinding> bindings,
    const svmp::MeshComm& comm)
{
  return svmp::FE::level_set::activeCutContextTopologyFingerprint(
      bindings, feCollectiveCommunicator(comm));
}

// The generated-state synchronization-point predicates live in
// FE/TimeStepping/GeneratedState.h.
const char* stateSyncPointName(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point) noexcept
{
  return svmp::FE::timestepping::generatedStateSyncPointName(point);
}

using svmp::FE::timestepping::refreshesFrozenLevelSetExtensionAtStateSync;
using svmp::FE::timestepping::restoresAcceptedTimeStepGeneratedState;
using svmp::FE::timestepping::synchronizeTransientLineSearchTrials;

std::string geometryTangentPolicySummary(
    const std::vector<ActiveCutVolumeRequest>& requests)
{
  std::set<std::string> policies;
  for (const auto& request : requests) {
    policies.insert(svmp::FE::level_set::geometryTangentPolicyName(
        request.geometry_tangent_policy));
  }
  if (policies.empty()) {
    return {};
  }
  std::ostringstream oss;
  bool first = true;
  for (const auto& policy : policies) {
    if (!first) {
      oss << ",";
    }
    first = false;
    oss << policy;
  }
  return oss.str();
}

void applyJacobianCheckGeometryProvenance(
    svmp::FE::timestepping::NewtonOptions& options,
    const std::vector<ActiveCutVolumeRequest>& requests,
    bool refresh_generated_geometry_within_solve,
    bool has_frozen_algebraic_level_set_extension = false,
    bool use_external_state_fixed_point = false)
{
  if (use_external_state_fixed_point && !requests.empty()) {
    options.jacobian_check_geometry_mode =
        svmp::FE::timestepping::JacobianCheckGeometryMode::FixedGeometry;
    const auto policy = geometryTangentPolicySummary(requests);
    options.jacobian_check_geometry_tangent_policy =
        "outer-fixed-point frozen geometry";
    if (!policy.empty()) {
      options.jacobian_check_geometry_tangent_policy += " (" + policy + ")";
    }
  } else if (refresh_generated_geometry_within_solve && !requests.empty()) {
    // A finite-difference perturbation traverses R(u, G(u)) whenever the
    // synchronization callback regenerates any active cut geometry.  This is
    // true for LinearCorner as well as HighOrderImplicit requests; describing
    // the former as fixed geometry hides the same omitted dG/du terms from the
    // Jacobian diagnostic.
    options.jacobian_check_geometry_mode =
        svmp::FE::timestepping::JacobianCheckGeometryMode::RefreshedGeometry;
    options.jacobian_check_geometry_tangent_policy =
        geometryTangentPolicySummary(requests);
  } else if (has_frozen_algebraic_level_set_extension) {
    options.jacobian_check_geometry_mode =
        svmp::FE::timestepping::JacobianCheckGeometryMode::FixedGeometry;
    options.jacobian_check_geometry_tangent_policy =
        "fixed-topology algebraic wet-extension solve";
  }
}

using svmp::FE::level_set::logCutTopologyChange;

std::size_t writeWetVolumeFractionOutput(
    svmp::Mesh& mesh,
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::assembly::CutIntegrationContext* cut_context)
{
  if (requests.empty() || cut_context == nullptr) {
    return 0u;
  }

  std::size_t fields_written = 0u;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    const auto marker = generatedVolumeMarkerForRequest(
        *cut_context, request, i);
    if (!marker.has_value()) {
      continue;
    }

    // Write the field also where this rank has no active-side rule (zero
    // wet fraction) so every partition carries the same field set.
    const auto side = cutIntegrationSide(request.active_side);
    const auto rules =
        cut_context->generatedVolumeRulesForMarkerAndSide(*marker, side);

    const auto field_name = wetVolumeFractionFieldName(request, i);
    const auto measure_field_name = wetVolumeMeasureFieldName(request, i);
    fields_written +=
        application::core::writeWetVolumeFractionField(
            mesh, field_name, rules, measure_field_name);
  }

  return fields_written;
}

std::vector<WetVolumeDiagnostic> collectWetVolumeDiagnostics(
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::assembly::CutIntegrationContext* cut_context,
    const svmp::FE::assembly::IMeshAccess& mesh,
    std::size_t n_cells,
    const svmp::MeshComm& comm)
{
  std::vector<WetVolumeDiagnostic> diagnostics;
  if (requests.empty() || cut_context == nullptr) {
    return diagnostics;
  }

  diagnostics.reserve(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    const auto marker = generatedVolumeMarkerForRequest(
        *cut_context, request, i);
    if (!marker.has_value()) {
      continue;
    }

    const auto side = cutIntegrationSide(request.active_side);
    const auto rules =
        cut_context->generatedVolumeRulesForMarkerAndSide(*marker, side);
    // A fully dry partition holds no active-side rule; the diagnostics below
    // reduce over the communicator, so skip only when no rank has one.
    if (!globalAnyBool(!rules.empty(), comm)) {
      continue;
    }

    std::vector<double> wet_fraction(n_cells, 0.0);
    WetVolumeDiagnostic diagnostic;
    diagnostic.level_set_field_name = request.level_set_field_name;
    diagnostic.domain_id = request.domain_id;
    diagnostic.marker = *marker;
    diagnostic.active_side = request.active_side;
    diagnostic.isovalue = request.isovalue;
    if (cut_context->hasFreeSurfaceGeometrySnapshotForMarker(*marker)) {
      diagnostic.free_surface_snapshot_revision_key =
          cut_context->freeSurfaceGeometrySnapshotRevisionForMarker(*marker);
      const auto& snapshots = cut_context->freeSurfaceGeometrySnapshots();
      const auto found = std::find_if(
          snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
            return candidate &&
                   candidate->revision().snapshot_revision_key ==
                       diagnostic.free_surface_snapshot_revision_key;
          });
      if (found == snapshots.end()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wet-volume diagnostic could not "
            "resolve its authoritative geometry snapshot revision.");
      }
      diagnostic.source_value_revision =
          (*found)->revision().source_value_revision;
      const auto [minimum_snapshot_revision, maximum_snapshot_revision] =
          globalMinMaxUint64(
              diagnostic.free_surface_snapshot_revision_key, comm);
      const auto [minimum_source_revision, maximum_source_revision] =
          globalMinMaxUint64(diagnostic.source_value_revision, comm);
      if (minimum_snapshot_revision != maximum_snapshot_revision ||
          minimum_source_revision != maximum_source_revision) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wet-volume geometry revision "
            "differs across the FE communicator.");
      }
    }
    const auto local_measure_summary =
        application::core::collectCutVolumeMeasures(mesh, rules, cut_context);
    if (local_measure_summary.revisioned_rule_count != 0u &&
        (local_measure_summary.free_surface_snapshot_revision_key !=
             diagnostic.free_surface_snapshot_revision_key ||
         local_measure_summary.source_value_revision !=
             diagnostic.source_value_revision)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wet-volume measure does not match "
          "the authoritative geometry snapshot revision.");
    }
    application::core::CutVolumeMeasureSummary measure_summary;
    measure_summary.free_surface_snapshot_revision_key =
        diagnostic.free_surface_snapshot_revision_key;
    measure_summary.source_value_revision =
        diagnostic.source_value_revision;
    measure_summary.reference_measure = static_cast<svmp::FE::Real>(
        globalSumDouble(
            static_cast<double>(local_measure_summary.reference_measure),
            comm));
    measure_summary.physical_measure = static_cast<svmp::FE::Real>(
        globalSumDouble(
            static_cast<double>(local_measure_summary.physical_measure),
            comm));
    measure_summary.rule_count =
        globalSumSize(local_measure_summary.rule_count, comm);
    measure_summary.revisioned_rule_count =
        globalSumSize(local_measure_summary.revisioned_rule_count, comm);
    measure_summary.physical_rule_count =
        globalSumSize(local_measure_summary.physical_rule_count, comm);
    measure_summary.skipped_physical_rule_count =
        globalSumSize(local_measure_summary.skipped_physical_rule_count, comm);
    diagnostic.reference_wet_volume = measure_summary.reference_measure;
    diagnostic.physical_wet_volume = measure_summary.physical_measure;
    diagnostic.volume_rule_count = measure_summary.rule_count;
    diagnostic.physical_volume_rule_count =
        measure_summary.physical_rule_count;
    diagnostic.skipped_physical_volume_rule_count =
        measure_summary.skipped_physical_rule_count;
    const auto selected_wet_volume =
        application::core::selectWetVolumeForDrift(measure_summary);
    diagnostic.wet_volume = selected_wet_volume.wet_volume;
    diagnostic.wet_volume_frame = selected_wet_volume.frame;
    for (const auto* rule : rules) {
      if (rule != nullptr) {
        const auto cell = rule->provenance.parent_entity;
        if (cell >= 0 && static_cast<std::size_t>(cell) < wet_fraction.size() &&
            mesh.isOwnedCell(cell)) {
          wet_fraction[static_cast<std::size_t>(cell)] = std::clamp(
              wet_fraction[static_cast<std::size_t>(cell)] +
                  static_cast<double>(rule->volume_fraction),
              0.0,
              1.0);
        }
      }
    }
    constexpr double fraction_tol = 1.0e-12;
    for (std::size_t cell = 0; cell < wet_fraction.size(); ++cell) {
      if (!mesh.isOwnedCell(static_cast<svmp::FE::GlobalIndex>(cell))) {
        continue;
      }
      const auto fraction = wet_fraction[cell];
      if (fraction <= fraction_tol) {
        ++diagnostic.full_dry_cell_count;
      } else if (fraction >= 1.0 - fraction_tol) {
        ++diagnostic.full_wet_cell_count;
      } else {
        ++diagnostic.cut_cell_count;
      }
    }
    diagnostic.cut_cell_count =
        globalSumSize(diagnostic.cut_cell_count, comm);
    diagnostic.full_wet_cell_count =
        globalSumSize(diagnostic.full_wet_cell_count, comm);
    diagnostic.full_dry_cell_count =
        globalSumSize(diagnostic.full_dry_cell_count, comm);
    diagnostics.push_back(std::move(diagnostic));
  }

  return diagnostics;
}

void logActiveCutVolumeAvailabilityWarnings(
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::assembly::CutIntegrationContext* cut_context,
    int step,
    double time)
{
  if (requests.empty()) {
    return;
  }
  if (cut_context == nullptr) {
    svmp::FE::logRecord(
        svmp::FE::LogRecord("ApplicationDriver", "active-domain cut context is unavailable", svmp::FE::LogLevel::WARNING)
            .diagnostic("missing_active_cut_context")
            .value("step", step)
            .value("time", time)
            .value("requests", requests.size())
            .rootOnly());
    return;
  }

  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    const auto marker = generatedVolumeMarkerForRequest(
        *cut_context, request, i);
    if (!marker.has_value()) {
      svmp::FE::logRecord(
          svmp::FE::LogRecord("ApplicationDriver", "active-domain cut context has no generated marker", svmp::FE::LogLevel::WARNING)
              .diagnostic("missing_generated_cut_marker")
              .value("step", step)
              .value("time", time)
              .quoted("field", request.level_set_field_name)
              .quoted("domain_id", request.domain_id)
              .value("requested_marker", request.requested_interface_marker)
              .value("generated_marker_count", cut_context->generatedVolumeMarkers().size())
              .rootOnly());
      continue;
    }

    const auto side = cutIntegrationSide(request.active_side);
    const auto rules =
        cut_context->generatedVolumeRulesForMarkerAndSide(*marker, side);
    if (rules.empty()) {
      svmp::FE::logRecord(
          svmp::FE::LogRecord("ApplicationDriver", "active-domain cut context has no retained volume rules", svmp::FE::LogLevel::WARNING)
              .diagnostic("empty_active_cut_volume_rules")
              .value("step", step)
              .value("time", time)
              .value("marker", *marker)
              .quoted("field", request.level_set_field_name)
              .quoted("domain_id", request.domain_id)
              .value("active_side", activeSideName(request.active_side))
              .value("isovalue", request.isovalue)
              .value("pruned_volume_rules", cut_context->generatedPrunedVolumeRuleCount())
              .value("pruned_volume", cut_context->generatedPrunedVolumeMeasure())
              .rootOnly());
    }
  }
}

void logActiveFluidWetFractionDisagreementWarnings(
    const svmp::Mesh& mesh,
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::assembly::CutIntegrationContext* cut_context,
    int step,
    double time)
{
  if (requests.empty() || cut_context == nullptr) {
    return;
  }

  constexpr double fraction_tol = 1.0e-12;
  constexpr double strong_disagreement_threshold = 0.5;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    if (!mesh.has_field(svmp::EntityKind::Vertex,
                        request.level_set_field_name)) {
      continue;
    }
    const auto phi_handle =
        mesh.field_handle(svmp::EntityKind::Vertex,
                          request.level_set_field_name);
    if (mesh.field_type(phi_handle) != svmp::FieldScalarType::Float64 ||
        mesh.field_components(phi_handle) != 1u) {
      continue;
    }
    const auto* phi = static_cast<const double*>(mesh.field_data(phi_handle));
    if (phi == nullptr) {
      continue;
    }

    const auto marker = generatedVolumeMarkerForRequest(
        *cut_context, request, i);
    if (!marker.has_value()) {
      continue;
    }
    const auto side = cutIntegrationSide(request.active_side);
    const auto rules =
        cut_context->generatedVolumeRulesForMarkerAndSide(*marker, side);
    if (rules.empty()) {
      continue;
    }

    std::vector<double> wet_fraction(mesh.n_cells(), 0.0);
    for (const auto* rule : rules) {
      if (rule == nullptr) {
        continue;
      }
      const auto cell = rule->provenance.parent_entity;
      if (cell < 0 ||
          static_cast<std::size_t>(cell) >= wet_fraction.size()) {
        continue;
      }
      auto& fraction = wet_fraction[static_cast<std::size_t>(cell)];
      fraction = std::clamp(
          fraction + static_cast<double>(rule->volume_fraction),
          0.0,
          1.0);
    }

    std::size_t compared_cut_cell_count = 0u;
    std::size_t disagreeing_cut_cell_count = 0u;
    double max_abs_difference = 0.0;
    svmp::FE::MeshIndex max_difference_cell =
        static_cast<svmp::FE::MeshIndex>(-1);
    for (std::size_t c = 0; c < wet_fraction.size(); ++c) {
      const auto cut_fraction = wet_fraction[c];
      if (cut_fraction <= fraction_tol ||
          cut_fraction >= 1.0 - fraction_tol) {
        continue;
      }
      const auto [vertices, vertex_count] =
          mesh.cell_vertices_span(static_cast<svmp::index_t>(c));
      if (vertices == nullptr || vertex_count == 0u) {
        continue;
      }
      std::size_t active_vertex_count = 0u;
      std::size_t valid_vertex_count = 0u;
      for (std::size_t j = 0; j < vertex_count; ++j) {
        const auto vertex = vertices[j];
        if (vertex < 0 ||
            static_cast<std::size_t>(vertex) >= mesh.n_vertices()) {
          continue;
        }
        ++valid_vertex_count;
        if (activeSideContains(phi[static_cast<std::size_t>(vertex)],
                               request)) {
          ++active_vertex_count;
        }
      }
      if (valid_vertex_count == 0u) {
        continue;
      }

      ++compared_cut_cell_count;
      const auto vertex_fraction =
          static_cast<double>(active_vertex_count) /
          static_cast<double>(valid_vertex_count);
      const auto abs_difference =
          std::abs(vertex_fraction - cut_fraction);
      if (abs_difference > max_abs_difference) {
        max_abs_difference = abs_difference;
        max_difference_cell = static_cast<svmp::FE::MeshIndex>(c);
      }
      if (abs_difference >= strong_disagreement_threshold) {
        ++disagreeing_cut_cell_count;
      }
    }

    if (disagreeing_cut_cell_count == 0u) {
      continue;
    }
    svmp::FE::logRecord(
        svmp::FE::LogRecord("ApplicationDriver", "ActiveFluid and WetVolumeFraction disagree", svmp::FE::LogLevel::WARNING)
            .diagnostic("active_fluid_wet_fraction_disagreement")
            .value("step", step)
            .value("time", time)
            .quoted("field", request.level_set_field_name)
            .quoted("domain_id", request.domain_id)
            .value("marker", *marker)
            .value("active_side", activeSideName(request.active_side))
            .value("isovalue", request.isovalue)
            .value("compared_cut_cell_count", compared_cut_cell_count)
            .value("disagreeing_cut_cell_count", disagreeing_cut_cell_count)
            .value("threshold", strong_disagreement_threshold)
            .value("max_abs_difference", max_abs_difference)
            .value("max_difference_cell", max_difference_cell)
            .rootOnly());
  }
}

void logWetVolumeDiagnostics(
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::assembly::CutIntegrationContext* cut_context,
    const svmp::FE::assembly::IMeshAccess& mesh,
    std::size_t n_cells,
    const svmp::MeshComm& comm,
    int step,
    double time,
    std::map<std::string, svmp::FE::Real>& initial_wet_volume_by_key)
{
  if (cut_context != nullptr) {
    cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh);
  }
  logActiveCutVolumeAvailabilityWarnings(requests, cut_context, step, time);
  const auto diagnostics =
      collectWetVolumeDiagnostics(
          requests, cut_context, mesh, n_cells, comm);
  const double drift_warning_threshold =
      parseDoubleEnv("SVMP_WET_VOLUME_DRIFT_WARNING", 1.0e-3);
  for (const auto& diagnostic : diagnostics) {
    const std::string key = diagnostic.level_set_field_name + "|" +
                            diagnostic.domain_id + "|" +
                            std::to_string(diagnostic.marker);
    const auto drift = application::core::computeWetVolumeDrift(
        key, diagnostic.wet_volume, initial_wet_volume_by_key);
    svmp::FE::logRecord(
        svmp::FE::LogRecord("ApplicationDriver", "wet volume")
            .diagnostic("wet_volume")
            .value("step", step)
            .value("time", time)
            .quoted("field", diagnostic.level_set_field_name)
            .quoted("domain_id", diagnostic.domain_id)
            .value("marker", diagnostic.marker)
            .value("free_surface_snapshot_revision_key", diagnostic.free_surface_snapshot_revision_key)
            .value("source_value_revision", diagnostic.source_value_revision)
            .value("active_side", activeSideName(diagnostic.active_side))
            .value("isovalue", diagnostic.isovalue)
            .value("wet_volume", diagnostic.wet_volume)
            .value("wet_volume_frame", diagnostic.wet_volume_frame)
            .value("reference_wet_volume", diagnostic.reference_wet_volume)
            .value("physical_wet_volume", diagnostic.physical_wet_volume)
            .value("initial_wet_volume", drift.initial_wet_volume)
            .value("wet_volume_drift", drift.wet_volume_drift)
            .value("relative_wet_volume_drift", drift.relative_wet_volume_drift)
            .value("volume_rule_count", diagnostic.volume_rule_count)
            .value("physical_volume_rule_count", diagnostic.physical_volume_rule_count)
            .value("skipped_physical_volume_rule_count", diagnostic.skipped_physical_volume_rule_count)
            .value("cut_cell_count", diagnostic.cut_cell_count)
            .value("full_wet_cell_count", diagnostic.full_wet_cell_count)
            .value("full_dry_cell_count", diagnostic.full_dry_cell_count)
            .rootOnly());
    if (drift_warning_threshold > 0.0 &&
        std::abs(static_cast<double>(drift.relative_wet_volume_drift)) >
            drift_warning_threshold) {
      svmp::FE::logRecord(
          svmp::FE::LogRecord("ApplicationDriver", "wet-volume drift exceeds the diagnostic threshold", svmp::FE::LogLevel::WARNING)
              .diagnostic("nonconservative_level_set_volume_drift")
              .value("step", step)
              .value("time", time)
              .quoted("field", diagnostic.level_set_field_name)
              .quoted("domain_id", diagnostic.domain_id)
              .value("marker", diagnostic.marker)
              .value("active_side", activeSideName(diagnostic.active_side))
              .value("relative_wet_volume_drift", drift.relative_wet_volume_drift)
              .value("threshold", drift_warning_threshold)
              .rootOnly());
    }
  }
}

ActiveFluidReport writeActiveFluidVisualizationOutput(
    svmp::Mesh& mesh,
    const std::vector<ActiveCutVolumeRequest>& requests)
{
  constexpr const char* kActiveFluidVisualizationField = "ActiveFluid";
  ActiveFluidReport report{};
  report.total_vertices = mesh.n_vertices();
  if (requests.empty()) {
    return report;
  }

  // The current OOP free-surface path supports one active-domain level set for
  // the Navier-Stokes volume. If more are present, use the first one for the
  // visualization indicator instead of inventing ambiguous multi-interface
  // semantics here.
  const auto& request = requests.front();
  if (!mesh.has_field(svmp::EntityKind::Vertex, request.level_set_field_name)) {
    return report;
  }
  const auto phi_handle =
      mesh.field_handle(svmp::EntityKind::Vertex, request.level_set_field_name);
  if (mesh.field_type(phi_handle) != svmp::FieldScalarType::Float64 ||
      mesh.field_components(phi_handle) != 1u) {
    return report;
  }
  const auto* phi = static_cast<const double*>(mesh.field_data(phi_handle));
  if (phi == nullptr) {
    return report;
  }

  svmp::FieldHandle active_handle;
  if (mesh.has_field(svmp::EntityKind::Vertex, kActiveFluidVisualizationField)) {
    active_handle = mesh.field_handle(svmp::EntityKind::Vertex,
                                      kActiveFluidVisualizationField);
    if (mesh.field_type(active_handle) != svmp::FieldScalarType::Float64 ||
        mesh.field_components(active_handle) != 1u) {
      mesh.remove_field(active_handle);
      active_handle = mesh.attach_field(svmp::EntityKind::Vertex,
                                        kActiveFluidVisualizationField,
                                        svmp::FieldScalarType::Float64,
                                        1u);
    }
  } else {
    active_handle = mesh.attach_field(svmp::EntityKind::Vertex,
                                      kActiveFluidVisualizationField,
                                      svmp::FieldScalarType::Float64,
                                      1u);
  }
  auto* active = static_cast<double*>(mesh.field_data(active_handle));
  if (active == nullptr) {
    return report;
  }

  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    const bool is_active = activeSideContains(phi[v], request);
    active[v] = is_active ? 1.0 : 0.0;
    report.dry_vertices += is_active ? 0u : 1u;
    report.active_vertices += is_active ? 1u : 0u;
  }

  return report;
}

void applyCoupledLevelSetFieldResidualCriteria(
    const svmp::FE::systems::FESystem& system,
    const Parameters& params,
    svmp::FE::timestepping::NewtonOptions& options)
{
  for (auto* equation : params.equation_parameters) {
    if (equation == nullptr || !equation->type.defined() ||
        !equation->coupled.defined() || !equation->coupled.value() ||
        !equation->tolerance.defined() ||
        !(equation->tolerance.value() > 0.0)) {
      continue;
    }
    const auto type = normalized_token(equation->type.value());
    if (type != "levelset" && type != "levelsettransport") {
      continue;
    }

    std::string field_name{"level_set"};
    const auto equation_parameters = equation->get_parameter_list();
    if (const auto configured_name = first_defined_parameter(
            equation_parameters,
            {"Level_set_field_name",
             "Field_name"})) {
      field_name = trim_copy(*configured_name);
    }

    const auto field = system.findFieldByName(field_name);
    if (field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Coupled level-set nonlinear "
          "convergence could not find field '" +
          field_name + "'.");
    }
    if (!system.fieldParticipatesInUnknownVector(field)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Coupled level-set nonlinear "
          "convergence requires unknown field '" +
          field_name + "'.");
    }

    const double tolerance = equation->tolerance.value();
    // Preserve the equation-local residual floor.  The monolithic linear
    // solve separately uses the strictest requested coupled tolerance, but a
    // stricter tolerance on an unrelated equation must not silently tighten
    // this named nonlinear block.  SimulationBuilder uses the same 1e-10
    // legacy default when Absolute_tolerance is omitted.
    const double absolute_tolerance =
        equation->linear_solver.absolute_tolerance.defined()
            ? equation->linear_solver.absolute_tolerance.value()
            : 1.0e-10;
    const double equation_absolute_tolerance =
        absolute_tolerance > 0.0 && std::isfinite(absolute_tolerance)
            ? absolute_tolerance
            : 0.0;
    const double effective_absolute_tolerance =
        equation_absolute_tolerance;
    const auto existing = std::find_if(
        options.field_residual_criteria.begin(),
        options.field_residual_criteria.end(),
        [field](const auto& criterion) { return criterion.field == field; });
    if (existing == options.field_residual_criteria.end()) {
      options.field_residual_criteria.push_back(
          svmp::FE::timestepping::NewtonOptions::FieldResidualCriterion{
              .field = field,
              .abs_tolerance = effective_absolute_tolerance,
              .rel_tolerance = tolerance});
    } else {
      // Multiple equation declarations for the same transported field share
      // one residual block.  Retain the strictest enabled absolute and
      // relative targets instead of creating duplicate criteria that Newton
      // rejects.
      if (effective_absolute_tolerance > 0.0 &&
          (!(existing->abs_tolerance > 0.0) ||
           effective_absolute_tolerance < existing->abs_tolerance)) {
        existing->abs_tolerance = effective_absolute_tolerance;
      }
      if (!(existing->rel_tolerance > 0.0) ||
          tolerance < existing->rel_tolerance) {
        existing->rel_tolerance = tolerance;
      }
    }
  }
}

void logLevelSetMaintenanceCoverageDiagnostics(
    const std::vector<ActiveCutVolumeRequest>& active_requests,
    const std::vector<LevelSetMaintenanceRequest>& maintenance_requests)
{
  if (active_requests.empty()) {
    return;
  }

  std::set<std::string> transport_maintained_fields;
  for (const auto& request : maintenance_requests) {
    if (request.configuration->transport.kinematic_reconciliation.enabled ||
        request.configuration->transport.sign_definite_patch_bounds.enabled ||
        request.configuration->transport.interface_preserving_redistancing.enabled) {
      transport_maintained_fields.insert(request.configuration->transport.level_set.field_name);
    }
    // Map entry S9.138 (tests/cases/fluid/free_surface_log_record_map.json).
    const auto& transport = request.configuration->transport;
    svmp::FE::LogRecord configuration(
        "LevelSetMaintenancePipeline", "maintenance configuration");
    configuration.diagnostic("level_set_maintenance_configuration")
        .rootOnly()
        .quoted("field", transport.level_set.field_name)
        .text("kinematic_reconciliation",
              transport.kinematic_reconciliation.enabled ? "enabled" : "disabled")
        .text("sign_definite_patch_bounds",
              transport.sign_definite_patch_bounds.enabled ? "enabled" : "disabled");
    if (transport.interface_preserving_redistancing.enabled) {
      configuration.text("interface_preserving_redistancing", "enabled");
    }
    configuration.text(
        "conservation_diagnostic",
        svmp::FE::level_set::levelSetConservationDiagnosticName(
            svmp::FE::level_set::levelSetConservationDiagnostic(
                transport.transport_form)));
    svmp::FE::logRecord(configuration);
  }

  std::set<std::string> warned_fields;
  for (const auto& request : active_requests) {
    if (transport_maintained_fields.find(request.level_set_field_name) !=
        transport_maintained_fields.end()) {
      continue;
    }
    if (!warned_fields.insert(request.level_set_field_name).second) {
      continue;
    }
    // Map entry S9.182: a warning, so stderr.
    svmp::FE::logRecord(
        svmp::FE::LogRecord(
            "ApplicationDriver",
            "unfitted free-surface level set has no accepted-state maintenance",
            svmp::FE::LogLevel::WARNING)
            .diagnostic("plain_level_set_advection_not_conservative")
            .rootOnly()
            .quoted("field", request.level_set_field_name)
            .quoted("domain_id", request.domain_id)
            .text("active_side", activeSideName(request.active_side)));
  }
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution)
{
  std::vector<svmp::FE::Real> values(static_cast<std::size_t>(solution.size()), 0.0);
  auto view = solution.createAssemblyView();
  if (!view) {
    const auto span = solution.localSpan();
    if (span.size() != values.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Could not gather FE-ordered solution values.");
    }
    std::copy(span.begin(), span.end(), values.begin());
    return values;
  }

  constexpr svmp::FE::GlobalIndex chunk_size = 4096;
  std::vector<svmp::FE::GlobalIndex> dofs;
  std::vector<svmp::FE::Real> chunk_values;
  dofs.reserve(static_cast<std::size_t>(std::min(solution.size(), chunk_size)));
  chunk_values.reserve(dofs.capacity());
  for (svmp::FE::GlobalIndex offset = 0; offset < solution.size();
       offset += chunk_size) {
    const auto chunk =
        std::min<svmp::FE::GlobalIndex>(chunk_size, solution.size() - offset);
    dofs.resize(static_cast<std::size_t>(chunk));
    chunk_values.resize(static_cast<std::size_t>(chunk));
    for (svmp::FE::GlobalIndex i = 0; i < chunk; ++i) {
      dofs[static_cast<std::size_t>(i)] = offset + i;
    }
    view->getVectorEntries(
        std::span<const svmp::FE::GlobalIndex>(dofs.data(), dofs.size()),
        std::span<svmp::FE::Real>(chunk_values.data(), chunk_values.size()));
    std::copy(chunk_values.begin(), chunk_values.end(),
              values.begin() + static_cast<std::ptrdiff_t>(offset));
  }
  return values;
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    const svmp::FE::systems::SystemStateView& state)
{
  if (state.u_vector != nullptr) {
    auto* solution =
        const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
    return gatherFeOrderedSolution(*solution);
  }
  if (!state.u.empty()) {
    return std::vector<svmp::FE::Real>(state.u.begin(), state.u.end());
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Could not gather FE-ordered state values.");
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm);

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    const svmp::FE::systems::SystemStateView& state,
    const svmp::MeshComm& comm);

double globalMaxDouble(double local, const svmp::MeshComm& comm);
std::size_t globalSumSize(std::size_t local, const svmp::MeshComm& comm);

bool acceptedPressureUpdateDiagnosticEnabled()
{
  return parseBoolEnv("SVMP_ACTIVE_PRESSURE_UPDATE_DIAGNOSTIC", false) ||
         parseBoolEnv("SVMP_ACTIVE_PRESSURE_UPDATE_GUARD", false) ||
         parseBoolEnv("SVMP_ACTIVE_PRESSURE_UPDATE_REJECT_ON_TRIGGER", false) ||
         parseDoubleEnv("SVMP_ACTIVE_PRESSURE_UPDATE_THRESHOLD_PA", -1.0) >
             0.0;
}

bool activePressureUpdateRejectOnTriggerEnabled()
{
  return parseBoolEnv("SVMP_ACTIVE_PRESSURE_UPDATE_REJECT_ON_TRIGGER", false);
}

const char* activePressureUpdateSupportClass(double max_wet_fraction,
                                             double min_positive_wet_fraction)
{
  constexpr double tiny_fraction = 1.0e-4;
  constexpr double full_wet_tolerance = 1.0e-12;
  if (!std::isfinite(max_wet_fraction) || max_wet_fraction <= 0.0) {
    return "active_support_no_wet_fraction";
  }
  if (max_wet_fraction <= tiny_fraction) {
    return "tiny_cut_supported";
  }
  if (std::isfinite(min_positive_wet_fraction) &&
      min_positive_wet_fraction >= 1.0 - full_wet_tolerance) {
    return "full_wet_supported";
  }
  return "cut_supported";
}

bool logAcceptedPressureUpdateDiagnostic(
    const svmp::FE::systems::FESystem& system,
    const Parameters& params,
    std::span<const svmp::FE::Real> previous_solution,
    std::span<const svmp::FE::Real> current_solution,
    int step,
    double time,
    double dt,
    std::string_view phase,
    bool honor_fail_on_trigger)
{
  if (!acceptedPressureUpdateDiagnosticEnabled()) {
    return false;
  }
  if (previous_solution.size() != current_solution.size()) {
    svmp::FE::logRecord(
        svmp::FE::LogRecord("ApplicationDriver", "accepted pressure update diagnostic skipped", svmp::FE::LogLevel::WARNING)
            .diagnostic("accepted_pressure_update_guard_skipped")
            .text("reason", "solution_size_mismatch")
            .quoted("phase", phase)
            .value("previous_size", previous_solution.size())
            .value("current_size", current_solution.size())
            .rootOnly());
    return false;
  }

  const char* pressure_field_env =
      std::getenv("SVMP_ACTIVE_PRESSURE_UPDATE_FIELD");
  const std::string pressure_field_name =
      pressure_field_env != nullptr && *pressure_field_env != '\0'
          ? std::string(pressure_field_env)
          : std::string("Pressure");
  const auto pressure_field = system.findFieldByName(pressure_field_name);
  if (pressure_field == svmp::FE::INVALID_FIELD_ID) {
    svmp::FE::logRecord(
        svmp::FE::LogRecord("ApplicationDriver", "accepted pressure update diagnostic skipped", svmp::FE::LogLevel::WARNING)
            .diagnostic("accepted_pressure_update_guard_skipped")
            .text("reason", "pressure_field_not_found")
            .quoted("phase", phase)
            .quoted("field", pressure_field_name)
            .rootOnly());
    return false;
  }

  const auto& pressure_dofs = system.fieldDofHandler(pressure_field);
  const auto* pressure_entity_map = pressure_dofs.getEntityDofMap();
  if (pressure_entity_map == nullptr) {
    svmp::FE::logRecord(
        svmp::FE::LogRecord("ApplicationDriver", "accepted pressure update diagnostic skipped", svmp::FE::LogLevel::WARNING)
            .diagnostic("accepted_pressure_update_guard_skipped")
            .text("reason", "no_entity_dof_map")
            .quoted("phase", phase)
            .quoted("field", pressure_field_name)
            .rootOnly());
    return false;
  }

  const auto requests = activeCutVolumeRequests(params);
  const auto* cut_context = system.cutIntegrationContext();
  if (requests.empty() || cut_context == nullptr) {
    svmp::FE::logRecord(
        svmp::FE::LogRecord("ApplicationDriver", "accepted pressure update diagnostic skipped", svmp::FE::LogLevel::WARNING)
            .diagnostic("accepted_pressure_update_guard_skipped")
            .text("reason", "no_active_cut_context")
            .quoted("phase", phase)
            .value("active_requests", requests.size())
            .value("has_cut_context", (cut_context != nullptr ? 1 : 0))
            .rootOnly());
    return false;
  }

  const auto& mesh = system.meshAccess();
  cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh);
  const auto n_vertices = static_cast<std::size_t>(
      std::max<svmp::FE::GlobalIndex>(0, mesh.numVertices()));
  std::vector<unsigned char> active_support(n_vertices, 0u);
  std::vector<double> max_wet_fraction(
      n_vertices, -std::numeric_limits<double>::infinity());
  std::vector<double> min_positive_wet_fraction(
      n_vertices, std::numeric_limits<double>::infinity());
  std::vector<svmp::FE::GlobalIndex> cell_nodes;
  std::size_t retained_rule_count = 0u;

  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    const auto marker = generatedVolumeMarkerForRequest(*cut_context, request, i);
    if (!marker.has_value()) {
      continue;
    }
    const auto side = cutIntegrationSide(request.active_side);
    const auto rules =
        cut_context->generatedVolumeRulesForMarkerAndSide(*marker, side);
    for (const auto* rule : rules) {
      if (rule == nullptr || !std::isfinite(rule->volume_fraction) ||
          rule->volume_fraction <= svmp::FE::Real{0.0}) {
        continue;
      }
      const auto parent = rule->provenance.parent_entity;
      if (parent < 0 || parent >= mesh.numCells()) {
        continue;
      }
      ++retained_rule_count;
      cell_nodes.clear();
      mesh.getCellNodes(parent, cell_nodes);
      for (const auto vertex : cell_nodes) {
        if (vertex < 0 ||
            static_cast<std::size_t>(vertex) >= active_support.size()) {
          continue;
        }
        const auto v = static_cast<std::size_t>(vertex);
        active_support[v] = 1u;
        const auto fraction = static_cast<double>(rule->volume_fraction);
        max_wet_fraction[v] = std::max(max_wet_fraction[v], fraction);
        min_positive_wet_fraction[v] =
            std::min(min_positive_wet_fraction[v], fraction);
      }
    }
  }

  const auto pressure_offset = system.fieldDofOffset(pressure_field);
  const auto pressure_field_dofs = pressure_dofs.getNumDofs();
  const auto solution_size =
      static_cast<svmp::FE::GlobalIndex>(current_solution.size());
  if (pressure_offset < 0 ||
      pressure_offset + pressure_field_dofs > solution_size) {
    svmp::FE::logRecord(
        svmp::FE::LogRecord("ApplicationDriver", "accepted pressure update diagnostic skipped", svmp::FE::LogLevel::WARNING)
            .diagnostic("accepted_pressure_update_guard_skipped")
            .text("reason", "dofs_outside_solution")
            .quoted("phase", phase)
            .quoted("field", pressure_field_name)
            .value("pressure_offset", pressure_offset)
            .value("pressure_dofs", pressure_field_dofs)
            .value("solution_size", current_solution.size())
            .rootOnly());
    return false;
  }

  std::size_t supported_vertices = 0u;
  std::size_t compared_vertices = 0u;
  std::size_t skipped_nonvertex_pressure_dofs = 0u;
  double local_max_abs_delta = 0.0;
  double local_signed_delta = 0.0;
  svmp::FE::GlobalIndex local_vertex = -1;
  svmp::FE::GlobalIndex local_dof = -1;
  double local_from_pressure = 0.0;
  double local_to_pressure = 0.0;
  double local_max_wet = std::numeric_limits<double>::quiet_NaN();
  double local_min_positive_wet = std::numeric_limits<double>::quiet_NaN();

  for (std::size_t vertex = 0; vertex < active_support.size(); ++vertex) {
    if (active_support[vertex] == 0u) {
      continue;
    }
    ++supported_vertices;
    const auto vertex_dofs = pressure_entity_map->getVertexDofs(
        static_cast<svmp::FE::GlobalIndex>(vertex));
    if (vertex_dofs.size() != 1u) {
      ++skipped_nonvertex_pressure_dofs;
      continue;
    }
    const auto global_dof = pressure_offset + vertex_dofs.front();
    if (global_dof < 0 ||
        static_cast<std::size_t>(global_dof) >= current_solution.size()) {
      ++skipped_nonvertex_pressure_dofs;
      continue;
    }
    ++compared_vertices;
    const auto previous = static_cast<double>(
        previous_solution[static_cast<std::size_t>(global_dof)]);
    const auto current = static_cast<double>(
        current_solution[static_cast<std::size_t>(global_dof)]);
    const auto delta = current - previous;
    const auto abs_delta = std::abs(delta);
    if (abs_delta > local_max_abs_delta) {
      local_max_abs_delta = abs_delta;
      local_signed_delta = delta;
      local_vertex = static_cast<svmp::FE::GlobalIndex>(vertex);
      local_dof = global_dof;
      local_from_pressure = previous;
      local_to_pressure = current;
      local_max_wet = max_wet_fraction[vertex];
      local_min_positive_wet = min_positive_wet_fraction[vertex];
    }
  }

  const auto comm = activeFESystemCommunicator(system);
  const auto global_supported_vertices = globalSumSize(supported_vertices, comm);
  const auto global_compared_vertices = globalSumSize(compared_vertices, comm);
  const auto global_skipped_vertices =
      globalSumSize(skipped_nonvertex_pressure_dofs, comm);
  const auto global_retained_rules = globalSumSize(retained_rule_count, comm);
  const auto global_max_abs_delta = globalMaxDouble(local_max_abs_delta, comm);
  const auto threshold =
      parseDoubleEnv("SVMP_ACTIVE_PRESSURE_UPDATE_THRESHOLD_PA", -1.0);
  const bool triggered = threshold > 0.0 && global_max_abs_delta > threshold;

  svmp::FE::logRecord(
      svmp::FE::LogRecord("ApplicationDriver", "accepted pressure update")
          .diagnostic("accepted_pressure_update_guard")
          .quoted("phase", phase)
          .value("step", step)
          .value("time", time)
          .value("dt", dt)
          .quoted("field", pressure_field_name)
          .value("retained_active_volume_rules", global_retained_rules)
          .value("active_supported_vertices", global_supported_vertices)
          .value("compared_vertex_pressure_dofs", global_compared_vertices)
          .value("skipped_nonvertex_pressure_vertices", global_skipped_vertices)
          .value("local_worst_vertex", local_vertex)
          .value("local_worst_dof", local_dof)
          .value("local_abs_pressure_delta_pa", local_max_abs_delta)
          .value("global_abs_pressure_delta_pa", global_max_abs_delta)
          .value("local_pressure_delta_pa", local_signed_delta)
          .value("local_from_pressure_pa", local_from_pressure)
          .value("local_to_pressure_pa", local_to_pressure)
          .value("support_class", activePressureUpdateSupportClass(local_max_wet, local_min_positive_wet))
          .value("incident_wet_fraction_max", local_max_wet)
          .value("incident_wet_fraction_min_positive", local_min_positive_wet)
          .value("threshold_pa", threshold)
          .value("triggered", (triggered ? 1 : 0))
          .rootOnly());

  if (triggered &&
      honor_fail_on_trigger &&
      parseBoolEnv("SVMP_ACTIVE_PRESSURE_UPDATE_FAIL_ON_TRIGGER", false)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted active pressure update "
        "exceeded SVMP_ACTIVE_PRESSURE_UPDATE_THRESHOLD_PA.");
  }
  return triggered;
}

void scatterFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution,
    std::span<const svmp::FE::Real> values)
{
  svmp::FE::systems::scatterFeOrderedSolution(solution, values);
}

double globalMaxAbsDifference(std::span<const svmp::FE::Real> left,
                              std::span<const svmp::FE::Real> right,
                              const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveMaxAbsDifference(
      left, right, feCollectiveCommunicator(comm));
}

double globalSumDouble(double local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveSumDouble(
      local, feCollectiveCommunicator(comm));
}

double globalMinDouble(double local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveMinDouble(
      local, feCollectiveCommunicator(comm));
}

double globalMaxDouble(double local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveMaxDouble(
      local, feCollectiveCommunicator(comm));
}

std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveMinMaxUint64(
      local, feCollectiveCommunicator(comm));
}

std::pair<int, int> globalMinMaxInt(int local,
                                    const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    int minimum = 0;
    int maximum = 0;
    MPI_Allreduce(&local, &minimum, 1, MPI_INT, MPI_MIN, comm.native());
    MPI_Allreduce(&local, &maximum, 1, MPI_INT, MPI_MAX, comm.native());
    return {minimum, maximum};
  }
#else
  (void)comm;
#endif

  return {local, local};
}

using svmp::FE::level_set::requireActiveCutRequestConsensus;

std::size_t globalSumSize(std::size_t local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveSumSize(
      local, feCollectiveCommunicator(comm));
}

bool globalAnyBool(bool local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveAnyTrue(
      local, feCollectiveCommunicator(comm));
}

void requireCollectivePhasePreparation(
    const std::exception_ptr& local_failure,
    const svmp::MeshComm& comm,
    std::string_view phase)
{
  svmp::FE::systems::requireCollectivePhasePreparation(
      local_failure, feCollectiveCommunicator(comm), phase);
}

// The accepted contact stages are FE services: the operator-stage capture
// (FE/TimeStepping/OperatorStageCapture.h), the contact-line stage
// observables (FE/Interfaces/ContactLineStageObservables.h), the accepted
// wall-contact cells (FE/LevelSet/Maintenance/LevelSetWallAwareMaintenance.h)
// and the step participant (FE/LevelSet/LevelSetContactStageParticipant.h).
using svmp::FE::interfaces::firstOrderGeneralizedAlphaContactProvenanceValid;
using svmp::FE::interfaces::contactStageRealValuesMatch;
using svmp::FE::timestepping::DynamicContactOperatorStageFieldSlice;
using svmp::FE::timestepping::DynamicContactOperatorStageState;
using svmp::FE::timestepping::DynamicContactFirstOrderGeneralizedAlphaObservation;
using svmp::FE::timestepping::dynamicContactStageMeshRevisionMatches;
using svmp::FE::timestepping::dynamicContactWallBoundaryFingerprint;
using svmp::FE::timestepping::dynamicContactStageProvenanceMatches;
using svmp::FE::timestepping::makeDynamicContactFirstOrderGeneralizedAlphaObservation;
using svmp::FE::timestepping::captureDynamicContactOperatorStageState;
using svmp::FE::timestepping::overlayDynamicContactOperatorStageState;
using svmp::FE::timestepping::requireDynamicContactOperatorStageObservationConsensus;
using svmp::FE::timestepping::FreeSurfaceContactStageTemporalCoordinates;
using svmp::FE::timestepping::resolveFreeSurfaceContactStageTemporalCoordinates;
using svmp::FE::interfaces::acceptedContactStageRevision;
using svmp::FE::interfaces::evaluateAcceptedFreeSurfaceContactStages;
using svmp::FE::level_set::makeAcceptedSnapshotWallConstraint;
using svmp::FE::level_set::canonicalizeAcceptedWallConstraints;
using svmp::FE::level_set::captureAcceptedContactStageWallConstraints;
using svmp::FE::interfaces::bindAcceptedFreeSurfaceContactStagesToEndpointRevision;
using svmp::FE::level_set::assertAcceptedFreeSurfaceContactStageEndpointUnchanged;

std::shared_ptr<svmp::FE::geometry::GeometryMapping>
createCellGeometryMapping(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell);

std::uint64_t collectiveLevelSetMaintenanceAlgebraicRevision(
    std::span<const svmp::FE::Real> state,
    const svmp::MeshComm& comm);

// The accepted discrete functionals are Navier-Stokes free-surface physics
// (FE/LevelSet/AcceptedFunctionalRecorder.h).
using svmp::FE::level_set::acceptedFunctionalDeclarationContentKey;
using svmp::FE::level_set::currentAcceptedFunctionalLayouts;
using svmp::FE::level_set::evaluateDeclaredAcceptedFunctionals;
using svmp::FE::level_set::evaluateDeclaredFreeSurfaceFunctionals;
using svmp::FE::level_set::collectiveFieldRevision;
using svmp::FE::level_set::levelSetMaintenanceFunctionalValues;
using svmp::FE::level_set::AcceptedFunctionalCoverage;
using svmp::FE::level_set::requireCommunicatorConsistentAcceptedFunctionalCoverage;
using svmp::FE::level_set::recordAcceptedFunctionals;

// The accepted-stage diagnostics of the two-fluid interface are an FE
// service (FE/LevelSet/TwoFluidAcceptedStage.h);
// buildCurrentTwoFluidMomentumReconciliations of the driver's maintenance
// requests follows ConservativePhaseRequestViews below.
using svmp::FE::level_set::evaluateCurrentTwoFluidStageDiagnostics;
using svmp::FE::level_set::TwoFluidMomentumReconciliationBatch;
using svmp::FE::level_set::makeTwoFluidAcceptedStageSolveTelemetry;

// The algebraic state revision is in
// FE/LevelSet/Maintenance/LevelSetMaintenanceRequestSchedule.h.
using svmp::FE::level_set::levelSetMaintenanceAlgebraicRevision;

std::uint64_t collectiveLevelSetMaintenanceAlgebraicRevision(
    std::span<const svmp::FE::Real> state,
    const svmp::MeshComm& comm)
{
  return svmp::FE::level_set::collectiveLevelSetMaintenanceAlgebraicRevision(
      state, feCollectiveCommunicator(comm));
}

std::uint64_t avalancheMeshBoundaryStateWord(
    std::uint64_t value) noexcept
{
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

std::uint64_t meshBoundaryExpectedOwnedIndexSum(
    std::uint64_t size) noexcept
{
  return size % 2u == 0u
      ? (size / 2u) * (size - (size == 0u ? 0u : 1u))
      : size * ((size - 1u) / 2u);
}

std::uint64_t meshBoundaryExpectedOwnedIndexXor(
    std::uint64_t size) noexcept
{
  if (size == 0u) {
    return 0u;
  }
  const auto last = size - 1u;
  switch (last & 3u) {
    case 0u:
      return last;
    case 1u:
      return 1u;
    case 2u:
      return last + 1u;
    default:
      return 0u;
  }
}

std::uint64_t collectiveMeshBoundaryStateFingerprint(
    const svmp::FE::backends::GenericVector& state,
    const svmp::MeshComm& comm)
{
  static_assert(
      sizeof(svmp::FE::Real) == sizeof(std::uint64_t),
      "Mesh-boundary state history requires a 64-bit FE scalar.");

  std::array<std::uint64_t, 5> local_sums{{0u, 0u, 0u, 0u, 0u}};
  std::array<std::uint64_t, 4> local_xors{{0u, 0u, 0u, 0u}};
  std::vector<svmp::FE::GlobalIndex> owned_dofs;
  std::exception_ptr local_fingerprint_failure;
  try {
    if (state.size() < 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Mesh-boundary algebraic state has a negative global size.");
    }
    owned_dofs = state.ownedGlobalRows();
    if (!std::is_sorted(owned_dofs.begin(), owned_dofs.end()) ||
        std::adjacent_find(owned_dofs.begin(), owned_dofs.end()) !=
            owned_dofs.end()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Mesh-boundary algebraic-state backend ownership is not strictly sorted and unique.");
    }

    // Backend read views may refresh internal caches but do not alter the
    // represented algebraic values. Keep that logical mutation contained so
    // candidate-stage snapshots remain const to every caller.
    auto view = const_cast<svmp::FE::backends::GenericVector&>(state)
                    .createAssemblyView();
    if (!view) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Mesh-boundary algebraic state has no readable assembly view.");
    }

    constexpr std::size_t chunk_capacity = 4096u;
    std::array<svmp::FE::Real, chunk_capacity> values{};
    for (std::size_t offset = 0u; offset < owned_dofs.size();
         offset += chunk_capacity) {
      const auto count =
          std::min(chunk_capacity, owned_dofs.size() - offset);
      const auto dofs = std::span<const svmp::FE::GlobalIndex>(
          owned_dofs.data() + static_cast<std::ptrdiff_t>(offset), count);
      const auto chunk =
          std::span<svmp::FE::Real>(values.data(), count);
      for (const auto dof : dofs) {
        if (dof < 0 || dof >= state.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Mesh-boundary algebraic-state ownership contains an out-of-range row.");
        }
      }
      view->getVectorEntries(dofs, chunk);
      for (std::size_t i = 0u; i < count; ++i) {
        const auto dof = static_cast<std::uint64_t>(dofs[i]);
        const auto bits = std::bit_cast<std::uint64_t>(chunk[i]);
        const auto first = avalancheMeshBoundaryStateWord(
            bits ^ avalancheMeshBoundaryStateWord(
                       dof ^ 0x243f6a8885a308d3ull));
        const auto second = avalancheMeshBoundaryStateWord(
            std::rotl(bits, 23) ^
            avalancheMeshBoundaryStateWord(
                dof ^ 0x13198a2e03707344ull));
        const auto third = avalancheMeshBoundaryStateWord(
            first + std::rotl(second, 31) +
            0xa4093822299f31d0ull);
        ++local_sums[0];
        local_sums[1] += dof;
        local_sums[2] += first;
        local_sums[3] += second;
        local_sums[4] += third;
        local_xors[0] ^= dof;
        local_xors[1] ^= first;
        local_xors[2] ^= second;
        local_xors[3] ^= third;
      }
    }
  } catch (...) {
    local_fingerprint_failure = std::current_exception();
  }

  if (globalAnyBool(local_fingerprint_failure != nullptr, comm)) {
    if (local_fingerprint_failure != nullptr) {
      std::rethrow_exception(local_fingerprint_failure);
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Another active FE communicator rank could not fingerprint its backend-owned mesh-boundary algebraic-state rows.");
  }

  const auto global_size = static_cast<std::uint64_t>(state.size());
  const auto [minimum_global_size, maximum_global_size] =
      globalMinMaxUint64(global_size, comm);
  if (minimum_global_size != maximum_global_size) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Mesh-boundary algebraic-state size differs across the active FE communicator.");
  }

  constexpr std::size_t occupancy_capacity = 65536u;
  std::vector<std::uint32_t> local_occupancy;
  std::vector<std::uint32_t> global_occupancy;
  std::exception_ptr local_occupancy_failure;
  try {
    const auto capacity = static_cast<std::size_t>(
        std::min<std::uint64_t>(global_size, occupancy_capacity));
    local_occupancy.resize(capacity, 0u);
    global_occupancy.resize(capacity, 0u);
  } catch (...) {
    local_occupancy_failure = std::current_exception();
  }
  if (globalAnyBool(local_occupancy_failure != nullptr, comm)) {
    if (local_occupancy_failure != nullptr) {
      std::rethrow_exception(local_occupancy_failure);
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Another active FE communicator rank could not allocate bounded ownership-validation storage.");
  }

  std::size_t owned_cursor = 0u;
  for (std::uint64_t chunk_begin = 0u; chunk_begin < global_size;
       chunk_begin += occupancy_capacity) {
    const auto chunk_size = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            occupancy_capacity, global_size - chunk_begin));
    std::fill_n(local_occupancy.begin(), chunk_size, 0u);
    const auto chunk_end = chunk_begin + chunk_size;
    while (owned_cursor < owned_dofs.size() &&
           static_cast<std::uint64_t>(owned_dofs[owned_cursor]) <
               chunk_end) {
      const auto row =
          static_cast<std::uint64_t>(owned_dofs[owned_cursor]);
      local_occupancy[static_cast<std::size_t>(row - chunk_begin)] = 1u;
      ++owned_cursor;
    }
    std::copy_n(
        local_occupancy.begin(), chunk_size, global_occupancy.begin());
#ifdef MESH_HAS_MPI
    if (comm.is_parallel()) {
#ifdef MPI_UINT32_T
      const MPI_Datatype occupancy_type = MPI_UINT32_T;
#else
      const MPI_Datatype occupancy_type = MPI_UNSIGNED;
#endif
      MPI_Allreduce(
          local_occupancy.data(),
          global_occupancy.data(),
          static_cast<int>(chunk_size),
          occupancy_type,
          MPI_SUM,
          comm.native());
    }
#else
    if (comm.is_parallel()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Distributed mesh-boundary algebraic-state ownership validation requires MPI.");
    }
#endif
    if (std::any_of(
            global_occupancy.begin(),
            global_occupancy.begin() +
                static_cast<std::ptrdiff_t>(chunk_size),
            [](std::uint32_t count) { return count != 1u; })) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Mesh-boundary algebraic-state backend ownership does not cover every global row exactly once.");
    }
  }
  const bool local_cursor_incomplete = owned_cursor != owned_dofs.size();
  if (globalAnyBool(local_cursor_incomplete, comm)) {
    if (local_cursor_incomplete) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Mesh-boundary algebraic-state ownership validation did not consume every local row.");
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Another active FE communicator rank did not consume every backend-owned row.");
  }

  auto global_sums = local_sums;
  auto global_xors = local_xors;
#ifdef MESH_HAS_MPI
  if (comm.is_parallel()) {
#ifdef MPI_UINT64_T
    const MPI_Datatype revision_type = MPI_UINT64_T;
#else
    const MPI_Datatype revision_type = MPI_UNSIGNED_LONG_LONG;
#endif
    MPI_Allreduce(
        local_sums.data(),
        global_sums.data(),
        static_cast<int>(local_sums.size()),
        revision_type,
        MPI_SUM,
        comm.native());
    MPI_Allreduce(
        local_xors.data(),
        global_xors.data(),
        static_cast<int>(local_xors.size()),
        revision_type,
        MPI_BXOR,
        comm.native());
  }
#else
  if (comm.is_parallel()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Distributed mesh-boundary algebraic-state fingerprinting requires MPI.");
  }
#endif

  if (global_sums[0] != global_size ||
      global_sums[1] != meshBoundaryExpectedOwnedIndexSum(global_size) ||
      global_xors[0] != meshBoundaryExpectedOwnedIndexXor(global_size)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Mesh-boundary algebraic-state ownership fingerprint is inconsistent with the certified global row range.");
  }

  std::uint64_t revision = kCutContextHashOffset;
  mixCutContextHash(revision, global_size);
  for (const auto word : global_sums) {
    mixCutContextHash(revision, word);
  }
  for (const auto word : global_xors) {
    mixCutContextHash(revision, word);
  }
  return revision == 0u ? 1u : revision;
}

void recordAcceptedMeshBoundaryProvenanceAtEndpoint(
    application::core::SimulationComponents& sim,
    std::uint64_t accepted_step,
    svmp::FE::Real accepted_time,
    svmp::FE::Real dt)
{
  auto& system = *sim.fe_system;
  const auto comm = activeFESystemCommunicator(system);
  const auto normal_declarations =
      system.meshNormalBoundaryConstraints();
  const auto tangential_declarations =
      system.meshTangentialBoundaryPolicies();
  const auto [minimum_normal_count, maximum_normal_count] =
      globalMinMaxUint64(
          static_cast<std::uint64_t>(normal_declarations.size()),
          comm);
  const auto [minimum_tangential_count, maximum_tangential_count] =
      globalMinMaxUint64(
          static_cast<std::uint64_t>(tangential_declarations.size()),
          comm);
  if (minimum_normal_count != maximum_normal_count ||
      minimum_tangential_count != maximum_tangential_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Mesh normal or tangential boundary "
        "history declaration count differs across the active FE "
        "communicator.");
  }
  if (maximum_normal_count == 0u &&
      maximum_tangential_count == 0u) {
    return;
  }

  const auto state_fingerprint =
      collectiveMeshBoundaryStateFingerprint(
          sim.time_history->u(), comm);
  system.recordAcceptedMeshBoundaryProvenance(
      accepted_step,
      accepted_time,
      dt,
      state_fingerprint);
}

void reimposeAcceptedMasterBearingState(
    svmp::FE::timestepping::TimeHistory& history,
    const svmp::FE::systems::FESystem& system)
{
  // TimeLoop repeats this operation after on_step_accepted returns. Apply it
  // here as soon as accepted maintenance has refreshed the constraint set so
  // mesh-boundary provenance and all later endpoint artifacts observe the
  // same constraint-consistent values. The repeated operation is idempotent.
  if (parseBoolEnv("SVMP_NO_MPC_STATE_DISTRIBUTE", false)) {
    return;
  }
  const auto& constraints = system.constraints();
  // Constraint lines are rank-local (a partition that does not touch the
  // interface holds no master-bearing line), while updateGhosts() below is
  // collective: decide on every rank of the FE communicator together.
  if (!globalAnyBool(!constraints.empty() &&
                         constraints.hasMasterBearingLines(),
                     activeFESystemCommunicator(system))) {
    return;
  }
  history.updateGhosts();
  constraints.distributeMasterBearing(history.u());
  constraints.distributeMasterBearing(history.uPrev());
  if (history.hasUDotState()) {
    constraints.distributeMasterBearingHomogeneous(history.uDot());
  }
  if (history.hasUDDotState()) {
    constraints.distributeMasterBearingHomogeneous(history.uDDot());
  }
}

using svmp::FE::level_set::recordInitialAcceptedFunctionalBaseline;

// The accepted-step maintenance pipeline is in
// FE/LevelSet/Maintenance/LevelSetMaintenancePipeline.h.  The driver passes
// the FE part of its requests (the configuration handle and a copy of the FE
// run state) and copies the run state back.
using svmp::FE::level_set::LevelSetMaintenanceStageObserver;

svmp::FE::level_set::LevelSetMaintenanceHost levelSetMaintenanceHost(
    const application::core::SimulationComponents& sim)
{
  svmp::FE::level_set::LevelSetMaintenanceHost host;
  host.fe_system = sim.fe_system.get();
  return host;
}

// The accepted wall-contact cells of one maintained level-set field (the
// conservative-phase reconciliation leaves them unchanged), and the accepted
// dynamic contact stage they were taken from, if any.
struct LevelSetWallAwareMaintenanceContext {
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
      local_constraints{};
  svmp::FE::Real stage_time{0.0};
  svmp::FE::Real stage_alpha_f{1.0};
  std::optional<svmp::FE::systems::
                    FreeSurfaceFirstOrderGeneralizedAlphaProvenance>
      first_order_generalized_alpha{};
  std::uint64_t previous_state_revision{0u};
  std::uint64_t endpoint_state_revision{0u};
  std::uint64_t stage_state_revision{0u};
};

LevelSetWallAwareMaintenanceContext
resolveLevelSetWallAwareMaintenanceContext(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    svmp::FE::FieldId level_set_field,
    svmp::FE::Real endpoint_time,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        accepted_contact_stages,
    std::span<const svmp::FE::level_set::LevelSetWallContactConstraint>
        accepted_contact_stage_constraints)
{
  LevelSetWallAwareMaintenanceContext resolved;
  const auto declarations =
      svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  std::uint64_t declaration_signature = 1469598103934665603ull;
  const auto mix_declaration_value = [&declaration_signature](
                                         std::uint64_t value) {
    declaration_signature ^= value;
    declaration_signature *= 1099511628211ull;
  };
  const auto mix_declaration_real = [&](svmp::FE::Real value) {
    static_assert(sizeof(value) <= sizeof(std::uint64_t));
    std::uint64_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(value));
    mix_declaration_value(bits);
  };
  mix_declaration_value(static_cast<std::uint64_t>(declarations.size()));
  for (const auto& declaration : declarations) {
    mix_declaration_value(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(declaration.level_set_field)));
    mix_declaration_value(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(declaration.interface_marker)));
    mix_declaration_value(static_cast<std::uint64_t>(
        declaration.parameters.liquid_side));
    mix_declaration_value(static_cast<std::uint64_t>(
        declaration.capillary_balance_method));
    mix_declaration_value(static_cast<std::uint64_t>(
        declaration.capillary_balance_qualification));
    mix_declaration_value(static_cast<std::uint64_t>(
        declaration.parameters.young_wall_coefficients.size()));
    for (const auto& coefficient :
         declaration.parameters.young_wall_coefficients) {
      mix_declaration_value(static_cast<std::uint64_t>(
          static_cast<std::int64_t>(coefficient.boundary_marker)));
      mix_declaration_real(
          coefficient.equilibrium_contact_angle_radians);
    }
    mix_declaration_value(static_cast<std::uint64_t>(
        declaration.parameters.dynamic_contact_coefficients.size()));
    for (const auto& coefficient :
         declaration.parameters.dynamic_contact_coefficients) {
      mix_declaration_value(static_cast<std::uint64_t>(
          static_cast<std::int64_t>(coefficient.boundary_marker)));
      mix_declaration_real(
          coefficient.equilibrium_contact_angle_radians);
      mix_declaration_value(static_cast<std::uint64_t>(coefficient.law));
      mix_declaration_real(coefficient.mobility);
      mix_declaration_real(coefficient.slip_length);
      mix_declaration_real(coefficient.dynamic_viscosity);
    }
  }
  const auto [minimum_declaration_signature,
              maximum_declaration_signature] =
      globalMinMaxUint64(declaration_signature, comm);
  if (minimum_declaration_signature != maximum_declaration_signature) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set maintenance declarations differ across the FE communicator.");
  }
  const bool has_matching_wall_law = std::any_of(
      declarations.begin(), declarations.end(), [&](const auto& declaration) {
        return declaration.level_set_field == level_set_field &&
               (!declaration.parameters.young_wall_coefficients.empty() ||
                !declaration.parameters.dynamic_contact_coefficients.empty());
      });
  if (globalMinDouble(has_matching_wall_law ? 1.0 : 0.0, comm) !=
      globalMaxDouble(has_matching_wall_law ? 1.0 : 0.0, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set maintenance field coverage differs across the FE communicator.");
  }
  if (!has_matching_wall_law) {
    return resolved;
  }

  const double local_stage_count =
      static_cast<double>(accepted_contact_stages.size());
  if (globalMinDouble(local_stage_count, comm) !=
      globalMaxDouble(local_stage_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set maintenance received inconsistent accepted contact-stage coverage.");
  }
  std::map<
      int,
      const svmp::FE::systems::FreeSurfaceAcceptedContactStageState*>
      stage_by_interface;
  bool local_stage_map_valid = true;
  for (const auto& stage : accepted_contact_stages) {
    const int marker = stage.geometry_revision.interface_marker;
    local_stage_map_valid =
        marker >= 0 && stage_by_interface.emplace(marker, &stage).second &&
        local_stage_map_valid;
  }
  if (globalMinDouble(local_stage_map_valid ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set maintenance requires unique accepted contact stages with nonnegative interface markers.");
  }

  // Decision D4: the momentum-side Young term is the only owner of the
  // contact angle, so a constraint carries only the identity of an accepted
  // contact cell, on a prescribed or a dynamic wall alike.  Walls differ only
  // in which accepted state supplies their contact rules.  The host validates
  // its declarations and accepted stages and states the walls of the field;
  // FE resolves the geometry (resolveLevelSetWallContactConstraints).
  using svmp::FE::level_set::LevelSetWallContactSource;
  svmp::FE::level_set::LevelSetWallAwareMaintenanceInput input;
  bool dynamic_stage_initialized = false;
  for (const auto& declaration : declarations) {
    if (declaration.level_set_field != level_set_field ||
        (declaration.parameters.young_wall_coefficients.empty() &&
         declaration.parameters.dynamic_contact_coefficients.empty())) {
      continue;
    }
    std::map<int, LevelSetWallContactSource> contact_source_by_wall;
    for (const auto& coefficient :
         declaration.parameters.young_wall_coefficients) {
      contact_source_by_wall.emplace(
          coefficient.boundary_marker,
          LevelSetWallContactSource::EndpointSnapshot);
    }
    for (const auto& coefficient :
         declaration.parameters.dynamic_contact_coefficients) {
      if (coefficient.law !=
          svmp::FE::interfaces::FreeSurfaceContactLaw::DynamicRenE) {
        continue;
      }
      // A dynamic law supersedes the equilibrium Young datum on the same
      // wall.  The latter remains the momentum-side energy coefficient; the
      // accepted stage owns the redistancing geometry.
      contact_source_by_wall[coefficient.boundary_marker] =
          LevelSetWallContactSource::AcceptedStage;
    }
    auto& field_interface = input.interfaces.emplace_back();
    field_interface.interface_marker = declaration.interface_marker;
    if (contact_source_by_wall.empty()) {
      continue;
    }
    std::map<int, bool> owned_contact_by_wall;

    const bool has_dynamic_contact_law = std::any_of(
        declaration.parameters.dynamic_contact_coefficients.begin(),
        declaration.parameters.dynamic_contact_coefficients.end(),
        [](const auto& coefficient) {
          return coefficient.law ==
                 svmp::FE::interfaces::FreeSurfaceContactLaw::DynamicRenE;
        });
    if (has_dynamic_contact_law) {
      const auto stage_it =
          stage_by_interface.find(declaration.interface_marker);
      const auto* accepted_dynamic_stage =
          stage_it != stage_by_interface.end() ? stage_it->second : nullptr;
      const auto expected_stage_time =
          endpoint_time -
          (svmp::FE::Real{1.0} -
           (accepted_dynamic_stage != nullptr
                ? accepted_dynamic_stage->stage_alpha_f
                : svmp::FE::Real{0.0})) *
              static_cast<svmp::FE::Real>(history.dt());
      const auto stage_time_tolerance =
          svmp::FE::Real{256.0} *
          std::numeric_limits<svmp::FE::Real>::epsilon() *
          std::max(
              {svmp::FE::Real{1.0},
               std::abs(expected_stage_time),
               accepted_dynamic_stage != nullptr
                   ? std::abs(accepted_dynamic_stage->stage_time)
                   : svmp::FE::Real{0.0}});
      const bool local_temporal_provenance_valid =
          accepted_dynamic_stage != nullptr &&
          (!accepted_dynamic_stage->first_order_generalized_alpha.has_value()
               ? accepted_dynamic_stage->stage_alpha_f ==
                     svmp::FE::Real{1.0}
               : firstOrderGeneralizedAlphaContactProvenanceValid(
                     *accepted_dynamic_stage
                          ->first_order_generalized_alpha) &&
                     contactStageRealValuesMatch(
                         accepted_dynamic_stage->stage_alpha_f,
                         accepted_dynamic_stage
                             ->first_order_generalized_alpha->alpha_f) &&
                     contactStageRealValuesMatch(
                         static_cast<svmp::FE::Real>(history.dt()),
                         accepted_dynamic_stage
                             ->first_order_generalized_alpha->dt));
      const bool local_stage_valid =
          accepted_dynamic_stage != nullptr &&
          std::isfinite(accepted_dynamic_stage->stage_time) &&
          std::isfinite(accepted_dynamic_stage->stage_alpha_f) &&
          accepted_dynamic_stage->stage_alpha_f > svmp::FE::Real{0.0} &&
          accepted_dynamic_stage->stage_alpha_f <= svmp::FE::Real{1.0} &&
          accepted_dynamic_stage->previous_state_revision != 0u &&
          accepted_dynamic_stage->endpoint_state_revision != 0u &&
          accepted_dynamic_stage->stage_state_revision != 0u &&
          accepted_dynamic_stage->geometry_revision.complete() &&
          accepted_dynamic_stage->geometry_revision.interface_marker ==
              declaration.interface_marker &&
          accepted_dynamic_stage->state.snapshot_revision_key ==
              accepted_dynamic_stage->geometry_revision
                  .snapshot_revision_key &&
          local_temporal_provenance_valid &&
          std::abs(accepted_dynamic_stage->stage_time -
                   expected_stage_time) <= stage_time_tolerance;
      if (globalMinDouble(local_stage_valid ? 1.0 : 0.0, comm) != 1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wall-aware level-set maintenance rejected incomplete, stale, or time-inconsistent accepted dynamic contact provenance for interface marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
      const auto [minimum_stage_revision, maximum_stage_revision] =
          globalMinMaxUint64(
              accepted_dynamic_stage->stage_state_revision, comm);
      const auto [minimum_previous_revision, maximum_previous_revision] =
          globalMinMaxUint64(
              accepted_dynamic_stage->previous_state_revision, comm);
      const auto [minimum_endpoint_revision, maximum_endpoint_revision] =
          globalMinMaxUint64(
              accepted_dynamic_stage->endpoint_state_revision, comm);
      const auto minimum_stage_time = globalMinDouble(
          static_cast<double>(accepted_dynamic_stage->stage_time), comm);
      const auto maximum_stage_time = globalMaxDouble(
          static_cast<double>(accepted_dynamic_stage->stage_time), comm);
      const auto minimum_stage_alpha = globalMinDouble(
          static_cast<double>(accepted_dynamic_stage->stage_alpha_f), comm);
      const auto maximum_stage_alpha = globalMaxDouble(
          static_cast<double>(accepted_dynamic_stage->stage_alpha_f), comm);
      const auto [minimum_generalized_alpha_present,
                  maximum_generalized_alpha_present] =
          globalMinMaxUint64(
              accepted_dynamic_stage->first_order_generalized_alpha.has_value()
                  ? 1u
                  : 0u,
              comm);
      if (minimum_stage_revision != maximum_stage_revision ||
          minimum_previous_revision != maximum_previous_revision ||
          minimum_endpoint_revision != maximum_endpoint_revision ||
          minimum_stage_time != maximum_stage_time ||
          minimum_stage_alpha != maximum_stage_alpha ||
          minimum_generalized_alpha_present !=
              maximum_generalized_alpha_present) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wall-aware level-set maintenance received different accepted dynamic stages across ranks for interface marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
      if (maximum_generalized_alpha_present != 0u) {
        const auto& provenance =
            *accepted_dynamic_stage->first_order_generalized_alpha;
        const std::array<double, 4> local_parameters{{
            static_cast<double>(provenance.alpha_m),
            static_cast<double>(provenance.alpha_f),
            static_cast<double>(provenance.gamma),
            static_cast<double>(provenance.dt),
        }};
        for (const auto parameter : local_parameters) {
          if (globalMinDouble(parameter, comm) !=
              globalMaxDouble(parameter, comm)) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Wall-aware level-set maintenance received different generalized-alpha contact-stage parameters across ranks for interface marker " +
                std::to_string(declaration.interface_marker) + ".");
          }
        }
      }
      if (!dynamic_stage_initialized) {
        resolved.stage_time = accepted_dynamic_stage->stage_time;
        resolved.stage_alpha_f = accepted_dynamic_stage->stage_alpha_f;
        resolved.first_order_generalized_alpha =
            accepted_dynamic_stage->first_order_generalized_alpha;
        resolved.previous_state_revision =
            accepted_dynamic_stage->previous_state_revision;
        resolved.endpoint_state_revision =
            accepted_dynamic_stage->endpoint_state_revision;
        resolved.stage_state_revision =
            accepted_dynamic_stage->stage_state_revision;
        dynamic_stage_initialized = true;
      } else {
        const bool local_same_stage_provenance =
            resolved.stage_time == accepted_dynamic_stage->stage_time &&
            resolved.stage_alpha_f ==
                accepted_dynamic_stage->stage_alpha_f &&
            resolved.first_order_generalized_alpha ==
                accepted_dynamic_stage->first_order_generalized_alpha &&
            resolved.previous_state_revision ==
                accepted_dynamic_stage->previous_state_revision &&
            resolved.endpoint_state_revision ==
                accepted_dynamic_stage->endpoint_state_revision;
        if (globalMinDouble(local_same_stage_provenance ? 1.0 : 0.0, comm) !=
            1.0) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] One maintained level-set field cannot use multiple accepted dynamic contact stages.");
        }
      }
      field_interface.accepted_stage_geometry_revision =
          accepted_dynamic_stage->geometry_revision.snapshot_revision_key;
      for (const auto& [boundary_marker, source] : contact_source_by_wall) {
        if (source != LevelSetWallContactSource::AcceptedStage) {
          continue;
        }
        const int dynamic_boundary_marker = boundary_marker;
        const auto wall_count = static_cast<std::size_t>(std::count_if(
            accepted_dynamic_stage->state.walls.begin(),
            accepted_dynamic_stage->state.walls.end(),
            [dynamic_boundary_marker](const auto& wall) {
              return wall.boundary_marker == dynamic_boundary_marker;
            }));
        const auto wall_it = std::find_if(
            accepted_dynamic_stage->state.walls.begin(),
            accepted_dynamic_stage->state.walls.end(),
            [dynamic_boundary_marker](const auto& wall) {
              return wall.boundary_marker == dynamic_boundary_marker;
            });
        const bool local_wall_state_valid =
            wall_count == 1u &&
            wall_it != accepted_dynamic_stage->state.walls.end() &&
            std::isfinite(wall_it->owned_contact_measure) &&
            wall_it->owned_contact_measure >= svmp::FE::Real{0.0};
        if (globalMinDouble(local_wall_state_valid ? 1.0 : 0.0, comm) !=
            1.0) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Wall-aware level-set maintenance is missing the accepted dynamic wall state for boundary marker " +
              std::to_string(boundary_marker) + ".");
        }
        owned_contact_by_wall[boundary_marker] =
            wall_it->owned_quadrature_point_count > 0u ||
            wall_it->owned_contact_measure > svmp::FE::Real{0.0};
      }
    }
    for (const auto& [boundary_marker, source] : contact_source_by_wall) {
      field_interface.walls.push_back(
          svmp::FE::level_set::LevelSetWallAwareWall{
              .boundary_marker = boundary_marker,
              .source = source,
              .accepted_stage_has_owned_contact =
                  source == LevelSetWallContactSource::AcceptedStage &&
                  owned_contact_by_wall[boundary_marker],
          });
    }
  }
  resolved.local_constraints =
      svmp::FE::level_set::resolveLevelSetWallContactConstraints(
          *sim.fe_system, input, accepted_contact_stage_constraints);
  return resolved;
}

void assignPostacceptMaintenanceVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    std::span<const svmp::FE::Real> values,
    const svmp::MeshComm& comm);

std::vector<svmp::FE::Real>
captureFeOrderedVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm);

std::vector<svmp::FE::Real>
capturePostacceptMaintenanceVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm);

using svmp::FE::level_set::LevelSetKinematicBaseline;
using svmp::FE::level_set::LevelSetMaintenanceCandidateValidator;

LevelSetKinematicBaseline makeLevelSetKinematicBaseline(
    const svmp::FE::systems::FESystem& system,
    const LevelSetMaintenanceRequest& request,
    std::span<const svmp::FE::Real> fe_solution,
    int step)
{
  return svmp::FE::level_set::makeLevelSetKinematicBaseline(
      system, *request.configuration, fe_solution, step);
}

bool applyLevelSetMaintenance(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    std::vector<LevelSetMaintenanceRequest>& requests,
    // The accepted contact-stage inputs fed the retired projection
    // reinitialization (decision D47); the slots stay while the driver keeps
    // its call sites.
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        /*accepted_contact_stages*/ = {},
    std::span<const svmp::FE::level_set::LevelSetWallContactConstraint>
        /*accepted_contact_stage_constraints*/ = {},
    std::span<const svmp::FE::Real> /*accepted_contact_stage_solution*/ = {},
    const LevelSetMaintenanceCandidateValidator& validate_candidate = {},
    const LevelSetMaintenanceStageObserver& observe_stage = {},
    std::vector<svmp::FE::LogRecord>* deferred_commit_logs = nullptr,
    bool* rate_publication_started = nullptr,
    std::span<const LevelSetKinematicBaseline> kinematic_baselines = {})
{
  auto fe_requests = feLevelSetMaintenanceRequests(requests);
  const bool changed = svmp::FE::level_set::applyLevelSetMaintenance(
      levelSetMaintenanceHost(sim),
      history,
      fe_requests,
      validate_candidate,
      observe_stage,
      deferred_commit_logs,
      rate_publication_started,
      kinematic_baselines);
  // The FE pipeline updates the run state only when it succeeds.
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    static_cast<svmp::FE::level_set::LevelSetMaintenanceRuntime&>(
        requests[index].runtime) = fe_requests[index].runtime;
  }
  return changed;
}

svmp::FE::systems::SystemStateView stateViewForHistory(
    svmp::FE::timestepping::TimeHistory& history)
{
  svmp::FE::systems::SystemStateView state;
  state.time = history.time();
  state.dt = history.dt();
  state.dt_prev = history.dtPrev();
  state.u = history.uSpan();
  state.u_prev = history.uPrevSpan();
  state.u_prev2 = history.uPrev2Span();
  state.u_vector = &history.u();
  state.u_prev_vector = &history.uPrev();
  state.u_prev2_vector = &history.uPrev2();
  state.u_history = history.uHistorySpans();
  state.dt_history = history.dtHistory();
  return state;
}

using svmp::FE::level_set::evaluateVertexField;

std::shared_ptr<svmp::FE::geometry::GeometryMapping> createCellGeometryMapping(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell)
{
  if (cell < 0 || cell >= mesh.numCells()) {
    return nullptr;
  }

  std::vector<std::array<svmp::FE::Real, 3>> coords;
  mesh.getCellCoordinates(cell, coords);
  if (coords.empty()) {
    return nullptr;
  }

  std::vector<svmp::FE::math::Vector<svmp::FE::Real, 3>> nodes;
  nodes.reserve(coords.size());
  for (const auto& coord : coords) {
    svmp::FE::math::Vector<svmp::FE::Real, 3> node{};
    node[0] = coord[0];
    node[1] = coord[1];
    node[2] = coord[2];
    nodes.push_back(node);
  }

  svmp::FE::geometry::MappingRequest map_request;
  map_request.element_type = mesh.getCellType(cell);
  map_request.geometry_order = mesh.getCellGeometryOrder(cell);
  map_request.use_affine = map_request.geometry_order <= 1;
  return svmp::FE::geometry::MappingFactory::create(map_request, nodes);
}

std::optional<int> generatedCutContextMarkerForMaintenance(
    const svmp::FE::systems::FESystem& system,
    const LevelSetMaintenanceRequest& request)
{
  return svmp::FE::level_set::generatedCutContextMarkerForMaintenance(
      system, request.configuration->volume_cut_request);
}

// SVMP_PDE_EXTENSION_FACTORIZATION selects the dry-region factorization of
// the PDE velocity extension: lu_colamd (default), the opt-in ldlt_amd, or
// the opt-in mumps (distributed over all ranks) or mumps_serial (sequential
// per component, deterministic; both FE_ENABLE_MUMPS builds).
application::core::PdeVelocityExtensionFactorization
pdeVelocityExtensionFactorizationSetting()
{
  static const auto setting = [] {
    const char* text = std::getenv("SVMP_PDE_EXTENSION_FACTORIZATION");
    if (text == nullptr || text[0] == '\0') {
      return application::core::PdeVelocityExtensionFactorization::LuColamd;
    }
    const auto parsed =
        application::core::pdeVelocityExtensionFactorizationFromToken(text);
    if (!parsed.has_value()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] SVMP_PDE_EXTENSION_FACTORIZATION='" +
          std::string(text) +
          "' is not one of lu_colamd, ldlt_amd, mumps, mumps_serial.");
    }
    if ((*parsed == application::core::PdeVelocityExtensionFactorization::Mumps ||
         *parsed ==
             application::core::PdeVelocityExtensionFactorization::MumpsSerial) &&
        !svmp::FE::backends::mumpsAvailable()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] SVMP_PDE_EXTENSION_FACTORIZATION=" +
          std::string(text) + " needs a build with FE_ENABLE_MUMPS=ON.");
    }
    return *parsed;
  }();
  return setting;
}

// The refresh of the extension is in
// FE/LevelSet/Extension/LevelSetVelocityExtensionRefresh.h.
using svmp::FE::level_set::PdeVelocityExtensionCaches;

// The dry-region factorization caches of the solves: factorizations are
// reused while their content key is unchanged (one cache per target field;
// see PdeVelocityExtensionCache).  The static-certificate probe keeps its own.
PdeVelocityExtensionCaches& solvePdeExtensionCaches()
{
  static PdeVelocityExtensionCaches caches;
  return caches;
}

bool levelSetAdvectionTraceEnabled()
{
  return parseBoolEnv("SVMP_TRACE_LEVEL_SET_ADVECTION", false) ||
         application::core::oopTraceEnabled();
}

// pde_extension_caches: the caches to use; nullptr uses the caches of the
// solves.
bool updateLevelSetAdvectionVelocitiesFromState(
    application::core::SimulationComponents& sim,
    const svmp::FE::systems::SystemStateView& state,
    const std::vector<LevelSetAdvectionVelocityRequest>& requests,
    bool refresh_frozen_algebraic_map = true,
    PdeVelocityExtensionCaches* pde_extension_caches = nullptr)
{
  if (!sim.fe_system || !sim.primary_mesh || requests.empty()) {
    return false;
  }
  return svmp::FE::level_set::refreshLevelSetVelocityExtensions(
      *sim.fe_system,
      *sim.primary_mesh,
      state,
      requests,
      svmp::FE::level_set::LevelSetVelocityExtensionRefreshOptions{
          .refresh_frozen_rows = refresh_frozen_algebraic_map,
          .factorization = pdeVelocityExtensionFactorizationSetting(),
          .trace = levelSetAdvectionTraceEnabled()},
      pde_extension_caches != nullptr ? *pde_extension_caches
                                      : solvePdeExtensionCaches());
}

// Registers the generated-state participant of the velocity extension
// (FE/LevelSet/Extension/LevelSetVelocityExtensionRefresh.h) when the run
// extends a level-set advection velocity.
void addLevelSetVelocityExtensionParticipant(
    svmp::FE::timestepping::GeneratedStateRegistry& generated_state,
    application::core::SimulationComponents& sim,
    const std::vector<LevelSetAdvectionVelocityRequest>& requests,
    svmp::FE::timestepping::GeneratedStateSolve solve,
    bool outer_fixed_point)
{
  if (requests.empty() || !sim.fe_system || !sim.primary_mesh) {
    return;
  }
  generated_state.add(
      std::make_shared<svmp::FE::level_set::LevelSetVelocityExtensionParticipant>(
          *sim.fe_system,
          *sim.primary_mesh,
          requests,
          svmp::FE::level_set::LevelSetVelocityExtensionParticipant::Options{
              .solve = solve,
              .outer_fixed_point = outer_fixed_point,
              .factorization = pdeVelocityExtensionFactorizationSetting(),
              .trace = levelSetAdvectionTraceEnabled()},
          solvePdeExtensionCaches()));
}

bool updateLevelSetAdvectionVelocities(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    const std::vector<LevelSetAdvectionVelocityRequest>& requests)
{
  const auto state = stateViewForHistory(history);
  return updateLevelSetAdvectionVelocitiesFromState(sim, state, requests, true);
}

// FNV-1a fingerprint of a name that enters the extension revision.
std::uint64_t velocityExtensionNameFingerprint(std::string_view key) noexcept
{
  std::uint64_t fingerprint = 1469598103934665603ull;
  for (const char character : key) {
    fingerprint ^=
        static_cast<std::uint64_t>(static_cast<unsigned char>(character));
    fingerprint *= 1099511628211ull;
  }
  return fingerprint == 0u ? 1u : fingerprint;
}

// The revision of the installed frozen extension rows of every monolithic
// request, or nullopt when no request has frozen rows (prescribed coupling).
std::optional<std::uint64_t> currentLevelSetVelocityExtensionRevision(
    const svmp::FE::systems::FESystem& system,
    const std::vector<LevelSetAdvectionVelocityRequest>& requests)
{
  std::uint64_t revision = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto mix = [&](std::uint64_t value) {
    for (std::size_t byte = 0u; byte < sizeof(value); ++byte) {
      revision ^= (value >> (byte * 8u)) & 0xffu;
      revision *= prime;
    }
  };
  bool found_current = false;
  for (const auto& request : requests) {
    const auto target =
        system.findFieldByName(request.target_velocity_field_name);
    if (target == svmp::FE::INVALID_FIELD_ID) {
      continue;
    }
    const auto kernel = svmp::FE::level_set::
        findLevelSetVelocityExtensionConstraintKernel(
            system, request.operator_tag, target);
    if (!kernel || !kernel->hasFrozenMap()) {
      continue;
    }
    found_current = true;
    mix(velocityExtensionNameFingerprint(request.level_set_field_name));
    mix(velocityExtensionNameFingerprint(request.target_velocity_field_name));
    mix(velocityExtensionNameFingerprint(request.operator_tag));
    mix(static_cast<std::uint64_t>(target));
    mix(kernel->frozenMapRevision());
  }
  std::optional<std::uint64_t> resolved;
  if (found_current) {
    resolved = revision == 0u ? std::uint64_t{1u} : revision;
  }
  const auto [minimum, maximum] = globalMinMaxUint64(
      resolved.value_or(0u), activeFESystemCommunicator(system));
  if (minimum != maximum) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance extension-map revision differs across the FE communicator.");
  }
  return resolved;
}

// ---- generated cut data on deep ghost cells --------------------------------
//
// Generated cut data is kept on the owned cells and on ghost cells at most
// <Generated_cut_ghost_layers> face layers away from them (default 2; -1
// keeps every stored cell); the mesh halo, the DOF layout and the numbering
// are unchanged.  On the production path no consumer reads cut data beyond
// the first ghost layer: assembly (ReverseScatter) assembles owned cells and
// the interior faces of owned cells, small-cut aggregation classifies owned
// cells and copies the canonical classes of ghosts from their owners, and
// the active-side vertex constraint routes its marks through cell owners.
// Runs keeping one or two ghost layers were bitwise identical to runs
// keeping all twelve (3D sphere and sessile drop on 8 and 16 ranks, 2D
// sessile drop, capillary wave and rise, the extended reference decks); the
// withheld layers only cost generation bookkeeping, snapshot records and
// their checks in every consumer.  The owned-rows assembly policy reads
// deeper (it assembles every stored cell) and keeps every layer.  The
// rank-local curvature recoveries that also did were retired with the
// level-set curvature projection (decisions D47 and D48).
[[nodiscard]] bool generatedCutNeedsEveryGhostLayer(
    const application::core::SimulationComponents& sim,
    const std::vector<ActiveCutVolumeRequest>& /*requests*/)
{
  return sim.fe_system->assemblyGhostPolicy() !=
         svmp::FE::assembly::GhostPolicy::ReverseScatter;
}

// The cut-context lifecycle (refresh, cache, signatures, resync, reinstall,
// restore retention and the geometry transaction) lives in
// FE/LevelSet/LevelSetGeometryService.h.  The driver keeps its old
// signatures as forwards: the host comes from SimulationComponents and the
// requests are parsed from the parameters, as before, at every call.
using svmp::FE::level_set::ActiveCutContextRefreshCache;
using svmp::FE::level_set::ActiveCutContextRefreshSignature;
using svmp::FE::level_set::CurrentCandidateProducerLinkageResult;
using svmp::FE::level_set::GhostCutWithholding;
using svmp::FE::level_set::armActiveCutRestoreRetention;
using svmp::FE::level_set::ghostCellLayers;
using svmp::FE::level_set::observeActiveCutContextRefresh;
using svmp::FE::level_set::releaseActiveCutRestoreRetention;
using svmp::FE::level_set::withholdGhostCutLayersBeyond;

svmp::FE::level_set::LevelSetGeometryHost levelSetGeometryHost(
    application::core::SimulationComponents& sim)
{
  return svmp::FE::level_set::LevelSetGeometryHost{
      .fe_system = sim.fe_system.get(),
      .primary_mesh = sim.primary_mesh.get(),
      .free_surface_geometry_snapshot_cache =
          sim.free_surface_geometry_snapshot_cache,
      .free_surface_geometry_snapshot_reuse_caches =
          sim.free_surface_geometry_snapshot_reuse_caches,
      .validate_cut_volume_consumer =
          [](const svmp::FE::systems::FESystem& system,
             const ActiveCutVolumeRequest& request,
             int resolved_marker) {
            application::core::validateEquationLevelCutVolumeConsumer(
                system, request, resolved_marker);
          },
      .generated_cut_needs_every_ghost_layer =
          [&sim](const std::vector<ActiveCutVolumeRequest>& requests) {
            return generatedCutNeedsEveryGhostLayer(sim, requests);
          },
      .two_fluid_accepted_stage_diagnostics_declared =
          sim.fe_system &&
          !svmp::FE::systems::twoFluidStageRecords(std::as_const(*sim.fe_system))
               .twoFluidAcceptedStageDiagnosticDeclarations()
               .empty(),
  };
}

// The geometry reads a const SimulationComponents only through these
// functions, which do not modify it.
svmp::FE::level_set::LevelSetGeometryHost levelSetGeometryHost(
    const application::core::SimulationComponents& sim)
{
  return levelSetGeometryHost(
      const_cast<application::core::SimulationComponents&>(sim));
}

std::size_t syncActiveLevelSetVertexFieldsFromSolution(
    application::core::SimulationComponents& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::span<const svmp::FE::Real> fe_solution)
{
  return svmp::FE::level_set::syncActiveLevelSetVertexFieldsFromSolution(
      levelSetGeometryHost(sim), requests, fe_solution);
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextFromSolution(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    std::span<const svmp::FE::Real> fe_solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    const char* provenance,
    const char* solution_source = nullptr,
    bool request_consensus_already_checked = false)
{
  const auto requests = activeCutVolumeRequests(params);
  return svmp::FE::level_set::refreshActiveCutIntegrationContextFromSolution(
      levelSetGeometryHost(sim),
      requests,
      fe_solution,
      lifecycle,
      provenance,
      solution_source,
      request_consensus_already_checked);
}

std::optional<ActiveCutContextRefreshSignature> activeCutContextRefreshSignature(
    const application::core::SimulationComponents& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::span<const svmp::FE::Real> fe_solution)
{
  return svmp::FE::level_set::activeCutContextRefreshSignature(
      levelSetGeometryHost(sim), requests, fe_solution);
}

std::optional<ActiveCutContextRefreshSignature> activeCutContextRefreshSignature(
    const application::core::SimulationComponents& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::backends::GenericVector& fe_solution)
{
  return svmp::FE::level_set::activeCutContextRefreshSignature(
      levelSetGeometryHost(sim), requests, fe_solution);
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextFromSolutionCached(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    std::span<const svmp::FE::Real> fe_solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance,
    const char* solution_source = nullptr)
{
  const auto requests = activeCutVolumeRequests(params);
  return svmp::FE::level_set::refreshActiveCutIntegrationContextFromSolutionCached(
      levelSetGeometryHost(sim),
      requests,
      fe_solution,
      lifecycle,
      cache,
      provenance,
      solution_source);
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCached(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    svmp::FE::backends::GenericVector& solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance)
{
  const auto requests = activeCutVolumeRequests(params);
  return svmp::FE::level_set::refreshActiveCutIntegrationContextCached(
      levelSetGeometryHost(sim),
      requests,
      solution,
      lifecycle,
      cache,
      provenance);
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCached(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance)
{
  const auto requests = activeCutVolumeRequests(params);
  return svmp::FE::level_set::refreshActiveCutIntegrationContextCached(
      levelSetGeometryHost(sim),
      requests,
      state,
      lifecycle,
      cache,
      provenance);
}

void retainActiveCutContextAtAttemptEntry(
    const application::core::SimulationComponents& sim,
    ActiveCutContextRefreshCache& cache,
    bool outer_fixed_point_state,
    bool projected_outer_fixed_point_state)
{
  svmp::FE::level_set::retainActiveCutContextAtAttemptEntry(
      levelSetGeometryHost(sim),
      cache,
      outer_fixed_point_state,
      projected_outer_fixed_point_state);
}

std::optional<ActiveCutContextRefreshReport>
reinstallRetainedActiveCutContextForRestoredState(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    const svmp::FE::systems::SystemStateView& state,
    ActiveCutContextRefreshCache& cache,
    const char* provenance)
{
  const auto requests = activeCutVolumeRequests(params);
  return svmp::FE::level_set::reinstallRetainedActiveCutContextForRestoredState(
      levelSetGeometryHost(sim), requests, state, cache, provenance);
}

// The geometry transaction of the driver: the FE transaction with the
// driver's constructor and its refresh and producer-linkage spellings, which
// parse the cut-volume requests from the parameters at every call as before.
// It is also the geometry transaction of the conservative-phase workflow.
class LevelSetMaintenanceGeometryTransaction final
    : public svmp::FE::level_set::LevelSetGeometryTransaction {
public:
  LevelSetMaintenanceGeometryTransaction(
      application::core::SimulationComponents& sim,
      svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
      ActiveCutContextRefreshCache& refresh_cache,
      const std::vector<ActiveCutVolumeRequest>& requests,
      std::span<const svmp::FE::FieldId> prescribed_fields = {})
      : LevelSetGeometryTransaction(
            levelSetGeometryHost(sim),
            lifecycle,
            refresh_cache,
            requests,
            prescribed_fields)
  {
  }

  using LevelSetGeometryTransaction::currentCandidateProducerLinkage;
  using LevelSetGeometryTransaction::refresh;

  [[nodiscard]] ActiveCutContextRefreshReport refresh(
      const Parameters& params,
      std::span<const svmp::FE::Real> candidate,
      bool force_rebuild = false)
  {
    return LevelSetGeometryTransaction::refresh(
        activeCutVolumeRequests(params), candidate, force_rebuild);
  }

  [[nodiscard]] CurrentCandidateProducerLinkageResult
  currentCandidateProducerLinkage(
      const Parameters& params,
      std::span<const svmp::FE::Real> candidate,
      const ActiveCutContextRefreshReport& refresh_report) const
  {
    return LevelSetGeometryTransaction::currentCandidateProducerLinkage(
        activeCutVolumeRequests(params), candidate, refresh_report);
  }
};

enum class PostacceptMaintenanceTopologyDecision : std::uint8_t {
  Commit,
  RejectMaintenance,
  InvariantFailure,
};

struct PostacceptMaintenanceTopologyEvidence {
  bool baseline_present{false};
  std::uint64_t baseline_key{0u};
  bool final_present{false};
  std::uint64_t final_key{0u};
  bool keys_equal{false};
};

PostacceptMaintenanceTopologyEvidence
postacceptMaintenanceTopologyEvidence(
    const std::optional<std::uint64_t>& baseline,
    const std::optional<std::uint64_t>& final) noexcept
{
  const bool baseline_present = baseline.has_value();
  const bool final_present = final.has_value();
  const auto baseline_key = baseline.value_or(0u);
  const auto final_key = final.value_or(0u);
  return PostacceptMaintenanceTopologyEvidence{
      .baseline_present = baseline_present,
      .baseline_key = baseline_key,
      .final_present = final_present,
      .final_key = final_key,
      .keys_equal = baseline_present && final_present &&
                    baseline_key == final_key,
  };
}

PostacceptMaintenanceTopologyDecision
classifyPostacceptMaintenanceTopologyEvidence(
    const PostacceptMaintenanceTopologyEvidence& evidence) noexcept
{
  const bool expected_equality =
      evidence.baseline_present && evidence.final_present &&
      evidence.baseline_key == evidence.final_key;
  if (!evidence.baseline_present || evidence.baseline_key == 0u ||
      !evidence.final_present || evidence.final_key == 0u ||
      evidence.keys_equal != expected_equality) {
    return PostacceptMaintenanceTopologyDecision::InvariantFailure;
  }
  return evidence.keys_equal
      ? PostacceptMaintenanceTopologyDecision::Commit
      : PostacceptMaintenanceTopologyDecision::RejectMaintenance;
}

bool collectivePostacceptMaintenanceRecoveryPhaseReady(
    bool local_ready,
    const svmp::MeshComm& comm)
{
  return !globalAnyBool(!local_ready, comm);
}

bool postacceptMaintenanceRejectionEvidenceMayPublish(
    bool geometry_restored,
    bool checkpoint_restored,
    bool restored_topology_verified,
    bool ledger_rejection_preflight_ready) noexcept
{
  return geometry_restored && checkpoint_restored &&
         restored_topology_verified &&
         ledger_rejection_preflight_ready;
}

void requirePostacceptMaintenanceRecoveryPhase(
    const std::exception_ptr& local_failure,
    const svmp::MeshComm& comm,
    std::string_view phase)
{
  if (collectivePostacceptMaintenanceRecoveryPhaseReady(
          local_failure == nullptr, comm)) {
    return;
  }
  if (local_failure != nullptr) {
    std::rethrow_exception(local_failure);
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Another active FE communicator rank "
      "failed postaccept maintenance recovery phase '" +
      std::string(phase) + "'.");
}

std::vector<svmp::FE::Real>
captureFeOrderedVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm)
{
  return svmp::FE::systems::captureFeOrderedVectorCollectively(
      solution, feCollectiveCommunicator(comm));
}

std::vector<svmp::FE::Real>
capturePostacceptMaintenanceVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm)
{
  return captureFeOrderedVectorCollectively(solution, comm);
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm)
{
  return captureFeOrderedVectorCollectively(solution, comm);
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    const svmp::FE::systems::SystemStateView& state,
    const svmp::MeshComm& comm)
{
  if (state.u_vector != nullptr) {
    auto* solution =
        const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
    return gatherFeOrderedSolution(*solution, comm);
  }
  if (!state.u.empty()) {
    return std::vector<svmp::FE::Real>(state.u.begin(), state.u.end());
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Could not collectively gather FE-ordered state values.");
}

void assignPostacceptMaintenanceVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    std::span<const svmp::FE::Real> values,
    const svmp::MeshComm& comm)
{
  svmp::FE::systems::assignFeOrderedVectorCollectively(
      solution, values, feCollectiveCommunicator(comm));
}

enum class PostacceptMaintenanceRollbackReason : std::uint8_t {
  MaintenanceCutTopologyChanged,
  CollectiveConsensusRejection,
  CandidateException,
};

const char* postacceptMaintenanceRollbackReasonName(
    PostacceptMaintenanceRollbackReason reason) noexcept
{
  switch (reason) {
    case PostacceptMaintenanceRollbackReason::
        MaintenanceCutTopologyChanged:
      return "maintenance_cut_topology_changed";
    case PostacceptMaintenanceRollbackReason::
        CollectiveConsensusRejection:
      return "collective_consensus_rejection";
    case PostacceptMaintenanceRollbackReason::CandidateException:
      return "candidate_exception";
  }
  return "unknown";
}

enum class PostacceptMaintenanceRollbackScope : std::uint8_t {
  AcceptedEndpointMaintenanceOnly,
};

const char* postacceptMaintenanceRollbackScopeName(
    PostacceptMaintenanceRollbackScope scope) noexcept
{
  switch (scope) {
    case PostacceptMaintenanceRollbackScope::
        AcceptedEndpointMaintenanceOnly:
      return "accepted_endpoint_maintenance_only";
  }
  return "unknown";
}

enum class LevelSetMaintenancePublicationState : std::uint8_t {
  Staged,
  Publishing,
  Published,
  PublicationFailed,
};

const char* levelSetMaintenancePublicationStateName(
    LevelSetMaintenancePublicationState state) noexcept
{
  switch (state) {
    case LevelSetMaintenancePublicationState::Staged:
      return "staged";
    case LevelSetMaintenancePublicationState::Publishing:
      return "publishing";
    case LevelSetMaintenancePublicationState::Published:
      return "published";
    case LevelSetMaintenancePublicationState::PublicationFailed:
      return "publication_failed";
  }
  return "unknown";
}

// The conservative-phase workflow is in
// FE/LevelSet/LevelSetConservativePhaseWorkflow and
// FE/LevelSet/LevelSetPhaseGeometryReconciliation; its host adapter follows
// the contact-stage builder below.
using svmp::FE::level_set::ConservativePhaseCandidateResult;
using svmp::FE::level_set::ConservativePhaseCandidateStageSnapshot;
using svmp::FE::level_set::ConservativePhaseContactStageCandidate;
using svmp::FE::level_set::ConservativePhaseFixedGraphBinding;
using svmp::FE::level_set::ConservativePhaseGeometryReconciliationResult;
using svmp::FE::level_set::appendConservativePhaseSplitStageProvenanceWords;
using svmp::FE::level_set::mixConservativePhaseArtifactFingerprint;
using svmp::FE::level_set::rollbackConservativePhaseCandidate;
using svmp::FE::level_set::sameConservativePhaseGraphIdentity;
using svmp::FE::level_set::sameConservativePhaseStageRealBits;

using ConservativePhaseContactStageBuilder = std::function<
    ConservativePhaseContactStageCandidate(
        std::span<const svmp::FE::Real>,
        LevelSetMaintenanceGeometryTransaction*)>;

template <typename RestoreEndpoint>
void restoreAcceptedContactStageEndpointAndRequireCollectiveSuccess(
    const std::exception_ptr& local_stage_failure,
    RestoreEndpoint&& restore_endpoint,
    const svmp::MeshComm& comm)
{
  // Every rank must restore the endpoint before any rank reports a local
  // stage failure.  Otherwise successful ranks can advance to commit
  // consensus while a failed rank has already unwound the callback.
  auto local_failure = local_stage_failure;
  try {
    std::forward<RestoreEndpoint>(restore_endpoint)();
  } catch (...) {
    // Keep the original stage diagnostic when both stage evaluation and
    // endpoint restoration fail locally.
    if (local_failure == nullptr) {
      local_failure = std::current_exception();
    }
  }
  requireCollectivePhasePreparation(
      local_failure,
      comm,
      "accepted_contact_stage_evaluation_and_endpoint_restore");
}

ConservativePhaseContactStageCandidate
buildAcceptedFreeSurfaceContactStageCandidate(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& refresh_cache,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::span<const svmp::FE::Real> previous_solution,
    std::span<const svmp::FE::Real> endpoint_solution,
    LevelSetMaintenanceGeometryTransaction* active_transaction,
    const std::optional<svmp::FE::systems::
                            FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha = std::nullopt,
    const DynamicContactFirstOrderGeneralizedAlphaObservation*
        exact_operator_stage_observation = nullptr)
{
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  ConservativePhaseContactStageCandidate contact_stage;
  std::optional<LevelSetMaintenanceGeometryTransaction>
      owned_transaction;
  const bool locally_owns_transaction = active_transaction == nullptr;
  const bool any_owned_transaction =
      globalAnyBool(locally_owns_transaction, comm);
  const bool any_reused_transaction =
      globalAnyBool(!locally_owns_transaction, comm);
  if (any_owned_transaction && any_reused_transaction) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Contact-stage reconstruction transaction ownership differs across the FE communicator.");
  }

  std::exception_ptr local_preparation_failure;
  try {
    if (!locally_owns_transaction && !active_transaction->active()) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Contact-stage reconstruction received an inactive geometry transaction.");
    }
    if (endpoint_solution.size() != previous_solution.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Contact-stage reconstruction requires equal endpoint and previous solution layouts.");
    }
    contact_stage.stage_solution.assign(
        endpoint_solution.size(), svmp::FE::Real{0.0});
    for (std::size_t i = 0; i < contact_stage.stage_solution.size(); ++i) {
      contact_stage.stage_solution[i] =
          (svmp::FE::Real{1.0} - stage_alpha_f) * previous_solution[i] +
          stage_alpha_f * endpoint_solution[i];
    }
    if (first_order_generalized_alpha.has_value() !=
        (exact_operator_stage_observation != nullptr)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] First-order generalized-alpha "
          "contact-stage construction requires its exact operator-stage "
          "field snapshot.");
    }
    if (exact_operator_stage_observation != nullptr) {
      const auto& observation = *exact_operator_stage_observation;
      if (observation.provenance != *first_order_generalized_alpha ||
          !contactStageRealValuesMatch(
              observation.state_time, stage_time) ||
          !contactStageRealValuesMatch(
              observation.provenance.alpha_f, stage_alpha_f) ||
          !dynamicContactStageProvenanceMatches(
              observation,
              sim.fe_system->meshAccess(),
              svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations())) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Exact generalized-alpha "
            "operator-stage base-mesh or contact-wall snapshot is stale "
            "for contact-stage "
            "construction.");
      }
      overlayDynamicContactOperatorStageState(
          *sim.fe_system,
          svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations(),
          observation.operator_stage_state,
          contact_stage.stage_solution);
    }
  } catch (...) {
    local_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_preparation_failure,
      comm,
      "accepted_contact_stage_geometry_refresh");
  const auto [minimum_exact_stage_present,
              maximum_exact_stage_present] =
      globalMinMaxUint64(
          exact_operator_stage_observation != nullptr ? 1u : 0u, comm);
  if (minimum_exact_stage_present != maximum_exact_stage_present) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Exact generalized-alpha "
        "operator-stage snapshot presence differs across the FE "
        "communicator.");
  }
  if (exact_operator_stage_observation != nullptr) {
    requireDynamicContactOperatorStageObservationConsensus(
        *exact_operator_stage_observation, comm);
  }

  std::exception_ptr local_transaction_begin_failure;
  try {
    if (locally_owns_transaction) {
      // In-place storage avoids an allocation before the transaction
      // constructor's communicator-coordinated preparation phases.
      owned_transaction.emplace(
          sim, lifecycle, refresh_cache, active_cut_requests);
      active_transaction = &*owned_transaction;
    }
  } catch (...) {
    local_transaction_begin_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_transaction_begin_failure,
      comm,
      "accepted_contact_stage_transaction_begin");

  std::exception_ptr local_stage_failure;
  try {
    (void)active_transaction->refresh(
        params,
        std::span<const svmp::FE::Real>(
            contact_stage.stage_solution.data(),
            contact_stage.stage_solution.size()));
    contact_stage.stages = evaluateAcceptedFreeSurfaceContactStages(
        *sim.fe_system,
        svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations(),
        stage_time,
        stage_alpha_f,
        previous_state_revision,
        endpoint_state_revision,
        contact_stage.stage_solution,
        first_order_generalized_alpha);
    contact_stage.constraints =
        captureAcceptedContactStageWallConstraints(
            *sim.fe_system,
            svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations(),
            contact_stage.stages);
  } catch (...) {
    local_stage_failure = std::current_exception();
  }

  restoreAcceptedContactStageEndpointAndRequireCollectiveSuccess(
      local_stage_failure,
      [&] {
        (void)active_transaction->refresh(params, endpoint_solution);
      },
      comm);
  std::exception_ptr local_commit_failure;
  try {
    if (owned_transaction.has_value()) {
      owned_transaction->commit();
    }
  } catch (...) {
    local_commit_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_commit_failure,
      comm,
      "accepted_contact_stage_transaction_commit");
  return contact_stage;
}

// ---------------------------------------------------------------------------
// Host adapter of the FE conservative-phase workflow.  The request views, the
// geometry transaction and the host callbacks bind the workflow to this
// driver's maintenance requests, cut-context transaction and maintenance
// stages; each callback calls the driver function the workflow called before
// it moved to FE, with the same arguments.  The functions after them keep the
// driver signatures that runSteady, runTransient and the white-box tests use.
// ---------------------------------------------------------------------------

/// Views of maintenance requests for the FE workflow.  They refer to each
/// request's configuration and conservative-phase state and are valid while
/// the requests are.
class ConservativePhaseRequestViews {
public:
  // The workflow changes the referenced state only in functions that take
  // mutable requests; views of const requests go only to functions that
  // take const requests.
  explicit ConservativePhaseRequestViews(
      std::span<const LevelSetMaintenanceRequest> requests)
  {
    configurations_.reserve(requests.size());
    views_.reserve(requests.size());
    for (const auto& request : requests) {
      configurations_.push_back(
          svmp::FE::level_set::ConservativePhaseRequest::Configuration{
              request.configuration->transport,
              request.configuration->volume_cut_request,
              request.configuration->isovalue});
    }
    for (std::size_t index = 0u; index < requests.size(); ++index) {
      auto& runtime = const_cast<LevelSetMaintenanceRuntimeState&>(
          requests[index].runtime);
      views_.push_back(svmp::FE::level_set::ConservativePhaseRequest{
          &configurations_[index],
          {runtime.conservative_phase_initialized,
           runtime.conservative_phase_graph}});
    }
  }

  ConservativePhaseRequestViews(const ConservativePhaseRequestViews&) = delete;
  ConservativePhaseRequestViews& operator=(
      const ConservativePhaseRequestViews&) = delete;

  [[nodiscard]] std::span<svmp::FE::level_set::ConservativePhaseRequest>
  all() noexcept
  {
    return views_;
  }

  [[nodiscard]] svmp::FE::level_set::ConservativePhaseRequest& front() noexcept
  {
    return views_.front();
  }

private:
  std::vector<svmp::FE::level_set::ConservativePhaseRequest::Configuration>
      configurations_{};
  std::vector<svmp::FE::level_set::ConservativePhaseRequest> views_{};
};

/// The two-fluid momentum reconciliation of the driver's maintenance
/// requests, which the Physics function reads through conservative-phase
/// request views.
TwoFluidMomentumReconciliationBatch
buildCurrentTwoFluidMomentumReconciliations(
    application::core::SimulationComponents& sim,
    std::span<const LevelSetMaintenanceRequest> maintenance_requests,
    std::span<const svmp::FE::Real> corrected_solution,
    std::uint64_t corrected_algebraic_revision,
    svmp::FE::Real effective_dt,
    const svmp::MeshComm& comm)
{
  ConservativePhaseRequestViews views(maintenance_requests);
  return svmp::FE::level_set::buildCurrentTwoFluidMomentumReconciliations(
      sim,
      views.all(),
      corrected_solution,
      corrected_algebraic_revision,
      effective_dt,
      comm);
}

/// The driver transaction behind a workflow transaction (null for null).
LevelSetMaintenanceGeometryTransaction* driverGeometryTransaction(
    svmp::FE::level_set::ConservativePhaseGeometryTransaction* transaction)
{
  if (transaction == nullptr) {
    return nullptr;
  }
  auto* const driver_transaction =
      dynamic_cast<LevelSetMaintenanceGeometryTransaction*>(transaction);
  if (driver_transaction == nullptr) {
    throw std::logic_error(
        "[svMultiPhysics::Application] Conservative phase geometry transaction was not created by the Application driver.");
  }
  return driver_transaction;
}

/// The workflow host of @p sim: system, history, vertex-field evaluation on
/// the primary mesh (evaluateVertexField) and the wall-aware maintenance
/// context (resolveLevelSetWallAwareMaintenanceContext).
svmp::FE::level_set::ConservativePhaseWorkflowHost
conservativePhaseWorkflowHost(application::core::SimulationComponents& sim)
{
  svmp::FE::level_set::ConservativePhaseWorkflowHost host;
  host.fe_system = sim.fe_system.get();
  host.time_history = sim.time_history.get();
  if (sim.fe_system && sim.primary_mesh) {
    host.evaluate_vertex_field =
        [&sim](svmp::FE::FieldId field,
               const svmp::FE::systems::SystemStateView& state,
               std::size_t components,
               std::string_view action) {
          return evaluateVertexField(
              *sim.fe_system,
              *sim.primary_mesh,
              field,
              state,
              components,
              action);
        };
    host.primary_mesh_vertex_count = [&sim] {
      return sim.primary_mesh->n_vertices();
    };
  }
  host.wall_contact_constraints =
      [&sim](svmp::FE::timestepping::TimeHistory& history,
             svmp::FE::FieldId level_set_field,
             svmp::FE::Real endpoint_time,
             std::span<const svmp::FE::systems::
                           FreeSurfaceAcceptedContactStageState>
                 accepted_contact_stages,
             std::span<const svmp::FE::level_set::LevelSetWallContactConstraint>
                 accepted_contact_stage_constraints) {
        return resolveLevelSetWallAwareMaintenanceContext(
                   sim,
                   history,
                   level_set_field,
                   endpoint_time,
                   accepted_contact_stages,
                   accepted_contact_stage_constraints)
            .local_constraints;
      };
  return host;
}

/// Adds to @p host the request schedule
/// (canonicalLevelSetMaintenanceRequestSchedule) of the driver's
/// @p requests.
void bindConservativePhaseMaintenance(
    svmp::FE::level_set::ConservativePhaseWorkflowHost& host,
    std::vector<LevelSetMaintenanceRequest>& requests)
{
  host.prospective_request_schedule = [&requests](int completed_step) {
    return canonicalLevelSetMaintenanceRequestSchedule(
        requests,
        LevelSetMaintenanceScheduleStage::ProspectiveAcceptedEndpoint,
        completed_step);
  };
}

svmp::FE::level_set::LevelSetP1PhaseTransportGraph&
requireCurrentConservativePhaseGraph(
    svmp::FE::systems::FESystem& system,
    LevelSetMaintenanceRequest& request,
    bool allow_graph_rebuild = true)
{
  ConservativePhaseRequestViews views(
      std::span<const LevelSetMaintenanceRequest>(&request, 1u));
  return svmp::FE::level_set::requireCurrentConservativePhaseGraph(
      system, views.front(), allow_graph_rebuild);
}

svmp::FE::level_set::LevelSetP1PhaseProjectionResult
projectCurrentConservativePhaseGeometry(
    svmp::FE::systems::FESystem& system,
    LevelSetMaintenanceRequest& request,
    bool allow_graph_rebuild = true)
{
  ConservativePhaseRequestViews views(
      std::span<const LevelSetMaintenanceRequest>(&request, 1u));
  return svmp::FE::level_set::projectCurrentConservativePhaseGeometry(
      system, views.front(), allow_graph_rebuild);
}

void initializeConservativePhaseStates(
    application::core::SimulationComponents& sim,
    std::vector<LevelSetMaintenanceRequest>& requests)
{
  ConservativePhaseRequestViews views(requests);
  svmp::FE::level_set::initializeConservativePhaseStates(
      conservativePhaseWorkflowHost(sim), views.all());
}

std::vector<std::array<svmp::FE::Real, 3>>
sampleConservativePhaseVelocity(
    application::core::SimulationComponents& sim,
    const LevelSetMaintenanceRequest& request,
    const svmp::FE::level_set::LevelSetP1PhaseTransportGraph& graph,
    const svmp::FE::systems::SystemStateView& state)
{
  ConservativePhaseRequestViews views(
      std::span<const LevelSetMaintenanceRequest>(&request, 1u));
  return svmp::FE::level_set::sampleConservativePhaseVelocity(
      conservativePhaseWorkflowHost(sim), views.front(), graph, state);
}

std::vector<std::uint8_t> conservativePhaseContactProtectedNodes(
    const svmp::FE::systems::FESystem& system,
    const LevelSetMaintenanceRequest& request,
    const svmp::FE::level_set::LevelSetP1PhaseTransportGraph& graph,
    std::span<const svmp::FE::level_set::LevelSetWallContactConstraint>
        constraints)
{
  ConservativePhaseRequestViews views(
      std::span<const LevelSetMaintenanceRequest>(&request, 1u));
  return svmp::FE::level_set::conservativePhaseContactProtectedNodes(
      system, views.front(), graph, constraints);
}

bool collectiveConservativePhaseVelocityBitsAgree(
    std::span<const std::array<svmp::FE::Real, 3>> velocity,
    const svmp::MeshComm& comm)
{
  return svmp::FE::level_set::collectiveConservativePhaseVelocityBitsAgree(
      velocity, feCollectiveCommunicator(comm));
}

bool collectiveConservativePhaseCandidateStageSnapshotAgrees(
    const ConservativePhaseCandidateStageSnapshot& stage,
    const svmp::MeshComm& comm)
{
  return svmp::FE::level_set::
      collectiveConservativePhaseCandidateStageSnapshotAgrees(
          stage, feCollectiveCommunicator(comm));
}

std::vector<ConservativePhaseFixedGraphBinding>
captureConservativePhaseFixedGraphBindings(
    const svmp::FE::systems::FESystem& system,
    const std::vector<LevelSetMaintenanceRequest>& requests,
    const svmp::MeshComm& comm)
{
  ConservativePhaseRequestViews views(requests);
  return svmp::FE::level_set::captureConservativePhaseFixedGraphBindings(
      system, views.all(), feCollectiveCommunicator(comm));
}

void requireConservativePhaseFixedGraphBindingsCurrent(
    const svmp::FE::systems::FESystem& system,
    const std::vector<LevelSetMaintenanceRequest>& requests,
    std::span<const ConservativePhaseFixedGraphBinding> bindings,
    const svmp::MeshComm& comm,
    std::string_view phase)
{
  ConservativePhaseRequestViews views(requests);
  svmp::FE::level_set::requireConservativePhaseFixedGraphBindingsCurrent(
      system, views.all(), bindings, feCollectiveCommunicator(comm), phase);
}

ConservativePhaseCandidateStageSnapshot
buildConservativePhaseCandidateStageSnapshot(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    std::vector<LevelSetMaintenanceRequest>& requests,
    std::span<const ConservativePhaseFixedGraphBinding> fixed_graph_bindings,
    const svmp::FE::timestepping::CandidateStageObservation& observation,
    const svmp::MeshComm& comm)
{
  auto host = conservativePhaseWorkflowHost(sim);
  bindConservativePhaseMaintenance(host, requests);
  host.operator_state_fingerprint =
      [&comm](const svmp::FE::backends::GenericVector& state) {
        return collectiveMeshBoundaryStateFingerprint(state, comm);
      };
  ConservativePhaseRequestViews views(requests);
  return svmp::FE::level_set::buildConservativePhaseCandidateStageSnapshot(
      host,
      history,
      views.all(),
      fixed_graph_bindings,
      observation,
      feCollectiveCommunicator(comm));
}

void writeAcceptedConservativePhaseArtifacts(
    const Parameters& params,
    const std::vector<LevelSetMaintenanceRequest>& requests,
    const ConservativePhaseCandidateResult& candidate,
    std::uint64_t accepted_step,
    svmp::FE::Real accepted_time,
    svmp::FE::Real time_step,
    std::uint64_t state_revision,
    const svmp::MeshComm& comm)
{
  const auto& save_results_in_folder =
      params.general_simulation_parameters.save_results_in_folder;
  const std::string results_folder = save_results_in_folder.defined()
      ? save_results_in_folder.value()
      : std::string{};
  ConservativePhaseRequestViews views(requests);
  svmp::FE::level_set::writeAcceptedConservativePhaseArtifacts(
      results_folder,
      views.all(),
      candidate,
      accepted_step,
      accepted_time,
      time_step,
      state_revision,
      feCollectiveCommunicator(comm));
}

ConservativePhaseGeometryReconciliationResult
reconcileConservativePhaseGeometry(
    application::core::SimulationComponents& sim,
    LevelSetMaintenanceRequest& request,
    const Parameters& params,
    std::span<const svmp::FE::Real> target_liquid_phase_mass,
    std::vector<svmp::FE::Real>& candidate,
    LevelSetMaintenanceGeometryTransaction& transaction,
    std::span<const std::uint8_t> contact_protected_nodes = {})
{
  ConservativePhaseRequestViews views(
      std::span<const LevelSetMaintenanceRequest>(&request, 1u));
  // The transaction refreshes with the cut-volume requests of its begin,
  // which every caller parses from these parameters.
  (void)params;
  return svmp::FE::level_set::reconcileConservativePhaseGeometry(
      conservativePhaseWorkflowHost(sim),
      views.front(),
      target_liquid_phase_mass,
      candidate,
      transaction,
      contact_protected_nodes);
}

ConservativePhaseCandidateResult applyConservativePhaseCandidates(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    std::vector<LevelSetMaintenanceRequest>& requests,
    const Parameters& params,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& refresh_cache,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    ConservativePhaseCandidateStageSnapshot& temporal_stage,
    std::span<const ConservativePhaseFixedGraphBinding>
        fixed_graph_bindings,
    std::uint64_t expected_attempt,
    const ConservativePhaseContactStageBuilder& contact_stage_builder = {},
    const LevelSetMaintenanceStageObserver& observe_stage = {})
{
  auto host = conservativePhaseWorkflowHost(sim);
  bindConservativePhaseMaintenance(host, requests);
  host.begin_geometry_transaction =
      [&]() -> std::unique_ptr<
                svmp::FE::level_set::ConservativePhaseGeometryTransaction> {
    return std::make_unique<LevelSetMaintenanceGeometryTransaction>(
        sim, lifecycle, refresh_cache, active_cut_requests);
  };
  svmp::FE::level_set::ConservativePhaseContactStageCallback
      contact_stage_callback;
  if (contact_stage_builder) {
    contact_stage_callback =
        [&contact_stage_builder](
            std::span<const svmp::FE::Real> endpoint_solution,
            svmp::FE::level_set::ConservativePhaseGeometryTransaction*
                transaction) {
          return contact_stage_builder(
              endpoint_solution, driverGeometryTransaction(transaction));
        };
  }
  ConservativePhaseRequestViews views(requests);
  return svmp::FE::level_set::applyConservativePhaseCandidates(
      host,
      history,
      views.all(),
      temporal_stage,
      fixed_graph_bindings,
      expected_attempt,
      contact_stage_callback,
      observe_stage);
}

using svmp::FE::systems::ZeroTimeDerivativeIntegrator;

std::uint64_t collectiveConstraintSemanticsKey(
    const svmp::FE::systems::FESystem& system,
    const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveConstraintSemanticsKey(
      system, feCollectiveCommunicator(comm));
}

struct StaticCapillaryPressureCertificate {
  svmp::FE::timestepping::NewtonReport report{};
  std::vector<svmp::FE::Real> certified_solution{};
};

// The probe is one Newton step on a copy of the candidate.  A monolithic
// (algebraic) advection velocity extension assembles its frozen map, which
// the production solves refresh before a nonlinear solve and keep frozen
// within it.  The probe therefore synchronizes its generated state as those
// solves do, with the Extension participant of the algebraic extensions of
// extension_requests only: at the probe's first accepted nonlinear state
// (the candidate) it refreshes their frozen maps, with PDE extension caches
// of its own, and keeps them frozen for the rest of the probe.  The kernels'
// maps are restored exactly afterwards.  Without an algebraic extension the
// probe has no state synchronization.
StaticCapillaryPressureCertificate
evaluateStaticCapillaryPressureCertificate(
    application::core::SimulationComponents& sim,
    std::span<const svmp::FE::Real> candidate,
    const application::core::StaticCapillaryEquilibriumOptions&
        certificate_options,
    bool initialize_compatible_pressure,
    const std::vector<LevelSetAdvectionVelocityRequest>& extension_requests = {})
{
  if (!sim.fe_system || !sim.backend || !sim.linear_solver ||
      !sim.time_history) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary pressure certificate requires an FE system, backend, linear solver, and time history.");
  }
  if (candidate.empty() ||
      candidate.size() >
          static_cast<std::size_t>(
              std::numeric_limits<svmp::FE::GlobalIndex>::max())) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary pressure certificate received an invalid candidate layout.");
  }

  auto& system = *sim.fe_system;
  const auto comm = activeFESystemCommunicator(system);
  const auto auxiliary_state = system.checkpointAuxiliaryState();
  const auto bordered_state = system.borderedCoupling();
  struct FrozenExtensionMap {
    std::shared_ptr<svmp::FE::level_set::LevelSetVelocityExtensionConstraintKernel>
        kernel{};
    svmp::FE::level_set::LevelSetVelocityExtensionConstraintKernel::
        FrozenMapState state{};
  };
  std::vector<LevelSetAdvectionVelocityRequest> algebraic_extension_requests;
  std::vector<FrozenExtensionMap> frozen_extension_maps;
  for (const auto& request : extension_requests) {
    const auto target =
        system.findFieldByName(request.target_velocity_field_name);
    if (target == svmp::FE::INVALID_FIELD_ID ||
        system.fieldRecord(target).source_kind !=
            svmp::FE::systems::FieldSourceKind::Unknown) {
      continue;
    }
    auto kernel = svmp::FE::level_set::
        findLevelSetVelocityExtensionConstraintKernel(
            system, request.operator_tag, target);
    if (!kernel) {
      continue;
    }
    algebraic_extension_requests.push_back(request);
    auto state = kernel->frozenMapState();
    frozen_extension_maps.push_back(FrozenExtensionMap{
        .kernel = std::move(kernel),
        .state = std::move(state),
    });
  }
  const auto restore_system_state = [&] {
    system.restoreAuxiliaryState(auxiliary_state);
    system.borderedCoupling() = bordered_state;
    for (const auto& frozen : frozen_extension_maps) {
      frozen.kernel->restoreFrozenMapState(frozen.state);
    }
  };

  try {
    svmp::FE::timestepping::NewtonOptions options;
    options.residual_op = "equations";
    options.jacobian_op = "equations";
    options.max_iterations = 1;
    options.min_iterations = 0;
    options.abs_tolerance =
        std::numeric_limits<double>::max();
    options.rel_tolerance = 0.0;
    options.step_tolerance = 0.0;
    options.use_line_search = false;
    options.assemble_both_when_possible = false;
    options.initial_residual_only_certificate =
        !initialize_compatible_pressure;
    options.external_state_fixed_point.enabled = false;
    options.read_static_free_surface_environment_options = false;
    options.initialize_static_compatible_free_surface_pressure =
        initialize_compatible_pressure;
    options
        .accepted_static_pressure_representability_max_relative_distance =
        static_cast<double>(certificate_options
                                .pressure_representability_max_relative_distance);

    PdeVelocityExtensionCaches probe_pde_extension_caches;
    bool probe_extension_maps_refreshed = false;
    if (!algebraic_extension_requests.empty()) {
      using svmp::FE::timestepping::FunctionGeneratedStateParticipant;
      using svmp::FE::timestepping::GeneratedStateRole;
      using svmp::FE::timestepping::GeneratedStateSync;
      const auto generated_state =
          std::make_shared<svmp::FE::timestepping::GeneratedStateRegistry>(
              svmp::FE::timestepping::GeneratedStatePolicy{
                  .solve = svmp::FE::timestepping::GeneratedStateSolve::Steady});
      generated_state->add(std::make_shared<FunctionGeneratedStateParticipant>(
          "level_set_advection_velocity",
          GeneratedStateRole::Extension,
          FunctionGeneratedStateParticipant::Callbacks{
              .synchronize =
                  [&](const svmp::FE::systems::SystemStateView& state,
                      GeneratedStateSync& sync) {
                    if (probe_extension_maps_refreshed ||
                        sync.point !=
                            svmp::FE::timestepping::NewtonOptions::
                                StateSynchronizationPoint::
                                    AcceptedNonlinearState) {
                      return;
                    }
                    (void)updateLevelSetAdvectionVelocitiesFromState(
                        sim,
                        state,
                        algebraic_extension_requests,
                        /*refresh_frozen_algebraic_map=*/true,
                        &probe_pde_extension_caches);
                    probe_extension_maps_refreshed = true;
                  }}));
      generated_state->bind(options);
    }

    svmp::FE::timestepping::NewtonSolver probe(options);
    svmp::FE::timestepping::NewtonWorkspace workspace;
    probe.allocateWorkspace(system, *sim.backend, workspace);

    const auto history_depth =
        std::max(2, sim.time_history->historyDepth());
    auto history = svmp::FE::timestepping::TimeHistory::allocate(
        *sim.backend,
        static_cast<svmp::FE::GlobalIndex>(candidate.size()),
        history_depth,
        sim.time_history->hasSecondOrderState() ||
            system.temporalOrder() > 1);
    scatterFeOrderedSolution(history.u(), candidate);
    for (int slot = 1; slot <= history.historyDepth(); ++slot) {
      scatterFeOrderedSolution(history.uPrevK(slot), candidate);
    }
    const double dt =
        std::isfinite(sim.time_history->dt()) &&
                sim.time_history->dt() > 0.0
            ? sim.time_history->dt()
            : 1.0;
    history.setTime(sim.time_history->time());
    history.setDt(dt);
    history.setPrevDt(dt);
    history.primeDtHistory(dt);
    history.setStepIndex(sim.time_history->stepIndex());
    history.updateGhosts();

    const auto integrator =
        std::make_shared<const ZeroTimeDerivativeIntegrator>();
    svmp::FE::systems::TransientSystem transient(
        system, integrator);
    auto report = probe.solveStep(
        transient,
        *sim.linear_solver,
        history.time(),
        history,
        workspace);
    auto certified_solution =
        captureFeOrderedVectorCollectively(
            history.u(), comm);
    restore_system_state();
    return StaticCapillaryPressureCertificate{
        .report = std::move(report),
        .certified_solution = std::move(certified_solution),
    };
  } catch (...) {
    const auto failure = std::current_exception();
    try {
      restore_system_state();
    } catch (...) {
      std::throw_with_nested(std::runtime_error(
          "[svMultiPhysics::Application] Static capillary pressure certificate failed and could not restore auxiliary solver state."));
    }
    std::rethrow_exception(failure);
  }
}

void zeroHistoryFieldRateSlice(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field,
    svmp::FE::backends::GenericVector& rate,
    const svmp::MeshComm& comm)
{
  auto values = captureFeOrderedVectorCollectively(
      rate, comm);
  const auto raw_offset = system.fieldDofOffset(field);
  const auto raw_count =
      system.fieldDofHandler(field).getNumDofs();
  if (raw_offset < 0 || raw_count < 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization found an invalid rate-field layout.");
  }
  const auto offset = static_cast<std::size_t>(raw_offset);
  const auto count = static_cast<std::size_t>(raw_count);
  if (offset > values.size() ||
      count > values.size() - offset) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization rate-field slice exceeds the FE layout.");
  }
  std::fill(
      values.begin() + static_cast<std::ptrdiff_t>(offset),
      values.begin() +
          static_cast<std::ptrdiff_t>(offset + count),
      svmp::FE::Real{0.0});
  scatterFeOrderedSolution(rate, values);
}

std::string staticCapillaryHistoryPublicationTargetLayoutError(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId level_set_field,
    svmp::FE::FieldId pressure_field,
    std::span<const svmp::FE::Real> certified_solution,
    const svmp::FE::timestepping::TimeHistory& history)
{
  if (level_set_field == svmp::FE::INVALID_FIELD_ID ||
      pressure_field == svmp::FE::INVALID_FIELD_ID ||
      level_set_field == pressure_field ||
      history.historyDepth() < 0 ||
      certified_solution.size() !=
          static_cast<std::size_t>(history.u().size())) {
    return "[svMultiPhysics::Application] Static capillary publication received an invalid solution or field layout.";
  }

  for (const auto field : {level_set_field, pressure_field}) {
    const auto raw_field_offset = system.fieldDofOffset(field);
    const auto raw_field_count =
        system.fieldDofHandler(field).getNumDofs();
    if (raw_field_offset < 0 || raw_field_count <= 0) {
      return "[svMultiPhysics::Application] Static capillary publication received an invalid stationary-field layout.";
    }
    const auto field_offset =
        static_cast<std::size_t>(raw_field_offset);
    const auto field_count =
        static_cast<std::size_t>(raw_field_count);
    if (field_offset > certified_solution.size() ||
        field_count >
            certified_solution.size() - field_offset) {
      return "[svMultiPhysics::Application] Static capillary publication stationary-field slice exceeds the FE layout.";
    }
  }

  for (int slot = 1; slot <= history.historyDepth(); ++slot) {
    if (static_cast<std::size_t>(
            history.uPrevK(slot).size()) !=
            certified_solution.size()) {
      return "[svMultiPhysics::Application] Static capillary publication target-history layout differs from the certified solution.";
    }
  }
  if ((history.hasUDotState() &&
       static_cast<std::size_t>(history.uDot().size()) !=
           certified_solution.size()) ||
      (history.hasUDDotState() &&
       static_cast<std::size_t>(history.uDDot().size()) !=
           certified_solution.size())) {
    return "[svMultiPhysics::Application] Static capillary publication rate layout differs from the certified solution.";
  }
  return {};
}

std::string staticCapillaryHistoryPublicationLayoutError(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId level_set_field,
    svmp::FE::FieldId pressure_field,
    std::span<const svmp::FE::Real> certified_solution,
    std::span<const std::vector<svmp::FE::Real>>
        preserved_history,
    const svmp::FE::timestepping::TimeHistory& history)
{
  const auto target_error =
      staticCapillaryHistoryPublicationTargetLayoutError(
          system,
          level_set_field,
          pressure_field,
          certified_solution,
          history);
  if (!target_error.empty()) {
    return target_error;
  }
  if (preserved_history.size() !=
      static_cast<std::size_t>(history.historyDepth())) {
    return "[svMultiPhysics::Application] Static capillary publication preserved-history depth differs from the target history.";
  }
  for (const auto& values : preserved_history) {
    if (values.size() != certified_solution.size()) {
      return "[svMultiPhysics::Application] Static capillary publication preserved-history layout differs from the certified solution.";
    }
  }
  return {};
}

void stageStaticCapillaryHistoryForPublication(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId level_set_field,
    svmp::FE::FieldId pressure_field,
    std::span<const svmp::FE::Real> certified_solution,
    std::span<const std::vector<svmp::FE::Real>>
        preserved_history,
    svmp::FE::timestepping::TimeHistory& history)
{
  const auto layout_error =
      staticCapillaryHistoryPublicationLayoutError(
          system,
          level_set_field,
          pressure_field,
          certified_solution,
          preserved_history,
          history);
  if (!layout_error.empty()) {
    throw std::runtime_error(layout_error);
  }

  const auto comm = activeFESystemCommunicator(system);

  scatterFeOrderedSolution(history.u(), certified_solution);
  for (int slot = 1; slot <= history.historyDepth(); ++slot) {
    auto accepted_history = preserved_history[
        static_cast<std::size_t>(slot - 1)];
    for (const auto field : {level_set_field, pressure_field}) {
      const auto field_offset = static_cast<std::size_t>(
          system.fieldDofOffset(field));
      const auto field_count = static_cast<std::size_t>(
          system.fieldDofHandler(field).getNumDofs());
      std::copy(
          certified_solution.begin() +
              static_cast<std::ptrdiff_t>(field_offset),
          certified_solution.begin() +
              static_cast<std::ptrdiff_t>(
                  field_offset + field_count),
          accepted_history.begin() +
              static_cast<std::ptrdiff_t>(field_offset));
    }
    scatterFeOrderedSolution(
        history.uPrevK(slot), accepted_history);
  }
  if (history.hasUDotState()) {
    for (const auto field : {level_set_field, pressure_field}) {
      zeroHistoryFieldRateSlice(
          system, field, history.uDot(), comm);
    }
  }
  if (history.hasUDDotState()) {
    for (const auto field : {level_set_field, pressure_field}) {
      zeroHistoryFieldRateSlice(
          system, field, history.uDDot(), comm);
    }
  }
  history.updateGhosts();
}

struct StaticCapillaryFunctionalCandidate {
  svmp::FE::level_set::LevelSetVolumeConstrainedEvaluation
      evaluation;
  std::vector<svmp::FE::Real> candidate;
};

// The static cap as the minimizer's problem: two energy components, and
// three pressure certificates of the unprojected physical-potential load,
// in this order.
enum StaticCapillaryEnergyComponent : std::size_t {
  kStaticCapillarySurfaceWallEnergy = 0u,
  kStaticCapillaryGravitationalPotentialEnergy = 1u,
};

enum StaticCapillaryCertificate : std::size_t {
  kStaticCapillaryPressureRepresentability = 0u,
  kStaticCapillaryPhysicalEquilibrium = 1u,
  kStaticCapillaryConstantPressureKkt = 2u,
};

svmp::FE::level_set::LevelSetVolumeConstrainedProblem
staticCapillaryEquilibriumProblem(
    const application::core::StaticCapillaryEquilibriumOptions& options)
{
  svmp::FE::level_set::LevelSetVolumeConstrainedProblem problem;
  problem.energy_components = {"surface_wall_energy",
                               "gravitational_potential_energy"};
  // The constrained pressure-space certificate.
  svmp::FE::level_set::LevelSetVolumeConstrainedAcceptanceCheck pressure;
  pressure.name = "pressure_representability";
  pressure.limits = {options.pressure_representability_max_residual_norm,
                     options.pressure_representability_max_relative_distance};
  pressure.invalid_state_diagnostic =
      "candidate_evaluator_returned_invalid_pressure_representability_state";
  pressure.unavailable_diagnostic =
      "pressure_representability_unavailable_at_parameter_stationary_geometry";
  pressure.failed_diagnostic =
      "pressure_representability_gate_failed_at_parameter_stationary_geometry";
  // The production residual (always measured with the certificate).
  svmp::FE::level_set::LevelSetVolumeConstrainedAcceptanceCheck equilibrium;
  equilibrium.name = "physical_equilibrium_residual";
  equilibrium.limits = {options.physical_equilibrium_max_residual_norm};
  equilibrium.failed_diagnostic =
      "physical_equilibrium_residual_gate_failed_at_parameter_stationary_geometry";
  // The constant-pressure KKT certificate, required when the pressure
  // constraints keep the constant mode at zero gravity.
  svmp::FE::level_set::LevelSetVolumeConstrainedAcceptanceCheck kkt;
  kkt.name = "constant_pressure_kkt";
  kkt.limits = {options.constant_pressure_kkt_max_residual_norm,
                options.constant_pressure_kkt_max_relative_distance};
  kkt.invalid_state_diagnostic =
      "candidate_evaluator_returned_invalid_constant_pressure_kkt_state";
  kkt.unavailable_diagnostic =
      "constant_pressure_kkt_unavailable_at_parameter_stationary_geometry";
  kkt.failed_diagnostic =
      "constant_pressure_kkt_gate_failed_at_parameter_stationary_geometry";
  problem.acceptance_checks = {std::move(pressure), std::move(equilibrium),
                               std::move(kkt)};
  problem.forbidden_substitution_evaluation_diagnostic =
      "candidate_evaluator_projected_production_force";
  problem.forbidden_substitution_diagnostic =
      "production_force_projection_is_forbidden";
  return problem;
}

// The certificates of a functional trial, which measures none of them.
std::vector<svmp::FE::level_set::LevelSetVolumeConstrainedCheckValue>
unmeasuredStaticCapillaryCertificates()
{
  constexpr auto nan = std::numeric_limits<svmp::FE::Real>::quiet_NaN();
  std::vector<svmp::FE::level_set::LevelSetVolumeConstrainedCheckValue>
      checks(3u);
  checks[kStaticCapillaryPressureRepresentability].converged = false;
  checks[kStaticCapillaryPressureRepresentability].values = {nan, nan};
  checks[kStaticCapillaryPhysicalEquilibrium].available = true;
  checks[kStaticCapillaryPhysicalEquilibrium].values = {nan};
  checks[kStaticCapillaryConstantPressureKkt].values = {nan, nan};
  return checks;
}

// The minimizer's final certificates under their Navier-Stokes names (the
// unmeasured values before the initial candidate is evaluated).
struct StaticCapillaryCertificateValues {
  bool pressure_representability_available{false};
  bool pressure_representability_converged{false};
  bool pressure_representability_breakdown{false};
  svmp::FE::Real pressure_representability_residual_norm{
      std::numeric_limits<svmp::FE::Real>::quiet_NaN()};
  svmp::FE::Real pressure_representability_relative_distance{
      std::numeric_limits<svmp::FE::Real>::quiet_NaN()};
  svmp::FE::Real production_residual_norm{
      std::numeric_limits<svmp::FE::Real>::quiet_NaN()};
  bool constant_pressure_kkt_required{true};
  bool constant_pressure_kkt_available{false};
  svmp::FE::Real constant_pressure_kkt_residual_norm{
      std::numeric_limits<svmp::FE::Real>::quiet_NaN()};
  svmp::FE::Real constant_pressure_kkt_relative_distance{
      std::numeric_limits<svmp::FE::Real>::quiet_NaN()};
};

StaticCapillaryCertificateValues staticCapillaryCertificateValues(
    std::span<const svmp::FE::level_set::LevelSetVolumeConstrainedCheckValue>
        checks)
{
  StaticCapillaryCertificateValues values;
  if (checks.size() != 3u) {
    return values;
  }
  const auto& pressure = checks[kStaticCapillaryPressureRepresentability];
  const auto& equilibrium = checks[kStaticCapillaryPhysicalEquilibrium];
  const auto& kkt = checks[kStaticCapillaryConstantPressureKkt];
  values.pressure_representability_available = pressure.available;
  values.pressure_representability_converged = pressure.converged;
  values.pressure_representability_breakdown = pressure.breakdown;
  values.pressure_representability_residual_norm = pressure.values[0];
  values.pressure_representability_relative_distance = pressure.values[1];
  values.production_residual_norm = equilibrium.values[0];
  values.constant_pressure_kkt_required = kkt.required;
  values.constant_pressure_kkt_available = kkt.available;
  values.constant_pressure_kkt_residual_norm = kkt.values[0];
  values.constant_pressure_kkt_relative_distance = kkt.values[1];
  return values;
}

// One energy component of the minimizer's result (NaN before the initial
// candidate is evaluated).
svmp::FE::Real staticCapillaryEnergy(
    std::span<const svmp::FE::Real> components,
    StaticCapillaryEnergyComponent component)
{
  return component < components.size()
             ? components[component]
             : std::numeric_limits<svmp::FE::Real>::quiet_NaN();
}

StaticCapillaryFunctionalCandidate
evaluateStaticCapillaryFunctionalCandidate(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    const LevelSetMaintenanceRequest& request,
    const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration&
        declaration,
    std::span<const svmp::FE::Real> baseline,
    LevelSetMaintenanceGeometryTransaction& transaction,
    std::span<const svmp::FE::Real> coefficients,
    svmp::FE::level_set::LevelSetVolumeConstrainedEvaluationPurpose purpose)
{
  StaticCapillaryFunctionalCandidate result;
  auto& evaluation = result.evaluation;
  auto& candidate = result.candidate;
  auto& system = *sim.fe_system;
  auto& history = *sim.time_history;
  const auto comm = activeFESystemCommunicator(system);
  const auto level_set_field = declaration.level_set_field;
  const auto marker = declaration.interface_marker;
  const auto raw_field_offset = system.fieldDofOffset(level_set_field);
  const auto raw_field_count =
      system.fieldDofHandler(level_set_field).getNumDofs();
  const auto field_offset = static_cast<std::size_t>(raw_field_offset);
  const auto field_count = static_cast<std::size_t>(raw_field_count);
  const double constraint_dt =
      std::isfinite(history.dt()) && history.dt() > 0.0
          ? history.dt()
          : 1.0;

  if (globalAnyBool(coefficients.size() != field_count, comm)) {
    evaluation.diagnostic =
        "candidate_level_set_coefficient_count_mismatch";
    return result;
  }
  (void)collectiveLevelSetMaintenanceAlgebraicRevision(
      coefficients, comm);

  candidate.assign(baseline.begin(), baseline.end());
  std::copy(
      coefficients.begin(),
      coefficients.end(),
      candidate.begin() + static_cast<std::ptrdiff_t>(field_offset));
  std::optional<std::uint64_t> previous_constraint_key;
  std::uint64_t constraint_key = 0u;
  ActiveCutContextRefreshReport refresh_report;
  bool fixed_point_reached = false;
  constexpr int maximum_projection_passes = 4;
  for (int pass = 0; pass < maximum_projection_passes; ++pass) {
    refresh_report = transaction.refresh(
        params,
        candidate,
        purpose ==
                    svmp::FE::level_set::
                        LevelSetVolumeConstrainedEvaluationPurpose::
                            AcceptanceCertificate &&
                pass == 0);
    if (refresh_report.topology_key == 0u) {
      evaluation.diagnostic =
          "candidate_cut_topology_key_unavailable";
      return result;
    }
    system.updateConstraints(history.time(), constraint_dt);
    constraint_key = collectiveConstraintSemanticsKey(system, comm);
    auto projected = candidate;
    system.constraints().distribute(projected);
    const bool projected_values_finite = std::all_of(
        projected.begin(), projected.end(), [](svmp::FE::Real value) {
          return std::isfinite(value);
        });
    if (globalAnyBool(!projected_values_finite, comm)) {
      evaluation.diagnostic =
          "candidate_constraint_projection_is_nonfinite";
      return result;
    }
    const auto projection_change =
        globalMaxAbsDifference(candidate, projected, comm);
    candidate = std::move(projected);
    if (pass > 0 && previous_constraint_key.has_value() &&
        *previous_constraint_key == constraint_key &&
        projection_change == 0.0) {
      fixed_point_reached = true;
      break;
    }
    previous_constraint_key = constraint_key;
  }
  if (!fixed_point_reached) {
    evaluation.diagnostic =
        "candidate_constraint_geometry_fixed_point_failed";
    return result;
  }

  auto states = currentAcceptedFunctionalLayouts(sim);
  // D49 step E4: the volume and the energies are the declared functionals
  // of the interface (measured 0-forms, exact reduction) on the candidate:
  // the liquid volume, the liquid-gas surface energy plus the Young energy
  // of the walls (one form), and the gravitational energy.
  const std::array<std::vector<std::string_view>, 3> declared_quantities{{
      {"liquid_volume"},
      {"liquid_gas_surface_energy", "young_wall_energy"},
      {"gravitational_energy"}}};
  const auto declared_values = evaluateDeclaredFreeSurfaceFunctionals(
      sim, marker, declared_quantities, candidate);
  if (!declared_values.has_value()) {
    evaluation.diagnostic =
        "candidate_declared_functional_unavailable";
    return result;
  }
  const bool local_functional_available =
      states.size() == 1u &&
      states.front().interface_marker == marker;
  if (globalAnyBool(!local_functional_available, comm)) {
    evaluation.diagnostic =
        "candidate_authoritative_functional_unavailable";
    return result;
  }
  const auto& state = states.front();
  const auto volume = (*declared_values)[0];
  const auto energy = (*declared_values)[1];
  const auto gravitational_energy = (*declared_values)[2];
  const bool local_functional_valid =
      std::isfinite(energy) && std::isfinite(gravitational_energy) &&
      std::isfinite(energy + gravitational_energy) &&
      std::isfinite(volume) && volume >= svmp::FE::Real{0.0};
  if (globalAnyBool(!local_functional_valid, comm)) {
    evaluation.diagnostic =
        "candidate_authoritative_functional_is_nonfinite";
    return result;
  }
  const auto [minimum_snapshot_revision, maximum_snapshot_revision] =
      globalMinMaxUint64(
          state.geometry_revision.snapshot_revision_key, comm);
  const bool functional_decision_data_identical =
      minimum_snapshot_revision == maximum_snapshot_revision &&
      globalMinDouble(static_cast<double>(energy), comm) ==
          globalMaxDouble(static_cast<double>(energy), comm) &&
      globalMinDouble(static_cast<double>(gravitational_energy), comm) ==
          globalMaxDouble(static_cast<double>(gravitational_energy), comm) &&
      globalMinDouble(static_cast<double>(volume), comm) ==
          globalMaxDouble(static_cast<double>(volume), comm);
  if (!functional_decision_data_identical) {
    evaluation.diagnostic =
        "candidate_functional_decision_data_differs_across_communicator";
    return result;
  }

  evaluation.success = true;
  evaluation.snapshot_revision_key =
      state.geometry_revision.snapshot_revision_key;
  evaluation.cut_topology_key = refresh_report.topology_key;
  evaluation.constraint_semantics_key = constraint_key;
  evaluation.energy_components = {energy, gravitational_energy};
  evaluation.liquid_volume = volume;
  // Difference gradients: the functional supplies no derivatives (the
  // kinematic-area-gradient producer of exact derivatives was retired with
  // decision D48).
  evaluation.functional_derivatives_available = false;
  evaluation.checks = unmeasuredStaticCapillaryCertificates();
  evaluation.forbidden_substitution_applied = false;
  evaluation.diagnostic = "functional_trial_available";
  return result;
}

void appendStaticCapillaryFailureState(
    std::ostream& stream,
    const svmp::FE::level_set::LevelSetVolumeConstrainedFailureState& state)
{
  using Status =
      svmp::FE::level_set::LevelSetVolumeConstrainedFailureStateStatus;
  constexpr auto capacity =
      svmp::FE::level_set::
          kLevelSetVolumeConstrainedFailureStateCoefficientCapacity;

  bool recognized_status = true;
  const char* status_name = "invalid_record";
  switch (state.status) {
  case Status::NotApplicable:
    status_name = "not_applicable";
    break;
  case Status::Available:
    status_name = "available";
    break;
  case Status::CapacityExceeded:
    status_name = "capacity_exceeded";
    break;
  case Status::RecordingFailed:
    status_name = "recording_failed";
    break;
  default:
    recognized_status = false;
    break;
  }

  bool available_record_valid = false;
  if (state.status == Status::Available) {
    const bool shape_valid =
        state.coefficient_count <= capacity &&
        state.active_coefficient_count <= capacity &&
        state.coefficients.size() == state.coefficient_count &&
        state.active_coefficient_indices.size() ==
            state.active_coefficient_count &&
        state.energy_gradient.size() == state.active_coefficient_count &&
        state.volume_gradient.size() == state.active_coefficient_count &&
        state.direction.size() == state.active_coefficient_count;
    if (shape_valid) {
      const auto finite_values = [](const auto& values) {
        return std::all_of(
            values.begin(), values.end(), [](const auto value) {
              return std::isfinite(value);
            });
      };
      const bool finite =
          std::isfinite(state.current_merit) &&
          std::isfinite(state.merit_penalty) &&
          finite_values(state.coefficients) &&
          finite_values(state.energy_gradient) &&
          finite_values(state.volume_gradient) &&
          finite_values(state.direction);
      bool indices_valid = true;
      for (std::size_t active = 0u;
           finite && active < state.active_coefficient_indices.size();
           ++active) {
        const auto coefficient = state.active_coefficient_indices[active];
        if (coefficient >= state.coefficient_count ||
            (active > 0u &&
             coefficient <= state.active_coefficient_indices[active - 1u])) {
          indices_valid = false;
          break;
        }
      }
      available_record_valid = finite && indices_valid;
    }
    if (!available_record_valid) {
      status_name = "invalid_record";
    }
  } else if (!recognized_status) {
    status_name = "invalid_record";
  }

  std::ostringstream formatted;
  formatted.imbue(std::locale::classic());
  formatted << std::setprecision(
      std::numeric_limits<svmp::FE::Real>::max_digits10);
  formatted
      << " line_search_failure_state_summary={status=" << status_name
      << ",coefficient_count=" << state.coefficient_count
      << ",active_count=" << state.active_coefficient_count
      << ",capacity=" << capacity
      << ",accepted_iteration_index=" << state.accepted_iteration_index
      << ",snapshot_revision_key=" << state.snapshot_revision_key
      << ",cut_topology_key=" << state.cut_topology_key
      << ",constraint_semantics_key=" << state.constraint_semantics_key
      << ",current_merit=" << state.current_merit
      << ",merit_penalty=" << state.merit_penalty << "}";
  if (available_record_valid) {
    for (std::size_t index = 0u;
         index < state.coefficients.size();
         ++index) {
      formatted
          << " line_search_failure_coefficient={index=" << index
          << ",value=" << state.coefficients[index] << "}";
    }
    for (std::size_t active = 0u;
         active < state.active_coefficient_indices.size();
         ++active) {
      formatted
          << " line_search_failure_active={index=" << active
          << ",coefficient_index="
          << state.active_coefficient_indices[active]
          << ",energy_gradient=" << state.energy_gradient[active]
          << ",volume_gradient=" << state.volume_gradient[active]
          << ",direction=" << state.direction[active] << "}";
    }
  }
  stream << formatted.str();
}

// extension_requests: the advection velocity extensions of the solve; the
// acceptance certificate's probe refreshes the frozen maps of the algebraic
// ones (evaluateStaticCapillaryPressureCertificate).
bool initializeDiscreteStaticCapillaryEquilibrium(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    std::vector<LevelSetMaintenanceRequest>& requests,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& refresh_cache,
    const std::vector<LevelSetAdvectionVelocityRequest>& extension_requests = {})
{
  const auto enabled_count = static_cast<std::size_t>(std::count_if(
      requests.begin(), requests.end(), [](const auto& request) {
        return request.configuration->static_capillary_equilibrium_enabled &&
               !request.runtime.static_capillary_equilibrium_initialized;
      }));
  if (enabled_count == 0u) {
    return false;
  }
  if (enabled_count != 1u || !sim.fe_system || !sim.primary_mesh ||
      !sim.time_history || !sim.backend || !sim.linear_solver) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization currently requires exactly one enabled level-set request and a complete FE solve context.");
  }
  const auto active_requests = activeCutVolumeRequests(params);
  if (active_requests.size() != 1u) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization currently requires exactly one authoritative active cut-volume request.");
  }

  auto request_it = std::find_if(
      requests.begin(), requests.end(), [](const auto& request) {
        return request.configuration->static_capillary_equilibrium_enabled &&
               !request.runtime.static_capillary_equilibrium_initialized;
      });
  auto& request = *request_it;
  if (!request.configuration->volume_cut_request.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization requires a matching authoritative cut-volume request.");
  }

  auto& system = *sim.fe_system;
  auto& history = *sim.time_history;
  const auto comm = activeFESystemCommunicator(system);
  const auto level_set_field =
      system.findFieldByName(request.configuration->transport.level_set.field_name);
  if (level_set_field == svmp::FE::INVALID_FIELD_ID) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization could not resolve level-set field '" +
        request.configuration->transport.level_set.field_name + "'.");
  }
  const auto marker =
      generatedCutContextMarkerForMaintenance(system, request);
  if (globalAnyBool(!marker.has_value(), comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization could not resolve its authoritative interface marker.");
  }

  const auto declarations =
      svmp::FE::systems::functionalRecords(system).functionalDeclarations();
  const bool local_declaration_matches =
      declarations.size() == 1u &&
      declarations.front().interface_marker == *marker &&
      declarations.front().level_set_field == level_set_field &&
      declarations.front().geometry_domain_id ==
          request.configuration->volume_cut_request->domain_id &&
      declarations.front().parameters.liquid_side ==
          cutIntegrationSide(
              request.configuration->volume_cut_request->active_side) &&
      declarations.front().parameters.surface_tension >
          svmp::FE::Real{0.0} &&
      declarations.front().capillary_balance_method ==
          svmp::FE::systems::FreeSurfaceCapillaryBalanceMethod::
              DiscreteEnergyVolumeStationarity &&
      declarations.front().capillary_balance_qualification ==
          svmp::FE::systems::FreeSurfaceCapillaryBalanceQualification::
              PrerequisiteOnly;
  if (globalAnyBool(!local_declaration_matches, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization requires one matching positive-tension, active-side-consistent, prerequisite-only discrete-energy free-surface declaration.");
  }
  const auto& declaration = declarations.front();

  const auto declaration_content_key =
      acceptedFunctionalDeclarationContentKey(declaration);
  const auto [minimum_declaration_content_key,
              maximum_declaration_content_key] =
      globalMinMaxUint64(declaration_content_key, comm);
  if (minimum_declaration_content_key !=
      maximum_declaration_content_key) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization free-surface declaration content differs across the FE communicator.");
  }

  const bool local_body_force_contract_incomplete =
      !declaration.active_volume_energy_parameters.has_value() ||
      !declaration.static_conservative_body_force_complete;
  if (globalAnyBool(local_body_force_contract_incomplete, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization requires an active-volume energy declaration whose gravitational acceleration is the complete velocity-independent body acceleration in the production static momentum residual.");
  }
  const bool gravitational_acceleration_is_zero =
      std::all_of(
          declaration.active_volume_energy_parameters
              ->gravitational_acceleration.begin(),
          declaration.active_volume_energy_parameters
              ->gravitational_acceleration.end(),
          [](svmp::FE::Real value) {
            return value == svmp::FE::Real{0.0};
          });

  const auto baseline =
      captureFeOrderedVectorCollectively(
          history.u(), comm);
  const bool local_velocity_field_declared =
      declaration.velocity_field != svmp::FE::INVALID_FIELD_ID;
  if (globalMinDouble(
          local_velocity_field_declared ? 1.0 : 0.0, comm) !=
      globalMaxDouble(
          local_velocity_field_declared ? 1.0 : 0.0, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization velocity-field availability differs across the FE communicator.");
  }
  if (!local_velocity_field_declared) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization requires a declared velocity field so the zero-static-state preflight cannot be bypassed.");
  }

  svmp::FE::GlobalIndex raw_velocity_offset{-1};
  svmp::FE::GlobalIndex raw_velocity_count{0};
  std::string velocity_field_name;
  bool local_velocity_metadata_valid = false;
  try {
    const auto unknown_fields =
        system.unknownFieldIdsInDofMapOrder();
    const auto velocity_is_unknown =
        std::find(
            unknown_fields.begin(),
            unknown_fields.end(),
            declaration.velocity_field) != unknown_fields.end();
    if (velocity_is_unknown) {
      const auto& velocity_record =
          system.fieldRecord(declaration.velocity_field);
      const auto& velocity_dofs =
          system.fieldDofHandler(declaration.velocity_field);
      raw_velocity_offset =
          system.fieldDofOffset(declaration.velocity_field);
      raw_velocity_count = velocity_dofs.getNumDofs();
      const int dimension = system.meshAccess().dimension();
      local_velocity_metadata_valid =
          dimension >= 1 && dimension <= 3 &&
          velocity_record.id == declaration.velocity_field &&
          velocity_record.scope ==
              svmp::FE::systems::FieldScope::VolumeCell &&
          velocity_record.space != nullptr &&
          velocity_record.components >= dimension &&
          velocity_record.space->value_dimension() >= dimension &&
          raw_velocity_offset >= 0 && raw_velocity_count > 0;
      velocity_field_name = velocity_record.name;
    }
  } catch (...) {
    local_velocity_metadata_valid = false;
  }
  if (globalAnyBool(!local_velocity_metadata_valid, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization requires its declared velocity field to be a registered, finalized, dimension-compatible unknown vector volume field on every rank.");
  }
  {
    const bool local_velocity_layout_valid =
        raw_velocity_offset >= 0 && raw_velocity_count > 0 &&
        static_cast<std::size_t>(raw_velocity_offset) <= baseline.size() &&
        static_cast<std::size_t>(raw_velocity_count) <=
            baseline.size() -
                static_cast<std::size_t>(raw_velocity_offset);
    if (globalAnyBool(!local_velocity_layout_valid, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Static capillary initialization found an invalid declared velocity-field layout.");
    }
    const auto velocity_begin =
        baseline.begin() +
        static_cast<std::ptrdiff_t>(raw_velocity_offset);
    const auto velocity_end =
        velocity_begin +
        static_cast<std::ptrdiff_t>(raw_velocity_count);
    const bool local_velocity_is_static =
        std::all_of(velocity_begin, velocity_end, [](svmp::FE::Real value) {
          return std::isfinite(value) &&
                 value == svmp::FE::Real{0.0};
        });
    if (globalAnyBool(!local_velocity_is_static, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Static capillary initialization requires an exactly zero finite current velocity field because the current discrete minimizer and physical certificate omit kinetic, inertial, advective, and viscous contributions; field '" +
          velocity_field_name + "'.");
    }
  }

  const auto initial_states =
      currentAcceptedFunctionalLayouts(sim);
  const bool local_initial_functional_available =
      initial_states.size() == 1u &&
      initial_states.front().interface_marker == *marker;
  if (globalAnyBool(
          !local_initial_functional_available, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization could not evaluate its initial authoritative functional.");
  }
  // D49 step E4: the target is the declared liquid volume of the initial
  // geometry (a measured 0-form that reads no field).
  const std::array<std::vector<std::string_view>, 1> volume_quantity{{
      {"liquid_volume"}}};
  const auto declared_volume = evaluateDeclaredFreeSurfaceFunctionals(
      sim, *marker, volume_quantity, {});
  if (!declared_volume.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization requires the declared liquid volume free_surface." +
        std::to_string(*marker) +
        ".liquid_volume of its interface; the minimizer reads the functionals the formulation declares.");
  }
  const auto target_volume = declared_volume->front();
  const bool local_target_volume_valid =
      std::isfinite(target_volume) &&
      target_volume > svmp::FE::Real{0.0};
  if (globalAnyBool(!local_target_volume_valid, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization requires a finite positive initial liquid volume.");
  }
  if (globalMinDouble(
          static_cast<double>(target_volume), comm) !=
      globalMaxDouble(
          static_cast<double>(target_volume), comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization target volume differs across the FE communicator.");
  }
  request.runtime.measured_static_capillary_target =
      target_volume;
  const auto capillary_options = effectiveStaticCapillaryEquilibriumOptions(request);

  const auto raw_field_offset =
      system.fieldDofOffset(level_set_field);
  const auto raw_field_count =
      system.fieldDofHandler(level_set_field).getNumDofs();
  if (globalAnyBool(
          raw_field_offset < 0 || raw_field_count <= 0,
          comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization found an invalid level-set coefficient layout.");
  }
  const auto field_offset =
      static_cast<std::size_t>(raw_field_offset);
  const auto field_count =
      static_cast<std::size_t>(raw_field_count);
  const bool local_field_layout_valid =
      field_offset <= baseline.size() &&
      field_count <= baseline.size() - field_offset;
  if (globalAnyBool(!local_field_layout_valid, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization level-set coefficients exceed the FE solution layout.");
  }

  std::vector<svmp::FE::Real> parameters(
      baseline.begin() +
          static_cast<std::ptrdiff_t>(field_offset),
      baseline.begin() +
          static_cast<std::ptrdiff_t>(
              field_offset + field_count));
  (void)collectiveLevelSetMaintenanceAlgebraicRevision(
      parameters, comm);

  const auto [minimum_field_count, maximum_field_count] =
      globalMinMaxUint64(
          static_cast<std::uint64_t>(field_count), comm);
  if (minimum_field_count != maximum_field_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization level-set coefficient count differs across the FE communicator.");
  }

  // The difference gradients stay in the initial cut-topology epoch and keep
  // the authoritative cut-cell support of the interface.
  std::vector<std::size_t> local_active_parameters;
  bool local_active_support_invalid = false;
  try {
    const auto* cut_context = system.cutIntegrationContext();
    if (cut_context == nullptr ||
        !cut_context->hasFreeSurfaceGeometrySnapshotForMarker(*marker)) {
      local_active_support_invalid = true;
    } else {
      const auto bound_revision =
          cut_context->freeSurfaceGeometrySnapshotRevisionForMarker(
              *marker);
      const auto& snapshots =
          cut_context->freeSurfaceGeometrySnapshots();
      const auto snapshot = std::find_if(
          snapshots.begin(), snapshots.end(),
          [&](const auto& candidate) {
            return candidate != nullptr &&
                   candidate->revision().interface_marker == *marker &&
                   candidate->revision().snapshot_revision_key ==
                       bound_revision;
          });
      if (snapshot == snapshots.end()) {
        local_active_support_invalid = true;
      } else {
        const auto& field_dofs =
            system.fieldDofHandler(level_set_field);
        for (const auto cell :
             (*snapshot)->interfaceDomain().cutCells()) {
          for (const auto dof : field_dofs.getCellDofs(cell)) {
            if (dof < 0 ||
                static_cast<std::uint64_t>(dof) >=
                    static_cast<std::uint64_t>(field_count)) {
              local_active_support_invalid = true;
              continue;
            }
            local_active_parameters.push_back(
                static_cast<std::size_t>(dof));
          }
        }
      }
    }
  } catch (...) {
    local_active_support_invalid = true;
  }
  if (globalAnyBool(local_active_support_invalid, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization could not resolve its active level-set support on every FE communicator rank.");
  }
  auto active_parameters = communicatorWideIndexUnion(
      std::move(local_active_parameters),
      field_count,
      comm,
      "Static capillary active level-set support");
  if (active_parameters.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary initialization authoritative interface has no active level-set coefficients.");
  }

  LevelSetMaintenanceGeometryTransaction transaction(
      sim,
      lifecycle,
      refresh_cache,
      active_requests);
  auto minimization_options =
      capillary_options;
  minimization_options.allow_topology_epoch_transitions = false;
  std::vector<svmp::FE::Real> certified_solution;
  svmp::FE::FieldId certified_pressure_field{
      svmp::FE::INVALID_FIELD_ID};
  double certified_constant_pressure_jump =
      std::numeric_limits<double>::quiet_NaN();
  double certified_constant_pressure_volume_multiplier =
      std::numeric_limits<double>::quiet_NaN();
  bool certified_constant_pressure_multiplier_comparison_available = false;
  const auto evaluator =
      [&](std::span<const svmp::FE::Real> coefficients,
          svmp::FE::level_set::
              LevelSetVolumeConstrainedEvaluationPurpose purpose) {
        auto functional = evaluateStaticCapillaryFunctionalCandidate(
            sim,
            params,
            request,
            declaration,
            baseline,
            transaction,
            coefficients,
            purpose);
        auto evaluation = std::move(functional.evaluation);
        if (!evaluation.success) {
          return evaluation;
        }
        auto candidate = std::move(functional.candidate);
        const auto constraint_key =
            evaluation.constraint_semantics_key;

        if (purpose ==
            svmp::FE::level_set::
                LevelSetVolumeConstrainedEvaluationPurpose::
                    AcceptanceCertificate) {
          auto certificate =
              evaluateStaticCapillaryPressureCertificate(
                  sim,
                  candidate,
                  capillary_options,
                  /*initialize_compatible_pressure=*/true,
                  extension_requests);
          const auto& report = certificate.report;
          const auto post_certificate_constraint_key =
              collectiveConstraintSemanticsKey(system, comm);
          if (post_certificate_constraint_key !=
              constraint_key) {
            evaluation.success = false;
            evaluation.diagnostic =
                "pressure_certificate_changed_constraint_semantics";
            return evaluation;
          }
          const bool local_certified_solution_valid =
              certificate.certified_solution.size() == candidate.size() &&
              std::all_of(
                  certificate.certified_solution.begin(),
                  certificate.certified_solution.end(),
                  [](svmp::FE::Real value) {
                    return std::isfinite(value);
                  });
          if (globalAnyBool(
                  !local_certified_solution_valid, comm)) {
            evaluation.success = false;
            evaluation.diagnostic =
                "pressure_certificate_solution_is_invalid";
            return evaluation;
          }
          if (globalAnyBool(
                  !std::isfinite(report.residual_norm),
                  comm)) {
            evaluation.success = false;
            evaluation.diagnostic =
                "pressure_certificate_production_residual_is_nonfinite";
            return evaluation;
          }
          if (globalMinDouble(report.residual_norm, comm) !=
              globalMaxDouble(report.residual_norm, comm)) {
            evaluation.success = false;
            evaluation.diagnostic =
                "pressure_certificate_production_residual_differs_across_communicator";
            return evaluation;
          }
          const bool local_kkt_scalar_invalid =
              report.constant_pressure_kkt_available &&
              (!std::isfinite(
                   report
                       .constant_pressure_kkt_residual_norm) ||
               !std::isfinite(
                   report
                       .constant_pressure_kkt_relative_distance) ||
               !std::isfinite(
                   report.constant_pressure_kkt_pressure_jump) ||
               !std::isfinite(
                   report
                       .constant_pressure_kkt_volume_multiplier));
          if (globalAnyBool(
                  local_kkt_scalar_invalid, comm)) {
            evaluation.success = false;
            evaluation.diagnostic =
                "pressure_certificate_constant_kkt_scalar_is_nonfinite";
            return evaluation;
          }
          const bool local_diagnostic_sampled =
              report
                  .pressure_representability_diagnostic_sampled;
          const bool local_pressure_available =
              local_diagnostic_sampled &&
              report.pressure_representability_available;
          const bool local_pressure_field_invalid =
              local_pressure_available &&
              (report.pressure_representability_pressure_field ==
                   svmp::FE::INVALID_FIELD_ID ||
               report.pressure_representability_pressure_field ==
                   level_set_field);
          if (globalAnyBool(local_pressure_field_invalid, comm)) {
            evaluation.success = false;
            evaluation.diagnostic =
                "pressure_certificate_field_is_invalid";
            return evaluation;
          }
          const bool local_pressure_scalar_invalid =
              local_pressure_available &&
              (!std::isfinite(
                   report.pressure_representability_residual_norm) ||
               !std::isfinite(
                   report.pressure_representability_relative_distance));
          if (globalAnyBool(
                  local_pressure_scalar_invalid, comm)) {
            evaluation.success = false;
            evaluation.diagnostic =
                "pressure_certificate_representability_scalar_is_nonfinite";
            return evaluation;
          }
          const bool local_kkt_available =
              local_diagnostic_sampled &&
              report.constant_pressure_kkt_available;
          const bool diagnostic_sampled_identical =
              globalMinDouble(
                  local_diagnostic_sampled ? 1.0 : 0.0,
                  comm) ==
              globalMaxDouble(
                  local_diagnostic_sampled ? 1.0 : 0.0,
                  comm);
          const bool pressure_available_identical =
              globalMinDouble(
                  local_pressure_available ? 1.0 : 0.0,
                  comm) ==
              globalMaxDouble(
                  local_pressure_available ? 1.0 : 0.0,
                  comm);
          const bool pressure_field_identical =
              globalMinDouble(
                  static_cast<double>(
                      report.pressure_representability_pressure_field),
                  comm) ==
              globalMaxDouble(
                  static_cast<double>(
                      report.pressure_representability_pressure_field),
                  comm);
          const bool kkt_available_identical =
              globalMinDouble(
                  local_kkt_available ? 1.0 : 0.0,
                  comm) ==
              globalMaxDouble(
                  local_kkt_available ? 1.0 : 0.0,
                  comm);
          const bool pressure_convergence_identical =
              globalMinDouble(
                  report.pressure_representability_converged ? 1.0 : 0.0,
                  comm) ==
                  globalMaxDouble(
                      report.pressure_representability_converged ? 1.0 : 0.0,
                      comm) &&
              globalMinDouble(
                  report.pressure_representability_breakdown ? 1.0 : 0.0,
                  comm) ==
                  globalMaxDouble(
                      report.pressure_representability_breakdown ? 1.0 : 0.0,
                      comm);
          const bool constant_pressure_trace_flags_identical =
              globalMinDouble(
                  report
                          .constant_pressure_unit_coefficients_represent_constant
                      ? 1.0
                      : 0.0,
                  comm) ==
                  globalMaxDouble(
                      report
                              .constant_pressure_unit_coefficients_represent_constant
                          ? 1.0
                          : 0.0,
                      comm) &&
              globalMinDouble(
                  report.constant_pressure_constraints_preserve_constants
                      ? 1.0
                      : 0.0,
                  comm) ==
                  globalMaxDouble(
                      report.constant_pressure_constraints_preserve_constants
                          ? 1.0
                          : 0.0,
                      comm);
          if (!diagnostic_sampled_identical ||
              !pressure_available_identical ||
              !pressure_field_identical ||
              !kkt_available_identical ||
              !pressure_convergence_identical ||
              !constant_pressure_trace_flags_identical) {
            evaluation.success = false;
            evaluation.diagnostic =
                "pressure_certificate_availability_differs_across_communicator";
            return evaluation;
          }
          if (local_pressure_available) {
            const bool pressure_scalars_identical =
                globalMinDouble(
                    report.pressure_representability_residual_norm,
                    comm) ==
                    globalMaxDouble(
                        report.pressure_representability_residual_norm,
                        comm) &&
                globalMinDouble(
                    report.pressure_representability_relative_distance,
                    comm) ==
                    globalMaxDouble(
                        report.pressure_representability_relative_distance,
                        comm);
            if (!pressure_scalars_identical) {
              evaluation.success = false;
              evaluation.diagnostic =
                  "pressure_certificate_representability_scalars_differ_across_communicator";
              return evaluation;
            }
          }
          if (local_kkt_available) {
            const bool kkt_scalars_identical =
                globalMinDouble(
                    report
                        .constant_pressure_kkt_residual_norm,
                    comm) ==
                    globalMaxDouble(
                        report
                            .constant_pressure_kkt_residual_norm,
                        comm) &&
                globalMinDouble(
                    report
                        .constant_pressure_kkt_relative_distance,
                    comm) ==
                    globalMaxDouble(
                        report
                            .constant_pressure_kkt_relative_distance,
                        comm) &&
                globalMinDouble(
                    report
                        .constant_pressure_kkt_pressure_jump,
                    comm) ==
                    globalMaxDouble(
                        report
                            .constant_pressure_kkt_pressure_jump,
                        comm) &&
                globalMinDouble(
                    report
                        .constant_pressure_kkt_volume_multiplier,
                    comm) ==
                    globalMaxDouble(
                        report
                            .constant_pressure_kkt_volume_multiplier,
                        comm);
            if (!kkt_scalars_identical) {
              evaluation.success = false;
              evaluation.diagnostic =
                  "pressure_certificate_constant_kkt_scalars_differ_across_communicator";
              return evaluation;
            }
          }
          auto& pressure_certificate =
              evaluation.checks[kStaticCapillaryPressureRepresentability];
          pressure_certificate.available = local_pressure_available;
          pressure_certificate.converged =
              local_pressure_available &&
              report.pressure_representability_converged;
          pressure_certificate.breakdown =
              local_pressure_available &&
              report.pressure_representability_breakdown;
          pressure_certificate.values = {
              static_cast<svmp::FE::Real>(
                  report.pressure_representability_residual_norm),
              static_cast<svmp::FE::Real>(
                  report.pressure_representability_relative_distance)};
          evaluation.checks[kStaticCapillaryPhysicalEquilibrium].values = {
              static_cast<svmp::FE::Real>(report.residual_norm)};
          auto& kkt_certificate =
              evaluation.checks[kStaticCapillaryConstantPressureKkt];
          kkt_certificate.required =
              gravitational_acceleration_is_zero &&
              report
                  .constant_pressure_unit_coefficients_represent_constant &&
              report.constant_pressure_constraints_preserve_constants;
          kkt_certificate.available = local_kkt_available;
          kkt_certificate.values = {
              static_cast<svmp::FE::Real>(
                  report
                      .constant_pressure_kkt_residual_norm),
              static_cast<svmp::FE::Real>(
                  report
                      .constant_pressure_kkt_relative_distance)};
          evaluation.diagnostic =
              local_pressure_available
                  ? "acceptance_certificate_available"
                  : "acceptance_certificate_pressure_unavailable:" +
                        report.pressure_representability_reason;
          certified_constant_pressure_jump =
              report.constant_pressure_kkt_pressure_jump;
          certified_constant_pressure_volume_multiplier =
              report
                  .constant_pressure_kkt_volume_multiplier;
          certified_constant_pressure_multiplier_comparison_available =
              local_kkt_available && kkt_certificate.required;
          certified_pressure_field =
              report.pressure_representability_pressure_field;
          certified_solution =
              std::move(certificate.certified_solution);
        }
        return evaluation;
      };

  std::vector<svmp::FE::Real> accepted_parameters;
  svmp::FE::level_set::LevelSetVolumeConstrainedMinimizerResult result;
  try {
    result =
        svmp::FE::level_set::
            minimizeLevelSetVolumeConstrained(
                minimization_options,
                staticCapillaryEquilibriumProblem(minimization_options),
                parameters,
                active_parameters,
                evaluator,
                accepted_parameters);
  } catch (...) {
    transaction.rollback();
    throw;
  }
  const auto final_certificates =
      staticCapillaryCertificateValues(result.final_checks);
  const bool local_prepublication_result_valid =
      result.success &&
      result.converged &&
      result.accepted_coefficients_assigned &&
      accepted_parameters.size() == field_count &&
      certified_solution.size() == baseline.size() &&
      certified_pressure_field != svmp::FE::INVALID_FIELD_ID &&
      certified_pressure_field != level_set_field &&
      (!final_certificates.constant_pressure_kkt_required ||
       (final_certificates.constant_pressure_kkt_available &&
        std::isfinite(certified_constant_pressure_jump) &&
        std::isfinite(
            certified_constant_pressure_volume_multiplier)));
  if (globalAnyBool(
          !local_prepublication_result_valid, comm)) {
    transaction.rollback();
    const auto line_search_phase_name =
        [](svmp::FE::level_set::LevelSetVolumeConstrainedLineSearchPhase
               phase) {
          using Phase = svmp::FE::level_set::
              LevelSetVolumeConstrainedLineSearchPhase;
          switch (phase) {
          case Phase::Trial:
            return "trial";
          case Phase::DeferredReproduction:
            return "deferred_reproduction";
          }
          return "unknown";
        };
    const auto line_search_disposition_name =
        [](svmp::FE::level_set::
               LevelSetVolumeConstrainedLineSearchDisposition disposition) {
          using Disposition = svmp::FE::level_set::
              LevelSetVolumeConstrainedLineSearchDisposition;
          switch (disposition) {
          case Disposition::UnavailableOrForbidden:
            return "unavailable_or_forbidden";
          case Disposition::TopologyRejected:
            return "topology_rejected";
          case Disposition::ConstraintRejected:
            return "constraint_rejected";
          case Disposition::Deferred:
            return "deferred";
          case Disposition::SameTopologyMeritRejected:
            return "same_topology_merit_rejected";
          case Disposition::ArmijoAccepted:
            return "armijo_accepted";
          case Disposition::DerivativeResolutionAccepted:
            return "derivative_resolution_accepted";
          case Disposition::ReproductionAccepted:
            return "reproduction_accepted";
          case Disposition::ReproductionRejected:
            return "reproduction_rejected";
          }
          return "unknown";
        };
    std::ostringstream message;
    message
        << "[svMultiPhysics::Application] Static capillary initialization failed before publication: "
        << result.diagnostic
        << " active_coefficients=" << active_parameters.size()
        << " iterations=" << result.iterations
        << " functional_evaluations=" << result.functional_evaluations
        << " finite_difference_step_shrinks="
        << result.finite_difference_step_shrinks
        << " finite_difference_fourth_order_components="
        << result.finite_difference_fourth_order_components
        << " minimum_finite_difference_step_used="
        << result.minimum_finite_difference_step_used
        << " maximum_finite_difference_step_used="
        << result.maximum_finite_difference_step_used
        << " maximum_energy_derivative_relative_correction="
        << result.maximum_energy_derivative_relative_correction
        << " maximum_volume_derivative_relative_correction="
        << result.maximum_volume_derivative_relative_correction
        << " analytic_derivative_evaluations="
        << result.analytic_derivative_evaluations
        << " derivative_resolution_step_acceptances="
        << result.derivative_resolution_step_acceptances
        << " line_search_rejections=" << result.line_search_rejections
        << " topology_change_rejections="
        << result.topology_change_rejections
        << " topology_epoch_transitions="
        << result.topology_epoch_transitions
        << " max_topology_epoch_transitions="
        << minimization_options.max_topology_epoch_transitions
        << " constraint_change_rejections="
        << result.constraint_change_rejections
        << " limited_memory_updates="
        << result.limited_memory_updates
        << " limited_memory_resets="
        << result.limited_memory_resets
        << " limited_memory_peak_history="
        << result.limited_memory_peak_history
        << " projected_gradient_fallbacks="
        << result.projected_gradient_fallbacks
        << " final_projected_gradient_norm="
        << std::setprecision(17)
        << result.final_projected_gradient_norm
        << " final_volume_error=" << result.final_volume_error
        << " initial_physical_potential_energy="
        << result.initial_energy
        << " final_physical_potential_energy="
        << result.final_energy
        << " line_search_trace_summary={total_attempt_count="
        << result.line_search_trace_total_attempt_count
        << ",retained_count=" << result.line_search_trace.size()
        << ",omitted_count="
        << result.line_search_trace_omitted_count << "}";
    for (const auto& record : result.line_search_trace) {
      message
          << " line_search_trace_record={accepted_iteration_index="
          << record.accepted_iteration_index
          << ",line_search_trial_index="
          << record.line_search_trial_index
          << ",phase=" << line_search_phase_name(record.phase)
          << ",disposition="
          << line_search_disposition_name(record.disposition)
          << ",evaluation_available="
          << record.evaluation_available
          << ",used_limited_memory_direction="
          << record.used_limited_memory_direction
          << ",used_projected_gradient_fallback_direction="
          << record.used_projected_gradient_fallback_direction
          << ",step_size=" << record.step_size
          << ",current_cut_topology_key="
          << record.current_cut_topology_key
          << ",trial_cut_topology_key="
          << record.trial_cut_topology_key
          << ",current_constraint_semantics_key="
          << record.current_constraint_semantics_key
          << ",trial_constraint_semantics_key="
          << record.trial_constraint_semantics_key
          << ",current_merit=" << record.current_merit
          << ",trial_merit=" << record.trial_merit
          << ",armijo_bound=" << record.armijo_bound
          << ",predicted_merit_decrease="
          << record.predicted_merit_decrease
          << ",trial_volume_error="
          << record.trial_volume_error << "}";
    }
    message
        << " line_search_failure_field={field_id=" << level_set_field
        << ",field_offset=" << field_offset
        << ",coefficient_order=raw_field}";
    appendStaticCapillaryFailureState(
        message, result.line_search_failure_state);
    throw std::runtime_error(message.str());
  }

  (void)collectiveLevelSetMaintenanceAlgebraicRevision(
      certified_solution, comm);

  // Backend gather/scatter operations can enter communicator assembly.
  // Reject deterministic vector-layout drift collectively before the first
  // publication backup or mutation so every rank follows the same path.
  const auto local_target_layout_error =
      staticCapillaryHistoryPublicationTargetLayoutError(
          system,
          level_set_field,
          certified_pressure_field,
          certified_solution,
          history);
  if (globalAnyBool(
          !local_target_layout_error.empty(), comm)) {
    transaction.rollback();
    if (!local_target_layout_error.empty()) {
      throw std::runtime_error(local_target_layout_error);
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary target-history layout validation failed on another FE communicator rank.");
  }

  const auto current_backup =
      captureFeOrderedVectorCollectively(
          history.u(), comm);
  std::vector<std::vector<svmp::FE::Real>> history_backups;
  history_backups.reserve(
      static_cast<std::size_t>(history.historyDepth()));
  for (int slot = 1; slot <= history.historyDepth(); ++slot) {
    history_backups.push_back(
        captureFeOrderedVectorCollectively(
            history.uPrevK(slot), comm));
  }
  std::optional<std::vector<svmp::FE::Real>> u_dot_backup;
  std::optional<std::vector<svmp::FE::Real>> u_ddot_backup;
  if (history.hasUDotState()) {
    u_dot_backup = captureFeOrderedVectorCollectively(
        history.uDot(), comm);
  }
  if (history.hasUDDotState()) {
    u_ddot_backup = captureFeOrderedVectorCollectively(
        history.uDDot(), comm);
  }
  const auto local_staging_layout_error =
      staticCapillaryHistoryPublicationLayoutError(
          system,
          level_set_field,
          certified_pressure_field,
          certified_solution,
          history_backups,
          history);
  if (globalAnyBool(
          !local_staging_layout_error.empty(), comm)) {
    transaction.rollback();
    if (!local_staging_layout_error.empty()) {
      throw std::runtime_error(local_staging_layout_error);
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary history layout validation failed on another FE communicator rank.");
  }
  const auto restore_history = [&] {
    scatterFeOrderedSolution(history.u(), current_backup);
    for (int slot = 1; slot <= history.historyDepth(); ++slot) {
      scatterFeOrderedSolution(
          history.uPrevK(slot),
          history_backups[
              static_cast<std::size_t>(slot - 1)]);
    }
    if (u_dot_backup.has_value()) {
      scatterFeOrderedSolution(
          history.uDot(), *u_dot_backup);
    }
    if (u_ddot_backup.has_value()) {
      scatterFeOrderedSolution(
          history.uDDot(), *u_ddot_backup);
    }
    history.updateGhosts();
  };

  std::exception_ptr local_staging_failure;
  try {
    stageStaticCapillaryHistoryForPublication(
        system,
        level_set_field,
        certified_pressure_field,
        certified_solution,
        history_backups,
        history);
  } catch (...) {
    local_staging_failure = std::current_exception();
  }
  if (globalAnyBool(
          local_staging_failure != nullptr, comm)) {
    std::exception_ptr rollback_failure;
    try {
      restore_history();
    } catch (...) {
      rollback_failure = std::current_exception();
    }
    try {
      transaction.rollback();
    } catch (...) {
      if (!rollback_failure) {
        rollback_failure = std::current_exception();
      }
    }
    if (rollback_failure) {
      std::rethrow_exception(rollback_failure);
    }
    if (local_staging_failure) {
      std::rethrow_exception(local_staging_failure);
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Static capillary history staging failed on another FE communicator rank.");
  }

  try {
    transaction.commit();
  } catch (...) {
    const auto failure = std::current_exception();
    if (!transaction.publicationStarted()) {
      std::exception_ptr rollback_failure;
      try {
        restore_history();
      } catch (...) {
        rollback_failure = std::current_exception();
      }
      try {
        transaction.rollback();
      } catch (...) {
        if (!rollback_failure) {
          rollback_failure = std::current_exception();
        }
      }
      if (rollback_failure) {
        std::rethrow_exception(rollback_failure);
      }
    }
    std::rethrow_exception(failure);
  }

  request.runtime.static_capillary_equilibrium_initialized = true;
  application::core::oopCout()
      << std::setprecision(17)
      << "[svMultiPhysics::Application] Static capillary equilibrium initialized"
      << " diagnostic=static_capillary_equilibrium_initialization"
      << " field='" << request.configuration->transport.level_set.field_name << "'"
      << " marker=" << *marker
      << " active_coefficients=" << active_parameters.size()
      << " target_liquid_volume="
      << capillary_options.target_liquid_volume
      << " initial_surface_wall_energy="
      << staticCapillaryEnergy(result.initial_energy_components,
                               kStaticCapillarySurfaceWallEnergy)
      << " final_surface_wall_energy="
      << staticCapillaryEnergy(result.final_energy_components,
                               kStaticCapillarySurfaceWallEnergy)
      << " initial_gravitational_potential_energy="
      << staticCapillaryEnergy(result.initial_energy_components,
                               kStaticCapillaryGravitationalPotentialEnergy)
      << " final_gravitational_potential_energy="
      << staticCapillaryEnergy(result.final_energy_components,
                               kStaticCapillaryGravitationalPotentialEnergy)
      << " initial_physical_potential_energy="
      << result.initial_energy
      << " final_physical_potential_energy="
      << result.final_energy
      << " final_volume_error=" << result.final_volume_error
      << " final_projected_gradient_norm="
      << result.final_projected_gradient_norm
      << " final_volume_multiplier="
      << result.final_volume_multiplier
      << " constant_pressure_jump="
      << certified_constant_pressure_jump
      << " constant_pressure_volume_multiplier="
      << certified_constant_pressure_volume_multiplier
      << " parameter_to_physical_multiplier_comparison_available="
      << (certified_constant_pressure_multiplier_comparison_available ? 1 : 0)
      << " parameter_to_physical_multiplier_difference="
      << (certified_constant_pressure_multiplier_comparison_available
              ? result.final_volume_multiplier -
                    certified_constant_pressure_volume_multiplier
              : std::numeric_limits<double>::quiet_NaN())
      << " pressure_representability_available="
      << (final_certificates.pressure_representability_available ? 1 : 0)
      << " pressure_representability_converged="
      << (final_certificates.pressure_representability_converged ? 1 : 0)
      << " pressure_representability_breakdown="
      << (final_certificates.pressure_representability_breakdown ? 1 : 0)
      << " pressure_representability_residual_norm="
      << final_certificates.pressure_representability_residual_norm
      << " pressure_representability_relative_distance="
      << final_certificates.pressure_representability_relative_distance
      << " production_residual_norm="
      << final_certificates.production_residual_norm
      << " constant_pressure_kkt_required="
      << (final_certificates.constant_pressure_kkt_required ? 1 : 0)
      << " constant_pressure_kkt_available="
      << (final_certificates.constant_pressure_kkt_available ? 1 : 0)
      << " constant_pressure_kkt_residual_norm="
      << final_certificates.constant_pressure_kkt_residual_norm
      << " constant_pressure_kkt_relative_distance="
      << final_certificates.constant_pressure_kkt_relative_distance
      << " volume_tolerance="
      << capillary_options.volume_tolerance
      << " projected_gradient_tolerance="
      << capillary_options
             .projected_gradient_tolerance
      << " pressure_representability_max_residual_norm="
      << capillary_options
             .pressure_representability_max_residual_norm
      << " pressure_representability_max_relative_distance="
      << capillary_options
             .pressure_representability_max_relative_distance
      << " physical_equilibrium_max_residual_norm="
      << capillary_options
             .physical_equilibrium_max_residual_norm
      << " constant_pressure_kkt_max_residual_norm="
      << capillary_options
             .constant_pressure_kkt_max_residual_norm
      << " constant_pressure_kkt_max_relative_distance="
      << capillary_options
             .constant_pressure_kkt_max_relative_distance
      << " limited_memory_history_size="
      << capillary_options
             .limited_memory_history_size
      << " limited_memory_curvature_tolerance="
      << capillary_options
             .limited_memory_curvature_tolerance
      << " initial_snapshot_revision_key="
      << result.initial_snapshot_revision_key
      << " final_snapshot_revision_key="
      << result.final_snapshot_revision_key
      << " cut_topology_key="
      << result.cut_topology_key
      << " constraint_semantics_key="
      << result.constraint_semantics_key
      << " iterations=" << result.iterations
      << " functional_evaluations="
      << result.functional_evaluations
      << " acceptance_certificate_evaluations="
      << result.acceptance_certificate_evaluations
      << " finite_difference_step_shrinks="
      << result.finite_difference_step_shrinks
      << " finite_difference_fourth_order_components="
      << result.finite_difference_fourth_order_components
      << " minimum_finite_difference_step_used="
      << result.minimum_finite_difference_step_used
      << " maximum_finite_difference_step_used="
      << result.maximum_finite_difference_step_used
      << " maximum_energy_derivative_relative_correction="
      << result.maximum_energy_derivative_relative_correction
      << " maximum_volume_derivative_relative_correction="
      << result.maximum_volume_derivative_relative_correction
      << " analytic_derivative_evaluations="
      << result.analytic_derivative_evaluations
      << " derivative_resolution_step_acceptances="
      << result.derivative_resolution_step_acceptances
      << " line_search_rejections="
      << result.line_search_rejections
      << " topology_change_rejections="
      << result.topology_change_rejections
      << " topology_epoch_transitions="
      << result.topology_epoch_transitions
      << " max_topology_epoch_transitions="
      << minimization_options.max_topology_epoch_transitions
      << " constraint_change_rejections="
      << result.constraint_change_rejections
      << " limited_memory_updates="
      << result.limited_memory_updates
      << " limited_memory_resets="
      << result.limited_memory_resets
      << " limited_memory_peak_history="
      << result.limited_memory_peak_history
      << " projected_gradient_fallbacks="
      << result.projected_gradient_fallbacks
      << " production_force_projection_applied=0"
      << " qualification=prerequisite_only"
      << std::endl;
  return true;
}

} // namespace

namespace application {
namespace core {

struct VtkTimeSeriesCollection {
  struct Entry {
    double time{};
    std::string file{};
  };

  std::filesystem::path pvd_path{};
  std::vector<Entry> entries{};
};

namespace {

void write_pvd_collection(const VtkTimeSeriesCollection& pvd)
{
  if (pvd.pvd_path.empty() || pvd.entries.empty()) {
    return;
  }

  std::ofstream out(pvd.pvd_path);
  if (!out.is_open()) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to open PVD file '" + pvd.pvd_path.string() + "'.");
  }

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <Collection>\n";

  out << std::setprecision(16) << std::fixed;
  for (const auto& e : pvd.entries) {
    out << "    <DataSet timestep=\"" << e.time << "\" group=\"\" part=\"0\" file=\"" << e.file << "\"/>\n";
  }

  out << "  </Collection>\n";
  out << "</VTKFile>\n";
}

} // namespace

bool ApplicationDriver::shouldUseNewSolver(const std::string& xml_file)
{
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(xml_file.c_str()) != tinyxml2::XML_SUCCESS) {
    return false;
  }

  auto* root = doc.FirstChildElement(Parameters::FSI_FILE.c_str());
  if (!root) {
    return false;
  }

  auto* general = root->FirstChildElement("GeneralSimulationParameters");
  if (!general) {
    return false;
  }

  auto* flag_elem = general->FirstChildElement("Use_new_OOP_solver");
  if (!flag_elem || !flag_elem->GetText()) {
    return false;
  }

  return parse_bool_relaxed(flag_elem->GetText());
}

void ApplicationDriver::run(const std::string& xml_file, RunMode mode)
{
  oopCout() << "[svMultiPhysics::Application] ApplicationDriver::run(xml_file='" << xml_file << "')" << std::endl;
  if (!xml_file.empty()) {
    std::error_code ec;
    const auto abs = std::filesystem::absolute(xml_file, ec);
    if (!ec) {
      oopCout() << "[svMultiPhysics::Application] XML path: " << abs.string() << std::endl;
    }
  }

  Parameters params;
  params.read_xml(xml_file);
  runWithParameters(params, mode);
}

void ApplicationDriver::runWithParameters(const Parameters& params, RunMode mode)
{
  const auto comm = svmp::MeshComm::world();

  // Auto-configure OpenMP threads: hardware_cores / MPI_ranks_per_node.
  // Respects OMP_NUM_THREADS if the user has set it explicitly. With threaded
  // assembly requested (<Assembly_threads> or SVMP_ASSEMBLY_THREADS above one)
  // OpenMP stays at one thread instead: OpenMP threads change the FSILS
  // reduction order and select the coloured cell assembly, so results would
  // depend on the core count (FE/Docs/ThreadedAssembly.md).
  int assembly_threads = 1;
  {
    const auto& deck = params.general_simulation_parameters.assembly_threads;
    if (deck.defined() && deck.value() >= 1) {
      assembly_threads = deck.value();
    }
    if (const char* env = std::getenv("SVMP_ASSEMBLY_THREADS"); env != nullptr && *env != '\0') {
      char* end = nullptr;
      const long value = std::strtol(env, &end, 10);
      if (end != env && *end == '\0' && value >= 1 && value <= 4096) {
        assembly_threads = static_cast<int>(value);
      }
    }
  }
  if (assembly_threads > 1 && std::getenv("OMP_NUM_THREADS") == nullptr) {
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
  } else {
    configureOpenMPThreads(comm);
  }

  {
    int omp_threads = 1;
#ifdef _OPENMP
    omp_threads = omp_get_max_threads();
#endif
    oopCout() << "[svMultiPhysics::Application] Threading: MPI ranks=" << comm.size()
              << " OMP threads/rank=" << omp_threads
              << " (physical cores=" << detectPhysicalCores()
              << " logical cores=" << std::thread::hardware_concurrency() << ")" << std::endl;
  }

  if (comm.is_parallel() && comm.rank() == 0 && !oopTraceEnabled()) {
    oopCout() << "[svMultiPhysics::Application] MPI ranks=" << comm.size()
              << "; suppressing non-root log output (set SVMP_OOP_SOLVER_TRACE=1 for per-rank logs)." << std::endl;
  }

  oopCout() << "[svMultiPhysics::Application] <Use_new_OOP_solver>=true; running new OOP solver path." << std::endl;
  oopCout()
      << "[svMultiPhysics::Application] Supported (initial): equation types heatS/heatF (Poisson), fluid, stokes, level_set, mesh_motion, ustruct; "
         "single <Add_mesh>; steady constant BCs; selected file-driven temporal BCs; transient time loop (Generalized-α)."
      << std::endl;
  oopCout() << "[svMultiPhysics::Application] Not supported yet: Domain_file_path, multiple domains, "
               "general spatial/temporal BC files, user-defined profiles, restart/continuation, FSI/etc. "
               "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver."
            << std::endl;

  const auto mesh_blocks =
      static_cast<int>(std::count_if(params.mesh_parameters.begin(), params.mesh_parameters.end(),
                                     [](const auto* p) { return p != nullptr; }));
  const auto equation_blocks =
      static_cast<int>(std::count_if(params.equation_parameters.begin(), params.equation_parameters.end(),
                                     [](const auto* p) { return p != nullptr; }));
  oopCout() << "[svMultiPhysics::Application] Input summary: meshes=" << mesh_blocks
            << " equations=" << equation_blocks << std::endl;
  warnInputDeprecations(collectInputDeprecations(params));
  // Validate the initial-rate key against its legacy environment alias
  // before the setup work (it is applied by the transient solve).
  (void)generalizedAlphaInitialRateIsConsistent(params.general_simulation_parameters);

  SimulationBuilder builder(params);
  auto sim = builder.build();
  writeEffectiveConfigurationArtifact(sim, params, comm);
  if (mode == RunMode::ResolveOnly) {
    oopCout() << "[svMultiPhysics::Application] --resolve-only: the effective "
                 "configuration is written; stopping before the first solve."
              << std::endl;
    return;
  }

  if (sim.primary_mesh) {
    const auto global_verts = sim.primary_mesh->global_n_vertices();
    const auto global_cells = sim.primary_mesh->global_n_cells();
    const auto global_faces = sim.primary_mesh->global_n_faces();

    oopCout() << "[svMultiPhysics::Application] Loaded mesh '" << sim.primary_mesh_name << "' (global): "
              << global_verts << " vertices, " << global_cells << " cells, " << global_faces << " faces."
              << std::endl;
  } else {
    oopCout() << "[svMultiPhysics::Application] No meshes were loaded from <Add_mesh>." << std::endl;
  }

  oopCout() << "[svMultiPhysics::Application] Components: fe_system=" << (sim.fe_system ? "yes" : "no")
            << " physics_modules=" << static_cast<int>(sim.physics_modules.size())
            << " backend=" << (sim.backend ? svmp::FE::backends::backendKindToString(sim.backend->backendKind()) : "none")
            << " linear_solver=" << (sim.linear_solver ? svmp::FE::backends::backendKindToString(sim.linear_solver->backendKind()) : "none")
            << " time_history=" << (sim.time_history ? "yes" : "no") << std::endl;
  if (sim.fe_system) {
    oopCout() << "[svMultiPhysics::Application] FE system: ndofs=" << sim.fe_system->dofHandler().getNumDofs()
              << " constraints=" << sim.fe_system->constraints().numConstraints() << std::endl;
  }
  if (sim.time_history) {
    oopCout() << "[svMultiPhysics::Application] TimeHistory: time=" << sim.time_history->time()
              << " dt=" << sim.time_history->dt() << " step=" << sim.time_history->stepIndex()
              << " depth=" << sim.time_history->historyDepth() << std::endl;
  }

  const int num_steps = params.general_simulation_parameters.number_of_time_steps.value();
  const double dt = params.general_simulation_parameters.time_step_size.value();
  oopCout() << "[svMultiPhysics::Application] Time stepping: Number_of_time_steps=" << num_steps
            << " Time_step_size=" << dt << std::endl;
  bool requires_time_advancement = true;
  if (sim.fe_system) {
    const int temporal_order = sim.fe_system->temporalOrder();
    const bool has_explicit_time_terms = sim.fe_system->hasExplicitTimeDependency();
    const bool has_time_dependent_constraints = sim.fe_system->hasTimeDependentConstraints();
    requires_time_advancement = sim.fe_system->requiresTimeAdvancement();
    oopCout() << "[svMultiPhysics::Application] FE temporal dependency: max_dt_order=" << temporal_order
              << " explicit_time_terms=" << (has_explicit_time_terms ? "yes" : "no")
              << " time_dependent_constraints=" << (has_time_dependent_constraints ? "yes" : "no")
              << " requires_time_advancement=" << (requires_time_advancement ? "yes" : "no")
              << std::endl;
  }

  VtkTimeSeriesCollection pvd{};
  VtkTimeSeriesCollection* pvd_ptr = nullptr;
  if (params.general_simulation_parameters.combine_time_series.defined() &&
      params.general_simulation_parameters.combine_time_series.value()) {
    const bool vtk_enabled = params.general_simulation_parameters.save_results_to_vtk_format.defined() &&
                             params.general_simulation_parameters.save_results_to_vtk_format.value();
    if (!vtk_enabled) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] <Combine_time_series> is only implemented for VTK output "
          "(.pvd collection file). Other output formats are not implemented.");
    }

    std::filesystem::path out_dir = ".";
    if (params.general_simulation_parameters.save_results_in_folder.defined() &&
        !params.general_simulation_parameters.save_results_in_folder.value().empty()) {
      out_dir = params.general_simulation_parameters.save_results_in_folder.value();
    }

    std::string prefix = "result";
    if (params.general_simulation_parameters.name_prefix_of_saved_vtk_files.defined() &&
        !params.general_simulation_parameters.name_prefix_of_saved_vtk_files.value().empty()) {
      prefix = params.general_simulation_parameters.name_prefix_of_saved_vtk_files.value();
    }

    pvd.pvd_path = out_dir / (prefix + ".pvd");
    pvd_ptr = &pvd;
  }

  const bool run_quasi_static = (num_steps == 0) || (sim.fe_system && !requires_time_advancement);
  if (run_quasi_static) {
    if (num_steps > 0 && !requires_time_advancement) {
      oopCout() << "[svMultiPhysics::Application] No time-dependent FE terms or constraints detected; "
                   "running a single quasi-static solve instead of a transient time loop."
                << std::endl;
    }
    oopCout() << "[svMultiPhysics::Application] Starting steady-state solve." << std::endl;
    runSteadyState(sim, params, pvd_ptr);
  } else {
    oopCout() << "[svMultiPhysics::Application] Starting transient solve." << std::endl;
    runTransient(sim, params, pvd_ptr);
  }
}

void ApplicationDriver::runSteadyState(SimulationComponents& sim, const Parameters& params, VtkTimeSeriesCollection* pvd)
{
  if (!sim.fe_system) {
    throw std::runtime_error("[svMultiPhysics::Application] Steady solve requires an FE system.");
  }
  if (!sim.backend || !sim.linear_solver) {
    throw std::runtime_error("[svMultiPhysics::Application] Steady solve requires a backend + linear solver.");
  }
  if (!sim.time_history) {
    throw std::runtime_error("[svMultiPhysics::Application] Steady solve requires a TimeHistory.");
  }

  if (params.general_simulation_parameters.continue_previous_simulation.defined() &&
      params.general_simulation_parameters.continue_previous_simulation.value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] <Continue_previous_simulation> is not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  svmp::FE::timestepping::NewtonOptions newton_opts{};
  applyMonolithicEquationNewtonControls(params, newton_opts);

  // Use the unified "equations" operator tag (same as transient).
  newton_opts.residual_op = "equations";
  newton_opts.jacobian_op = "equations";
  const auto steady_active_cut_requests = activeCutVolumeRequests(params);
  // Generated active-domain cuts can make full Newton trial states invalid
  // before line search has a chance to reject them.
  newton_opts.use_line_search =
      parseBoolEnv("SVMP_NEWTON_LINE_SEARCH",
                   !steady_active_cut_requests.empty());
  applyNewtonLineSearchXmlOptions(params.general_simulation_parameters,
                                  newton_opts);
  applyNewtonLineSearchEnvOptions(newton_opts);
  applyNewtonToleranceXmlOptions(params.general_simulation_parameters,
                                 newton_opts);
  applyNewtonIterationXmlOptions(params.general_simulation_parameters,
                                 newton_opts);
  applyNewtonToleranceEnvOptions(newton_opts);
  newton_opts.accept_inexact_linear_solutions =
      parseBoolEnv("SVMP_NEWTON_ACCEPT_INEXACT_LINEAR", false);
  applyNewtonPseudoTransientEnvOptions(newton_opts);

  // Modified Newton: reuse Jacobian across multiple iterations.
  // Period 1 = full Newton (default), 2 = rebuild every 2nd iteration, etc.
  if (const char* jrp = std::getenv("SVMP_JACOBIAN_REBUILD_PERIOD")) {
    newton_opts.jacobian_rebuild_period = std::atoi(jrp);
  }

  // The cut geometry of the solve; the aliases keep its members' names.
  auto cut_geometry =
      std::make_shared<svmp::FE::level_set::LevelSetGeometryService>(
          levelSetGeometryHost(sim), steady_active_cut_requests);
  auto cut_lifecycle =
      std::shared_ptr<svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle>(
          cut_geometry, &cut_geometry->lifecycle());
  auto cut_refresh_cache = std::shared_ptr<ActiveCutContextRefreshCache>(
      cut_geometry, &cut_geometry->refreshCache());
  const auto maintenance_active_requests = activeCutVolumeRequests(params);
  const auto maintenance_configurations = resolveLevelSetMaintenanceConfigurations(
      sim.resolved_level_set_equations_by_input_index,
      maintenance_active_requests);
  auto level_set_maintenance = makeLevelSetMaintenanceRequests(
      maintenance_configurations);
  requireCollectiveLevelSetMaintenanceRequestSchedule(
      level_set_maintenance,
      LevelSetMaintenanceScheduleStage::SteadyInitialization,
      sim.time_history->stepIndex(),
      activeFESystemCommunicator(*sim.fe_system));
  // The semi-implicit capillary term is defined by the transient outer
  // fixed point; a steady solve fails closed here.
  (void)bindFreeSurfaceSemiImplicitReference(
      *sim.fe_system,
      {},
      /*transient_solve=*/false,
      /*outer_fixed_point=*/false,
      activeFESystemCommunicator(*sim.fe_system));
  requireCollectiveLevelSetMaintenanceRequestSchedule(
      level_set_maintenance,
      LevelSetMaintenanceScheduleStage::SteadyInitialization,
      sim.time_history->stepIndex(),
      activeFESystemCommunicator(*sim.fe_system));
  const auto steady_level_set_advection_velocity =
      levelSetAdvectionVelocityRequests(params);
  applyCoupledLevelSetFieldResidualCriteria(
      *sim.fe_system, params, newton_opts);
  const bool steady_has_generated_state =
      !steady_active_cut_requests.empty() || !level_set_maintenance.empty() ||
      !steady_level_set_advection_velocity.empty();
  const bool use_steady_external_state_fixed_point =
      steady_has_generated_state &&
      parseBoolEnv("SVMP_GENERATED_STATE_OUTER_FIXED_POINT", true);
  newton_opts.external_state_fixed_point.enabled =
      use_steady_external_state_fixed_point;
  newton_opts.external_state_fixed_point.max_iterations = std::max(
      1,
      parseIntEnv("SVMP_GENERATED_STATE_OUTER_MAX_ITERATIONS", 12));
  applyGeneratedStateOuterRelaxationEnvOptions(newton_opts);
  // Legacy refreshed/quasi-Newton mode remains available for controlled
  // comparisons.  Production generated geometry uses an outer Picard loop so
  // each inner Newton residual and Jacobian share exactly one frozen G.
  newton_opts.accepted_state_sync_invalidates_residual =
      steady_has_generated_state &&
      !use_steady_external_state_fixed_point;
  applyJacobianCheckGeometryProvenance(
      newton_opts,
      steady_active_cut_requests,
      /*refresh_generated_geometry_within_solve=*/
          !use_steady_external_state_fixed_point,
      /*has_frozen_algebraic_level_set_extension=*/false,
      /*use_external_state_fixed_point=*/
          use_steady_external_state_fixed_point);
  using StateSyncPoint =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  // The generated state of the solve, synchronized by the registry: the cut
  // geometry, then the advection velocity extended from it.
  using svmp::FE::timestepping::FunctionGeneratedStateParticipant;
  using svmp::FE::timestepping::GeneratedStateRole;
  using svmp::FE::timestepping::GeneratedStateSync;
  const auto generated_state =
      std::make_shared<svmp::FE::timestepping::GeneratedStateRegistry>(
          svmp::FE::timestepping::GeneratedStatePolicy{
              .solve = svmp::FE::timestepping::GeneratedStateSolve::Steady,
              .outer_fixed_point = use_steady_external_state_fixed_point});
  generated_state->add(
      std::make_shared<svmp::FE::level_set::LevelSetGeometryParticipant>(
          cut_geometry,
          svmp::FE::level_set::LevelSetGeometryParticipant::Options{
              .solve = svmp::FE::timestepping::GeneratedStateSolve::Steady}));
  addLevelSetVelocityExtensionParticipant(
      *generated_state,
      sim,
      steady_level_set_advection_velocity,
      svmp::FE::timestepping::GeneratedStateSolve::Steady,
      use_steady_external_state_fixed_point);
  generated_state->bind(newton_opts);

  const auto integrator = std::make_shared<const ZeroTimeDerivativeIntegrator>();
  svmp::FE::systems::TransientSystem transient(*sim.fe_system, integrator);

  svmp::FE::timestepping::NewtonSolver newton(newton_opts);
  svmp::FE::timestepping::NewtonWorkspace workspace;

  oopCout() << "[svMultiPhysics::Application] Steady: allocating Newton workspace." << std::endl;
  newton.allocateWorkspace(transient.system(), *sim.backend, workspace);
  oopCout() << "[svMultiPhysics::Application] Steady: Newton workspace allocated." << std::endl;

  // Ensure time-history vectors use the same backend layout as the Newton workspace.
  oopCout() << "[svMultiPhysics::Application] Steady: repacking TimeHistory for backend layout." << std::endl;
  sim.time_history->repack(*sim.backend);
  oopCout() << "[svMultiPhysics::Application] Steady: TimeHistory repacked." << std::endl;

  (void)cut_geometry->refreshCached(sim.time_history->u(), "steady_initial");
  if (initializeDiscreteStaticCapillaryEquilibrium(
          sim,
          params,
          level_set_maintenance,
          *cut_lifecycle,
          *cut_refresh_cache,
          steady_level_set_advection_velocity)) {
    (void)cut_geometry->refreshCached(
        sim.time_history->u(), "steady_static_capillary_accepted");
  }
  initializeConservativePhaseStates(sim, level_set_maintenance);
  (void)updateLevelSetAdvectionVelocities(
      sim, *sim.time_history, steady_level_set_advection_velocity);
  logLevelSetMaintenanceCoverageDiagnostics(
      steady_active_cut_requests,
      level_set_maintenance);

  const double solve_time = sim.time_history->time();
  oopCout() << "[svMultiPhysics::Application] Steady solve: time=" << solve_time
            << " newton(max_it=" << newton_opts.max_iterations << ", min_it=" << newton_opts.min_iterations
            << ", abs_tol=" << newton_opts.abs_tolerance
            << ", rel_tol=" << newton_opts.rel_tolerance << ")" << std::endl;

  const auto report = newton.solveStep(transient, *sim.linear_solver, solve_time, *sim.time_history, workspace);
  oopCout() << "[svMultiPhysics::Application] Steady Newton: converged=" << report.converged
            << " iterations=" << report.iterations << " residual_norm=" << report.residual_norm
            << " outer_iterations=" << report.outer_iterations
            << " inner_iterations_total=" << report.inner_iterations_total
            << " outer_state_change_norm=" << report.outer_state_change_norm
            << " field_residual_norm=" << report.field_residual_norm
            << " auxiliary_residual_norm=" << report.auxiliary_residual_norm
            << " outer_dynamic_relaxation_enabled="
            << report.outer_dynamic_relaxation_enabled
            << " outer_dynamic_relaxation_updates="
            << report.outer_dynamic_relaxation_updates
            << " outer_dynamic_relaxation_safeguards="
            << report.outer_dynamic_relaxation_safeguards
            << " outer_dynamic_relaxation_resets="
            << report.outer_dynamic_relaxation_resets
            << " outer_dynamic_relaxation_factor="
            << report.outer_dynamic_relaxation_factor
            << " outer_relaxed_state_change_norm="
            << report.outer_relaxed_state_change_norm
            << " outer_raw_contraction_ratio="
            << report.outer_raw_contraction_ratio << std::endl;

  if (!report.converged) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Steady solve did not converge (Newton reached max iterations). "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  sim.fe_system->commitTimeStep();
  (void)cut_geometry->refreshCached(sim.time_history->u(), "steady_accepted");
  (void)updateLevelSetAdvectionVelocities(
      sim, *sim.time_history, steady_level_set_advection_velocity);
  const auto steady_free_surface_coverage =
      requireCommunicatorConsistentAcceptedFunctionalCoverage(
          svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations(),
          activeFESystemCommunicator(*sim.fe_system));
  if (steady_free_surface_coverage.has_functional) {
    const auto steady_comm =
        activeFESystemCommunicator(*sim.fe_system);
    const auto steady_solution = gatherFeOrderedSolution(
        sim.time_history->u(), steady_comm);
    const auto steady_state_revision =
        collectiveLevelSetMaintenanceAlgebraicRevision(
            steady_solution, steady_comm);
    const auto steady_contact_stages =
        evaluateAcceptedFreeSurfaceContactStages(
            *sim.fe_system,
            svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations(),
            static_cast<svmp::FE::Real>(solve_time),
            svmp::FE::Real{1.0},
            steady_state_revision,
            steady_state_revision,
            std::span<const svmp::FE::Real>(
                steady_solution.data(), steady_solution.size()));
    (void)recordAcceptedFunctionals(
        sim,
        /*accepted_step=*/1u,
        solve_time,
        sim.time_history->dt(),
        steady_state_revision,
        steady_state_revision,
        steady_contact_stages,
        std::span<const svmp::FE::Real>(
            steady_solution.data(), steady_solution.size()),
        std::span<const svmp::FE::Real>{},
        currentLevelSetVelocityExtensionRevision(
            *sim.fe_system, steady_level_set_advection_velocity));
  }
  recordAcceptedMeshBoundaryProvenanceAtEndpoint(
      sim,
      /*accepted_step=*/1u,
      solve_time,
      sim.time_history->dt());
  outputResults(sim, params, /*step=*/1, solve_time, pvd);

  const auto comm = svmp::MeshComm::world();
  if (pvd && comm.rank() == 0) {
    write_pvd_collection(*pvd);
    if (!pvd->entries.empty()) {
      oopCout() << "[svMultiPhysics::Application] Wrote PVD: " << pvd->pvd_path.string() << std::endl;
    }
  }
}

void ApplicationDriver::runTransient(SimulationComponents& sim, const Parameters& params, VtkTimeSeriesCollection* pvd)
{
  if (!sim.fe_system) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve requires an FE system.");
  }
  if (!sim.backend || !sim.linear_solver) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve requires a backend + linear solver.");
  }
  if (!sim.time_history) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve requires a TimeHistory.");
  }

  if (params.general_simulation_parameters.continue_previous_simulation.defined() &&
      params.general_simulation_parameters.continue_previous_simulation.value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] <Continue_previous_simulation> is not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const int num_steps = params.general_simulation_parameters.number_of_time_steps.value();
  if (num_steps < 1) {
    throw std::runtime_error("[svMultiPhysics::Application] runTransient() requires Number_of_time_steps >= 1.");
  }

  const double dt = sim.time_history->dt();
  if (!(dt > 0.0)) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve requires Time_step_size > 0.");
  }

  const auto transient_scheme =
      resolveTransientTimeIntegrationSelection(
          params.general_simulation_parameters);
  svmp::FE::timestepping::TimeLoopOptions opts{};
  opts.t0 = params.general_simulation_parameters.start_time.defined()
                ? params.general_simulation_parameters.start_time.value()
                : 0.0;
  opts.dt = dt;
  opts.t_end = opts.t0 + static_cast<double>(num_steps) * dt;
  opts.max_steps = num_steps;
  opts.scheme = transient_scheme.scheme;
  if (transient_scheme.generalized_alpha_rho_inf.has_value()) {
    opts.generalized_alpha_rho_inf =
        *transient_scheme.generalized_alpha_rho_inf;
  }

  applyMonolithicEquationNewtonControls(params, opts.newton);
  // NOTE: Newton update scaling for dt(·) fields is available in the FE library
  // (NewtonOptions::scale_dt_increments), but enabling it globally can severely
  // slow convergence for linear problems (e.g., Stokes). Keep it off by default.
  opts.newton.scale_dt_increments = false;

  // Use the unified "equations" operator tag installed by the NS module (and
  // Poisson, etc.). Setting both tags equal enables same_op=true in NewtonSolver,
  // which uses the combined assembleJacobianAndResidual() path and reduces the
  // number of mesh traversals per Newton iteration.
  opts.newton.residual_op = "equations";
  opts.newton.jacobian_op = "equations";

  const auto transient_cut_topology_preflight_comm =
      activeFESystemCommunicator(*sim.fe_system);
  std::vector<ActiveCutVolumeRequest> transient_active_cut_requests;
  std::exception_ptr local_cut_request_preparation_failure;
  try {
    transient_active_cut_requests = activeCutVolumeRequests(params);
  } catch (...) {
    local_cut_request_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_cut_request_preparation_failure,
      transient_cut_topology_preflight_comm,
      "transient_cut_topology_request_preflight");
  const bool local_transient_cut_topology_fingerprint_supported =
      transientCutTopologyFingerprintSupportsRequests(
          transient_active_cut_requests);
  if (globalAnyBool(
          !local_transient_cut_topology_fingerprint_supported,
          transient_cut_topology_preflight_comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Transient fixed cut-topology "
        "acceptance requires geometry_mode=LinearCorner for every active-cut "
        "request. The current partition-independent structural fingerprint "
        "is a probabilistic certificate qualified for LinearCorner cuts, not "
        "a sound arbitrary high-order connectivity certificate; refusing to "
        "start the transient solve rather than disabling the gate.");
  }
  // A first-order generalized-alpha state is (u_n, uDot_n).  Starting an
  // active-cut free-surface problem with uDot_n=0 is generally inconsistent
  // with gravity, pressure, and capillary forces and excites the algorithmic
  // startup mode (the capillary-wave benchmark lost roughly half its apparent
  // stiffness this way).  TimeLoop regularizes only exact zero-mass rows
  // outside a retained cut domain, so the PDE-consistent solve is also the
  // correctness default for active-cut systems.  <Generalized_alpha_initial_rate>
  // Zero is the explicit opt-out (the capillary-rise protocol uses it).
  opts.initialize_first_order_rate_from_pde =
      opts.scheme == svmp::FE::timestepping::SchemeKind::GeneralizedAlpha &&
      generalizedAlphaPdeRateInitializationRequested(
          params.general_simulation_parameters,
          !transient_active_cut_requests.empty());
  const auto level_set_advection_velocity =
      levelSetAdvectionVelocityRequests(params);
  // Generated level-set active domains need line search to reject trial states
  // that temporarily erase or severely distort the retained wet side.
  opts.newton.use_line_search =
      parseBoolEnv("SVMP_NEWTON_LINE_SEARCH",
                   !transient_active_cut_requests.empty() ||
                       !level_set_advection_velocity.empty());
  applyNewtonLineSearchXmlOptions(params.general_simulation_parameters,
                                  opts.newton);
  applyNewtonLineSearchEnvOptions(opts.newton);
  applyNewtonToleranceXmlOptions(params.general_simulation_parameters,
                                 opts.newton);
  applyNewtonIterationXmlOptions(params.general_simulation_parameters,
                                 opts.newton);
  applyNewtonToleranceEnvOptions(opts.newton);
  opts.newton.accept_inexact_linear_solutions =
      parseBoolEnv("SVMP_NEWTON_ACCEPT_INEXACT_LINEAR", false);

  // Modified Newton: reuse Jacobian across multiple iterations.
  if (const char* jrp = std::getenv("SVMP_JACOBIAN_REBUILD_PERIOD")) {
    opts.newton.jacobian_rebuild_period = std::atoi(jrp);
  }

  // Pseudo-transient continuation (PTC): if the linear solve stalls on distorted meshes,
  // add a lumped dt-only diagonal to regularize early Newton iterations and relax it
  // as the nonlinear residual decreases.
  opts.newton.pseudo_transient.enabled = true;
  opts.newton.pseudo_transient.activate_on_linear_failure = true;
  applyNewtonPseudoTransientEnvOptions(opts.newton);

  const auto& general_params = params.general_simulation_parameters;
  const bool adaptive_time_loop_from_xml =
      general_params.enable_adaptive_time_loop.defined() &&
      general_params.enable_adaptive_time_loop.value();
  if (parseBoolEnv("SVMP_TIMELOOP_ADAPTIVE", adaptive_time_loop_from_xml)) {
    svmp::FE::timestepping::SimpleStepControllerOptions controller_opts{};
    const double default_min_dt =
        general_params.adaptive_time_loop_min_dt.defined()
            ? general_params.adaptive_time_loop_min_dt.value()
            : dt * 0.0625;
    const double default_max_dt =
        general_params.adaptive_time_loop_max_dt.defined()
            ? general_params.adaptive_time_loop_max_dt.value()
            : dt;
    const int default_max_retries =
        general_params.adaptive_time_loop_max_retries.defined()
            ? general_params.adaptive_time_loop_max_retries.value()
            : 8;
    const double default_decrease_factor =
        general_params.adaptive_time_loop_decrease_factor.defined()
            ? general_params.adaptive_time_loop_decrease_factor.value()
            : 0.5;
    const double default_increase_factor =
        general_params.adaptive_time_loop_increase_factor.defined()
            ? general_params.adaptive_time_loop_increase_factor.value()
            : 1.0;
    const int default_target_newton_iterations =
        general_params.adaptive_time_loop_target_newton_iterations.defined()
            ? general_params.adaptive_time_loop_target_newton_iterations.value()
            : 6;
    const int default_max_steps_multiplier =
        general_params.adaptive_time_loop_max_steps_multiplier.defined()
            ? general_params.adaptive_time_loop_max_steps_multiplier.value()
            : 8;
    controller_opts.min_dt = parseDoubleEnv("SVMP_TIMELOOP_MIN_DT", default_min_dt);
    controller_opts.max_dt = parseDoubleEnv("SVMP_TIMELOOP_MAX_DT", default_max_dt);
    controller_opts.max_retries = std::max(0, parseIntEnv("SVMP_TIMELOOP_MAX_RETRIES", default_max_retries));
    controller_opts.decrease_factor = parseDoubleEnv("SVMP_TIMELOOP_DECREASE_FACTOR", default_decrease_factor);
    controller_opts.increase_factor = parseDoubleEnv("SVMP_TIMELOOP_INCREASE_FACTOR", default_increase_factor);
    controller_opts.target_newton_iterations =
        std::max(1, parseIntEnv("SVMP_TIMELOOP_TARGET_NEWTON_ITERATIONS", default_target_newton_iterations));
    opts.step_controller =
        std::make_shared<svmp::FE::timestepping::SimpleStepController>(controller_opts);
    const int max_steps_multiplier =
        std::max(1, parseIntEnv("SVMP_TIMELOOP_MAX_STEPS_MULTIPLIER", default_max_steps_multiplier));
    opts.max_steps = std::max(opts.max_steps, num_steps * max_steps_multiplier);
    oopCout() << "[svMultiPhysics::Application] TimeLoop adaptive controller enabled:"
              << " min_dt=" << controller_opts.min_dt
              << " max_dt=" << controller_opts.max_dt
              << " max_retries=" << controller_opts.max_retries
              << " decrease_factor=" << controller_opts.decrease_factor
              << " increase_factor=" << controller_opts.increase_factor
              << " target_newton_iterations=" << controller_opts.target_newton_iterations
              << " max_steps=" << opts.max_steps << std::endl;
  }
  opts.last_step_absorb_fraction =
      parseDoubleEnv("SVMP_TIMELOOP_LAST_STEP_ABSORB_FRACTION",
                     opts.step_controller ? 1.0e-2 : 0.0);

  oopCout() << "[svMultiPhysics::Application] Transient solve: t0="
            << opts.t0 << " dt=" << opts.dt
            << " t_end=" << opts.t_end << " max_steps=" << opts.max_steps
            << " scheme=" << transient_scheme.canonical_name;
  if (transient_scheme.generalized_alpha_rho_inf.has_value()) {
    oopCout() << " rho_inf="
              << *transient_scheme.generalized_alpha_rho_inf;
  } else {
    oopCout() << " rho_inf=n/a";
  }
  oopCout() << " pde_udot_init="
            << (opts.initialize_first_order_rate_from_pde ? 1 : 0)
            << " last_step_absorb_fraction="
            << opts.last_step_absorb_fraction
            << " newton(max_it=" << opts.newton.max_iterations
            << ", min_it=" << opts.newton.min_iterations
            << ", abs_tol=" << opts.newton.abs_tolerance
            << ", rel_tol=" << opts.newton.rel_tolerance << ")"
            << std::endl;

  // Ensure time-history vectors use the same backend layout as the solver workspace.
  oopCout() << "[svMultiPhysics::Application] Transient: repacking TimeHistory for backend layout." << std::endl;
  traceStateVectorFields(*sim.fe_system, sim.time_history->u(), "before_transient_repack");
  sim.time_history->repack(*sim.backend);
  traceStateVectorFields(*sim.fe_system, sim.time_history->u(), "after_transient_repack");
  oopCout() << "[svMultiPhysics::Application] Transient: TimeHistory repacked." << std::endl;

  auto bdf1 = std::make_shared<const svmp::FE::systems::BDFIntegrator>(1);
  svmp::FE::systems::TransientSystem transient(*sim.fe_system, std::move(bdf1));
  const auto maintenance_active_requests = activeCutVolumeRequests(params);
  const auto maintenance_configurations = resolveLevelSetMaintenanceConfigurations(
      sim.resolved_level_set_equations_by_input_index,
      maintenance_active_requests);
  auto level_set_maintenance = makeLevelSetMaintenanceRequests(
      maintenance_configurations);
  requireCollectiveLevelSetMaintenanceRequestSchedule(
      level_set_maintenance,
      LevelSetMaintenanceScheduleStage::TransientInitialization,
      sim.time_history->stepIndex(),
      activeFESystemCommunicator(*sim.fe_system));
  applyCoupledLevelSetFieldResidualCriteria(
      *sim.fe_system, params, opts.newton);
  const bool has_frozen_algebraic_level_set_extension =
      std::any_of(
          level_set_advection_velocity.begin(),
          level_set_advection_velocity.end(),
          [&](const LevelSetAdvectionVelocityRequest& request) {
            const auto target = sim.fe_system->findFieldByName(
                request.target_velocity_field_name);
            return target != svmp::FE::INVALID_FIELD_ID &&
                   sim.fe_system->fieldRecord(target).source_kind ==
                       svmp::FE::systems::FieldSourceKind::Unknown &&
                   static_cast<bool>(svmp::FE::level_set::
                       findLevelSetVelocityExtensionConstraintKernel(
                           *sim.fe_system, request.operator_tag, target));
          });
  // Generated cut rules, active-set constraints, and the wet-extension map
  // form one residual-defining package G(u).  The assembled
  // tangent differentiates R(u,G) at fixed G; it does not contain every shape,
  // projection, or active-set derivative.  Solve the consistent nested
  // problem with outer regeneration and inner frozen-geometry Newton solves.
  const bool has_transient_generated_state =
      !transient_active_cut_requests.empty() ||
      !level_set_maintenance.empty() ||
      !level_set_advection_velocity.empty();
  const bool per_step_only_generated_state =
      parseBoolEnv("SVMP_CUT_REFRESH_PER_STEP_ONLY", false);
  const bool use_transient_external_state_fixed_point =
      has_transient_generated_state &&
      !per_step_only_generated_state &&
      parseBoolEnv("SVMP_GENERATED_STATE_OUTER_FIXED_POINT", true);
  opts.newton.external_state_fixed_point.enabled =
      use_transient_external_state_fixed_point;
  // Outer fixed-point budget and relaxation. Default: at most
  // kOuterFixedPointDefaultMaxPasses refreshes; the first
  // kOuterFixedPointPlainPasses are plain and the updates from then on use
  // the safeguarded delta-squared (Aitken) relaxation. A step that converges
  // within the plain passes is bitwise identical to the plain loop with that
  // cap; only steps that would have hit it continue, relaxed. XML keys
  // Outer_fixed_point_max_passes and Outer_fixed_point_relaxation_start_pass
  // (1-based pass whose update is the first relaxed one; 0 disables, 1
  // relaxes every update) take precedence over the environment variables
  // SVMP_GENERATED_STATE_OUTER_MAX_ITERATIONS,
  // SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION and
  // SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION_START_PASS.
  constexpr int kOuterFixedPointPlainPasses = 12;
  constexpr int kOuterFixedPointDefaultMaxPasses = 30;
  opts.newton.external_state_fixed_point.max_iterations = std::max(
      1,
      parseIntEnv("SVMP_GENERATED_STATE_OUTER_MAX_ITERATIONS",
                  kOuterFixedPointDefaultMaxPasses));
  opts.newton.external_state_fixed_point.dynamic_relaxation.enabled = true;
  opts.newton.external_state_fixed_point.dynamic_relaxation.start_iteration =
      kOuterFixedPointPlainPasses - 1;
  {
    const auto& general = params.general_simulation_parameters;
    if (general.outer_fixed_point_max_passes.defined()) {
      opts.newton.external_state_fixed_point.max_iterations =
          std::max(1, general.outer_fixed_point_max_passes.value());
    }
  }
  // A cut-topology change during a step is a normal event for a moving
  // interface: each outer fixed-point refresh may regenerate a new topology,
  // and the solve continues on it.  The number of such epoch changes is
  // therefore bounded by the outer iteration limit, which is the default.
  // Max_cut_topology_restarts_per_step overrides it (0 restores the
  // stop-and-reject behavior).
  rejectRemovedCutTopologyRestartEnvironment();
  const char* cut_topology_restart_limit_source = "outer_iteration_limit";
  int cut_topology_restart_limit =
      opts.newton.external_state_fixed_point.max_iterations;
  {
    const auto& limit_parameter =
        params.general_simulation_parameters
            .max_cut_topology_restarts_per_step;
    if (limit_parameter.defined() && limit_parameter.value() >= 0) {
      cut_topology_restart_limit = limit_parameter.value();
      cut_topology_restart_limit_source = "xml";
    }
  }
  opts.newton.external_state_fixed_point.max_discontinuity_restarts =
      cut_topology_restart_limit;
  applyGeneratedStateOuterRelaxationEnvOptions(opts.newton);
  {
    const auto& start_parameter = params.general_simulation_parameters
                                      .outer_fixed_point_relaxation_start_pass;
    if (start_parameter.defined()) {
      auto& relaxation =
          opts.newton.external_state_fixed_point.dynamic_relaxation;
      relaxation.enabled = start_parameter.value() > 0;
      relaxation.start_iteration = std::max(0, start_parameter.value() - 1);
    }
  }
  // Exit from a persistent two-state constraint cycle (see NewtonOptions::
  // ExternalStateFixedPointOptions::constraint_cycle_exit_start_iteration).
  // Default: from the first refresh beyond the plain passes, so a step that
  // converges within them is unaffected. XML key
  // Outer_fixed_point_constraint_cycle_exit_pass (1-based refresh; 0
  // disables) takes precedence over the environment variable
  // SVMP_GENERATED_STATE_OUTER_CONSTRAINT_CYCLE_EXIT_PASS.
  {
    int exit_pass = kOuterFixedPointPlainPasses + 1;
    if (std::getenv("SVMP_GENERATED_STATE_OUTER_CONSTRAINT_CYCLE_EXIT_PASS") !=
        nullptr) {
      exit_pass = parseIntEnv(
          "SVMP_GENERATED_STATE_OUTER_CONSTRAINT_CYCLE_EXIT_PASS", exit_pass);
    }
    const auto& exit_parameter = params.general_simulation_parameters
                                     .outer_fixed_point_constraint_cycle_exit_pass;
    if (exit_parameter.defined()) {
      exit_pass = exit_parameter.value();
    }
    opts.newton.external_state_fixed_point
        .constraint_cycle_exit_start_iteration =
        exit_pass > 0 ? exit_pass - 1 : -1;
  }
  // Opt-in stage predictor of first-order generalized-alpha (see
  // TimeLoopOptions::generalized_alpha_predictor); default ConstantRate. The
  // predictor changes the initial iterate only, so results change within
  // the nonlinear tolerances. XML keys Generalized_alpha_predictor
  // (ConstantRate | RateExtrapolation) and Generalized_alpha_predictor_fields
  // (field names; empty = all fields). When the predictor key is absent, the
  // environment variables SVMP_GENERALIZED_ALPHA_PREDICTOR and
  // SVMP_GENERALIZED_ALPHA_PREDICTOR_FIELDS are honored (comparison runs).
  {
    const auto& general = params.general_simulation_parameters;
    std::string predictor_name = general.generalized_alpha_predictor.value();
    std::string field_list = general.generalized_alpha_predictor_fields.value();
    const char* predictor_source = "xml";
    if (!general.generalized_alpha_predictor.defined()) {
      predictor_source = "default";
      if (const char* predictor =
              std::getenv("SVMP_GENERALIZED_ALPHA_PREDICTOR");
          predictor != nullptr && predictor[0] != '\0') {
        predictor_name = predictor;
        predictor_source = "environment";
        if (const char* fields =
                std::getenv("SVMP_GENERALIZED_ALPHA_PREDICTOR_FIELDS");
            fields != nullptr) {
          field_list = fields;
        }
      }
    }
    if (predictor_name == "RateExtrapolation") {
      opts.generalized_alpha_predictor = svmp::FE::timestepping::
          TimeLoopOptions::GeneralizedAlphaPredictor::LinearRateExtrapolation;
      std::replace(field_list.begin(), field_list.end(), ',', ' ');
      std::stringstream field_stream(field_list);
      std::string field_name;
      while (field_stream >> field_name) {
        const auto field = sim.fe_system->findFieldByName(field_name);
        if (field == svmp::FE::INVALID_FIELD_ID) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Generalized_alpha_predictor_fields "
              "names an unknown field '" + field_name + "'.");
        }
        opts.generalized_alpha_predictor_fields.push_back(field);
      }
      oopCout() << "[svMultiPhysics::Application] Generalized-alpha stage "
                   "predictor: RateExtrapolation fields="
                << (opts.generalized_alpha_predictor_fields.empty()
                        ? std::string("all")
                        : field_list)
                << " (" << predictor_source << ")" << std::endl;
    } else if (predictor_name != "ConstantRate") {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Generalized_alpha_predictor must be "
          "ConstantRate or RateExtrapolation, got '" + predictor_name + "'.");
    }
  }
  const bool refresh_generated_geometry_within_solve =
      has_transient_generated_state &&
      !use_transient_external_state_fixed_point &&
      !has_frozen_algebraic_level_set_extension;
  if (use_transient_external_state_fixed_point) {
    oopCout()
        << "[svMultiPhysics::Application] Level-set nonlinear geometry policy: "
           "outer_fixed_point (regenerate cuts, curvature, constraints, and "
           "wet-extension map; solve each inner Newton problem with frozen "
           "generated state), max_outer_iterations="
        << opts.newton.external_state_fixed_point.max_iterations
        << ", max_discontinuity_restarts="
        << opts.newton.external_state_fixed_point
               .max_discontinuity_restarts
        << " (" << cut_topology_restart_limit_source << ")"
        << ", dynamic_relaxation="
        << (opts.newton.external_state_fixed_point.dynamic_relaxation.enabled
                ? "enabled"
                : "disabled")
        << ", dynamic_relaxation_start_pass="
        << (opts.newton.external_state_fixed_point.dynamic_relaxation.enabled
                ? opts.newton.external_state_fixed_point.dynamic_relaxation
                          .start_iteration +
                      1
                : 0)
        << ", constraint_cycle_exit_pass="
        << opts.newton.external_state_fixed_point
                   .constraint_cycle_exit_start_iteration +
               1
        << std::endl;
  } else if (has_frozen_algebraic_level_set_extension) {
    oopCout()
        << "[svMultiPhysics::Application] Level-set nonlinear geometry policy: "
           "legacy fixed_topology comparison mode."
        << std::endl;
  }
  std::vector<FreeSurfaceSemiImplicitTransport> semi_implicit_transports;
  for (const auto& equation : sim.resolved_level_set_equations_by_input_index) {
    if (!equation) {
      continue;
    }
    FreeSurfaceSemiImplicitTransport transport;
    transport.level_set_field_name = equation->options.level_set.field_name;
    transport.velocity_source = equation->options.velocity.source;
    transport.advection_velocity_field_name =
        equation->options.velocity.field_name;
    transport.extension_source_velocity_field_name =
        equation->options.velocity.algebraic_extension_source_field_name;
    for (const auto& request : level_set_advection_velocity) {
      if (request.level_set_field_name == transport.level_set_field_name &&
          request.target_velocity_field_name ==
              transport.advection_velocity_field_name) {
        transport.extension_method = request.extension_method;
        transport.extension_source_velocity_field_name =
            request.source_velocity_field_name;
      }
    }
    semi_implicit_transports.push_back(std::move(transport));
  }
  const auto semi_implicit_reference = bindFreeSurfaceSemiImplicitReference(
      *sim.fe_system,
      semi_implicit_transports,
      /*transient_solve=*/true,
      use_transient_external_state_fixed_point,
      activeFESystemCommunicator(*sim.fe_system));
  if (semi_implicit_reference) {
    refreshFreeSurfaceSemiImplicitReference(
        *sim.fe_system,
        *semi_implicit_reference,
        stateViewForHistory(*sim.time_history));
    svmp::FE::logRecord(
        svmp::FE::LogRecord("ApplicationDriver", "semi-implicit capillary term")
            .diagnostic("free_surface_semi_implicit_reference")
            .text("surface_tension_semi_implicit", "NormalIncrement")
            .text("reference_field",
                  sim.fe_system
                      ->fieldRecord(semi_implicit_reference->reference_field)
                      .name)
            .text("velocity_field", semi_implicit_reference->velocity_field_name)
            .text("reference_refresh",
                  "projected_outer|endpoint|restored|before_solve")
            .rootOnly());
  }
  // When generated data is refreshed at accepted iterates, reassemble the
  // residual after that refresh before deciding that Newton has converged.
  opts.newton.accepted_state_sync_invalidates_residual =
      refresh_generated_geometry_within_solve &&
      has_transient_generated_state;
  const bool synchronize_line_search_trials =
      synchronizeTransientLineSearchTrials(
          opts.newton.accepted_state_sync_invalidates_residual);
  // The cut geometry of the run; the aliases keep its members' names.
  auto cut_geometry =
      std::make_shared<svmp::FE::level_set::LevelSetGeometryService>(
          levelSetGeometryHost(sim), transient_active_cut_requests);
  auto cut_lifecycle =
      std::shared_ptr<svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle>(
          cut_geometry, &cut_geometry->lifecycle());
  auto cut_refresh_cache = std::shared_ptr<ActiveCutContextRefreshCache>(
      cut_geometry, &cut_geometry->refreshCache());
  // Seed wet-volume drift diagnostics from the actual initialized level-set
  // state.  Waiting until the first accepted step silently made that state
  // the reference and hid all first-step volume loss.
  std::map<std::string, svmp::FE::Real> initial_wet_volume_by_key;
  (void)cut_geometry->refreshCached(sim.time_history->u(), "initial");
  if (initializeDiscreteStaticCapillaryEquilibrium(
          sim,
          params,
          level_set_maintenance,
          *cut_lifecycle,
          *cut_refresh_cache,
          level_set_advection_velocity)) {
    (void)cut_geometry->refreshCached(
        sim.time_history->u(), "static_capillary_accepted");
  }
  initializeConservativePhaseStates(sim, level_set_maintenance);
  if (parseBoolEnv("SVMP_CUT_RULE_DUMP", false)) {
    dumpActiveCutVolumeRulesForProbe(*sim.fe_system,
                                     sim.time_history->stepIndex());
  }
  logLevelSetMaintenanceCoverageDiagnostics(
      activeCutVolumeRequests(params),
      level_set_maintenance);
  logWetVolumeDiagnostics(
      activeCutVolumeRequests(params),
      sim.fe_system->cutIntegrationContext(),
      sim.fe_system->meshAccess(),
      sim.primary_mesh ? sim.primary_mesh->n_cells() : 0u,
      activeFESystemCommunicator(*sim.fe_system),
      sim.time_history->stepIndex(),
      sim.time_history->time(),
      initial_wet_volume_by_key);

  svmp::FE::timestepping::TimeLoopCallbacks callbacks{};
  std::vector<svmp::FE::Real> accepted_pressure_update_previous_solution;
  application::core::LevelSetMaintenanceWorkLedger
      level_set_maintenance_work;
  std::uint64_t level_set_maintenance_transaction_sequence{0u};
  std::uint64_t step_acceptance_attempt{0u};
  std::optional<int> step_acceptance_attempt_step{};
  const auto velocity_extension_artifact_comm =
      activeFESystemCommunicator(*sim.fe_system);
  // What a maintenance work row records of a candidate state.
  const svmp::FE::level_set::LevelSetMaintenanceWorkRowSource
      maintenance_work_rows{
          .functional_values =
              [&](std::span<const svmp::FE::Real> candidate) {
                return levelSetMaintenanceFunctionalValues(
                    sim,
                    currentAcceptedFunctionalLayouts(sim),
                    candidate);
              },
          .algebraic_state_revision =
              [&](std::span<const svmp::FE::Real> candidate) {
                return collectiveLevelSetMaintenanceAlgebraicRevision(
                    candidate, velocity_extension_artifact_comm);
              },
          .extension_map_revision =
              [&] {
                return currentLevelSetVelocityExtensionRevision(
                    *sim.fe_system, level_set_advection_velocity);
              },
      };
  const bool local_has_conservative_phase = std::any_of(
      level_set_maintenance.begin(),
      level_set_maintenance.end(),
      [](const auto& request) {
        return request.configuration->transport.conservative_phase.enabled;
      });
  const auto [minimum_has_conservative_phase,
              maximum_has_conservative_phase] =
      globalMinMaxUint64(
          local_has_conservative_phase ? 1u : 0u,
          velocity_extension_artifact_comm);
  if (minimum_has_conservative_phase !=
      maximum_has_conservative_phase) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase enablement differs across the active FE communicator.");
  }
  const bool has_conservative_phase =
      maximum_has_conservative_phase == 1u;
  const bool local_pointwise_wall_contract_requested =
      hasExplicitUnsupportedConservativePhasePointwiseWallContract(
          level_set_maintenance);
  const auto [minimum_pointwise_wall_contract_requested,
              maximum_pointwise_wall_contract_requested] =
      globalMinMaxUint64(
          local_pointwise_wall_contract_requested ? 1u : 0u,
          velocity_extension_artifact_comm);
  if (minimum_pointwise_wall_contract_requested !=
      maximum_pointwise_wall_contract_requested) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase pointwise wall-velocity contract configuration differs across the active FE communicator.");
  }
  if (maximum_pointwise_wall_contract_requested != 0u) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative_phase_impermeable_normal_velocity_tolerance requests a pointwise velocity-normal wall contract that the current conservative-phase split does not implement; the configured boundary policy enforces only discrete q-flux and can be blind where q=0.");
  }
  const bool local_conservative_phase_scheme_supported =
      !local_has_conservative_phase ||
      (opts.scheme ==
           svmp::FE::timestepping::SchemeKind::BackwardEuler &&
       sim.fe_system->temporalOrder() == 1);
  const auto [minimum_conservative_phase_scheme_supported,
              maximum_conservative_phase_scheme_supported] =
      globalMinMaxUint64(
          local_conservative_phase_scheme_supported ? 1u : 0u,
          velocity_extension_artifact_comm);
  if (minimum_conservative_phase_scheme_supported !=
      maximum_conservative_phase_scheme_supported) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase temporal-split support differs across the active FE communicator.");
  }
  if (has_conservative_phase &&
      maximum_conservative_phase_scheme_supported == 0u) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase split transport currently supports only temporal-order-one Backward Euler; generalized-alpha, DG, BDF, and other time schemes fail closed before TimeLoop.");
  }
  std::vector<ConservativePhaseFixedGraphBinding>
      conservative_phase_fixed_graph_bindings;
  std::optional<ConservativePhaseCandidateStageSnapshot>
      pending_conservative_phase_stage;
  if (has_conservative_phase) {
    conservative_phase_fixed_graph_bindings =
        captureConservativePhaseFixedGraphBindings(
            *sim.fe_system,
            level_set_maintenance,
            velocity_extension_artifact_comm);
    requireConservativePhaseFixedGraphBindingsCurrent(
        *sim.fe_system,
        level_set_maintenance,
        conservative_phase_fixed_graph_bindings,
        velocity_extension_artifact_comm,
        "initial_fixed_background_before_time_loop");
  }
  // The free-surface step participants share an FE step context
  // (FE/TimeStepping/StepContext.h): the run's FE system, time-integration
  // selection, active FE communicator and the driver services they call.
  const svmp::FE::timestepping::StepContext free_surface_step_context{
      .sim = svmp::FE::timestepping::RunComponents(sim),
      .transient = transient,
      .opts = {.scheme = opts.scheme},
      .transient_scheme = {.scheme = transient_scheme.scheme,
                           .canonical_name = transient_scheme.canonical_name},
      .active_comm = velocity_extension_artifact_comm,
      .collectiveMeshBoundaryStateFingerprint =
          [](const svmp::FE::backends::GenericVector& state,
             const svmp::MeshComm& comm) {
            return collectiveMeshBoundaryStateFingerprint(state, comm);
          },
      .current_velocity_extension_revision =
          [&] {
            return currentLevelSetVelocityExtensionRevision(
                *sim.fe_system, level_set_advection_velocity);
          },
  };
  // Fitted-ALE operator-stage measurements: FE mesh-normal stage participant
  // (FE/TimeStepping/MeshNormalStageParticipant.h) on the run's FE system,
  // time-integration selection and active FE communicator.
  const auto fitted_ale_stage_participant =
      std::make_shared<svmp::FE::timestepping::MeshNormalStageParticipant>(
          svmp::FE::timestepping::MeshNormalStageContext{
              .system = sim.fe_system.get(),
              .scheme = opts.scheme,
              .scheme_name = transient_scheme.canonical_name,
              .comm = feCollectiveCommunicator(velocity_extension_artifact_comm),
              .state_fingerprint =
                  [comm = velocity_extension_artifact_comm](
                      const svmp::FE::backends::GenericVector& state) {
                    return collectiveMeshBoundaryStateFingerprint(state, comm);
                  },
          });
  auto& fitted_ale_stage = *fitted_ale_stage_participant;
  const bool& evaluate_fitted_ale_operator_stage =
      fitted_ale_stage.evaluate_fitted_ale_operator_stage;
  const auto& fitted_ale_operator_stage_state_size =
      fitted_ale_stage.fitted_ale_operator_stage_state_size;
  // Two-fluid accepted-stage diagnostics: FE step participant
  // (FE/LevelSet/TwoFluidStageParticipant.h).
  const auto two_fluid_stage_participant =
      std::make_shared<svmp::FE::level_set::TwoFluidStageParticipant>(
          free_surface_step_context);
  auto& two_fluid_stage = *two_fluid_stage_participant;
  const bool& evaluate_two_fluid_stage =
      two_fluid_stage.evaluate_two_fluid_stage;
  const bool& reconcile_two_fluid_momentum =
      two_fluid_stage.reconcile_two_fluid_momentum;
  // Free-surface energy account: FE step participant
  // (FE/LevelSet/LevelSetEnergyParticipant.h).
  const auto energy_participant =
      std::make_shared<svmp::FE::level_set::EnergyParticipant>(
          free_surface_step_context);
  auto& energy = *energy_participant;
  const auto& free_surface_declarations = energy.free_surface_declarations;
  const bool& has_dynamic_contact_stage = energy.has_dynamic_contact_stage;
  const bool& evaluate_free_surface_residual_work =
      energy.evaluate_free_surface_residual_work;
  const std::uint64_t& free_surface_energy_attempt =
      energy.free_surface_energy_attempt;
  // Accepted contact stages: FE step participant
  // (FE/LevelSet/LevelSetContactStageParticipant.h).
  const auto contact_stage_participant =
      std::make_shared<svmp::FE::level_set::ContactStageParticipant>(
          free_surface_step_context,
          has_dynamic_contact_stage,
          free_surface_declarations,
          fitted_ale_operator_stage_state_size);
  auto& contact_stage = *contact_stage_participant;
  const bool& capture_dynamic_contact_generalized_alpha =
      contact_stage.capture_dynamic_contact_generalized_alpha;
  auto& pending_dynamic_contact_generalized_alpha_observation =
      contact_stage.pending_dynamic_contact_generalized_alpha_observation;
  const int& dynamic_contact_solve_attempt_index =
      contact_stage.dynamic_contact_solve_attempt_index;
  auto& pending_contact_stages = contact_stage.pending_contact_stages;
  auto& pending_contact_stage_constraints =
      contact_stage.pending_contact_stage_constraints;
  auto& pending_contact_stage_solution =
      contact_stage.pending_contact_stage_solution;
  auto& pending_contact_stage_endpoint_algebraic_revision =
      contact_stage.pending_contact_stage_endpoint_algebraic_revision;
  const auto resolve_contact_stage_temporal_coordinates_collectively =
      [&](const svmp::FE::timestepping::TimeHistory& history,
          int expected_attempt_index,
          std::string_view phase) {
        return contact_stage.resolveTemporalCoordinatesCollectively(
            history, expected_attempt_index, phase);
      };
  const bool track_transient_cut_topology =
      !transient_active_cut_requests.empty();
  auto cut_topology_attempt_tracker =
      std::shared_ptr<TransientCutTopologyAttemptTracker>(
          cut_geometry, &cut_geometry->topologyTracker());
  if (track_transient_cut_topology) {
    cut_geometry->trackTopology();
  }
  // The generated state of the transient solve, synchronized by the
  // registry; its participants are registered and bound to the Newton
  // options with the synchronization hook below.
  const auto generated_state =
      std::make_shared<svmp::FE::timestepping::GeneratedStateRegistry>(
          svmp::FE::timestepping::GeneratedStatePolicy{
              .solve = svmp::FE::timestepping::GeneratedStateSolve::Transient,
              .outer_fixed_point = use_transient_external_state_fixed_point,
              .refresh_within_solve = refresh_generated_geometry_within_solve,
              .synchronize_line_search_trials = synchronize_line_search_trials,
              .per_step_only = per_step_only_generated_state});
  const auto invalidate_transient_cut_topology_dependent_state = [&] {
    generated_state->invalidate();
  };
  // Accepted states that the kinematic reconciliation and the sign-definite
  // patch bounds of the next step compare against: captured before the first
  // step and after every accepted step, once its maintenance has committed or
  // rolled back.
  std::vector<LevelSetKinematicBaseline> kinematic_baselines;
  const auto capture_kinematic_baselines =
      [&](svmp::FE::timestepping::TimeHistory& history) {
        kinematic_baselines.clear();
        // Conservative-phase requests skip accepted-step maintenance, so the
        // patch bounds need no baseline for them.
        const auto needs_baseline = [](const auto& request) {
          const auto& transport = request.configuration->transport;
          return transport.kinematic_reconciliation.enabled ||
                 (transport.sign_definite_patch_bounds.enabled &&
                  !transport.conservative_phase.enabled);
        };
        const bool any_enabled = std::any_of(
            level_set_maintenance.begin(),
            level_set_maintenance.end(),
            needs_baseline);
        if (!any_enabled) {
          return;
        }
        const auto solution = gatherFeOrderedSolution(
            history.u(), activeFESystemCommunicator(*sim.fe_system));
        for (const auto& request : level_set_maintenance) {
          if (needs_baseline(request)) {
            kinematic_baselines.push_back(makeLevelSetKinematicBaseline(
                *sim.fe_system, request, solution, history.stepIndex()));
          }
        }
      };
  callbacks.on_step_start = [&](const svmp::FE::timestepping::TimeHistory& h) {
    energy.requireNoResidualWorkStageAtStepStart();
    contact_stage.beginAttempt(h);
    if (has_conservative_phase) {
      const auto [minimum_pending_stage, maximum_pending_stage] =
          globalMinMaxUint64(
              pending_conservative_phase_stage.has_value() ? 1u : 0u,
              velocity_extension_artifact_comm);
      if (minimum_pending_stage != maximum_pending_stage ||
          maximum_pending_stage != 0u) {
        throw std::logic_error(
            "[svMultiPhysics::Application] A conservative phase operator-stage snapshot survived into a new solve/retry attempt.");
      }
      requireCollectiveLevelSetMaintenanceRequestSchedule(
          level_set_maintenance,
          LevelSetMaintenanceScheduleStage::
              ProspectiveAcceptedEndpoint,
          h.stepIndex() + 1,
          velocity_extension_artifact_comm);
      requireConservativePhaseFixedGraphBindingsCurrent(
          *sim.fe_system,
          level_set_maintenance,
          conservative_phase_fixed_graph_bindings,
          velocity_extension_artifact_comm,
          "step_start_fixed_background_before_physical_solve");
    }
    if (!step_acceptance_attempt_step.has_value() ||
        *step_acceptance_attempt_step != h.stepIndex()) {
      step_acceptance_attempt_step = h.stepIndex();
      step_acceptance_attempt = 0u;
    }
    energy.beginAttempt(h);
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_start step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << std::endl;
    logCutGeometryMemory("step_start");
  };
  applyJacobianCheckGeometryProvenance(
      opts.newton,
      transient_active_cut_requests,
      refresh_generated_geometry_within_solve,
      has_frozen_algebraic_level_set_extension,
      use_transient_external_state_fixed_point);
  const auto expected_cut_request_policy_key =
      activeCutVolumeRequestPolicyKey(transient_active_cut_requests);
  using svmp::FE::timestepping::FunctionGeneratedStateParticipant;
  using svmp::FE::timestepping::GeneratedStateRole;
  using svmp::FE::timestepping::GeneratedStateSync;
  // The cut geometry first (it provides the cut-topology epochs of the outer
  // fixed point), then the advection velocity extended from it and the
  // semi-implicit reference state.
  generated_state->add(
      std::make_shared<svmp::FE::level_set::LevelSetGeometryParticipant>(
          cut_geometry,
          svmp::FE::level_set::LevelSetGeometryParticipant::Options{
              .solve = svmp::FE::timestepping::GeneratedStateSolve::Transient,
              .track_topology = track_transient_cut_topology,
              .reinstall_after_contact_stage_endpoint =
                  has_dynamic_contact_stage &&
                  !parseBoolEnv("SVMP_DISABLE_CUT_CONTEXT_REINSTALL", false),
              .expected_request_policy_key = expected_cut_request_policy_key}));
  addLevelSetVelocityExtensionParticipant(
      *generated_state,
      sim,
      level_set_advection_velocity,
      svmp::FE::timestepping::GeneratedStateSolve::Transient,
      use_transient_external_state_fixed_point);
  if (semi_implicit_reference) {
    generated_state->add(std::make_shared<FunctionGeneratedStateParticipant>(
        "free_surface_semi_implicit_reference",
        GeneratedStateRole::Reference,
        FunctionGeneratedStateParticipant::Callbacks{
            .synchronize =
                [&](const svmp::FE::systems::SystemStateView& state,
                    GeneratedStateSync&) {
                  // u_ref := the projected iterate that generated this
                  // geometry, so the semi-implicit capillary term vanishes
                  // in the fresh residual.
                  refreshFreeSurfaceSemiImplicitReference(
                      *sim.fe_system, *semi_implicit_reference, state);
                }}));
  }
  generated_state->bind(opts.newton);
  // The cut geometry's part of each time step (the cut-topology attempt and
  // the restore retention), composed with the callbacks below.
  svmp::FE::timestepping::StepParticipantList cut_geometry_step_participants;
  svmp::FE::level_set::LevelSetGeometryStepParticipant::addTo(
      cut_geometry_step_participants,
      cut_geometry,
      {.track_topology = track_transient_cut_topology});
  callbacks.on_before_physics_solve =
      [&](svmp::FE::timestepping::TimeHistory& h, double /*solve_time*/, double /*dt*/) {
        // The cut geometry step participant has armed the restore
        // retention, refreshed the geometry from this attempt-entry state
        // and checked the accepted topology baseline.
        (void)updateLevelSetAdvectionVelocities(
            sim, h, level_set_advection_velocity);
        if (semi_implicit_reference) {
          refreshFreeSurfaceSemiImplicitReference(
              *sim.fe_system, *semi_implicit_reference,
              stateViewForHistory(h));
        }
        energy.recordInitialBaseline(h);
        if (acceptedPressureUpdateDiagnosticEnabled()) {
          accepted_pressure_update_previous_solution =
              gatherFeOrderedSolution(
                  h.u(), velocity_extension_artifact_comm);
        } else {
          accepted_pressure_update_previous_solution.clear();
        }
        return true;
      };
  callbacks.on_nonlinear_done = [&](const svmp::FE::timestepping::TimeHistory& h,
                                   const svmp::FE::timestepping::NewtonReport& nr) {
    oopCout() << "[svMultiPhysics::Application] TimeLoop: nonlinear_done step=" << h.stepIndex()
              << " time=" << h.time() << " converged=" << nr.converged
              << " iters=" << nr.iterations << " ||r||=" << nr.residual_norm
              << " outer_iters=" << nr.outer_iterations
              << " inner_iters_total=" << nr.inner_iterations_total
              << " outer_state_change_norm="
              << nr.outer_state_change_norm
              << " ||r_field||=" << nr.field_residual_norm
              << " ||r_aux||=" << nr.auxiliary_residual_norm
              << " (linear: converged=" << nr.linear.converged
              << " iters=" << nr.linear.iterations
              << " rel=" << nr.linear.relative_residual << ")"
              << " outer_dynamic_relaxation_enabled="
              << nr.outer_dynamic_relaxation_enabled
              << " outer_dynamic_relaxation_updates="
              << nr.outer_dynamic_relaxation_updates
              << " outer_dynamic_relaxation_safeguards="
              << nr.outer_dynamic_relaxation_safeguards
              << " outer_dynamic_relaxation_resets="
              << nr.outer_dynamic_relaxation_resets
              << " outer_dynamic_relaxation_factor="
              << nr.outer_dynamic_relaxation_factor
              << " outer_relaxed_state_change_norm="
              << nr.outer_relaxed_state_change_norm
              << " outer_raw_contraction_ratio="
              << nr.outer_raw_contraction_ratio
              << " external_state_discontinuity="
              << nr.external_state_discontinuity
              << " external_state_discontinuity_restarts="
              << nr.external_state_discontinuity_restarts
              << " failure_message=" << std::quoted(nr.failure_message)
              << std::endl;
  };
  if (has_conservative_phase) {
    callbacks.on_candidate_stage =
        [&](const svmp::FE::timestepping::CandidateStageObservation&
                observation) {
          const auto [minimum_pending_stage, maximum_pending_stage] =
              globalMinMaxUint64(
                  pending_conservative_phase_stage.has_value() ? 1u : 0u,
                  velocity_extension_artifact_comm);
          if (minimum_pending_stage != maximum_pending_stage ||
              maximum_pending_stage != 0u) {
            throw std::logic_error(
                "[svMultiPhysics::Application] Conservative phase candidate-stage observation is not one-shot on every active FE rank.");
          }
          pending_conservative_phase_stage.emplace(
              buildConservativePhaseCandidateStageSnapshot(
                  sim,
                  *sim.time_history,
                  level_set_maintenance,
                  conservative_phase_fixed_graph_bindings,
                  observation,
                  velocity_extension_artifact_comm));
        };
  }
  const bool has_precommit_maintenance = has_conservative_phase;
  if (has_dynamic_contact_stage &&
      opts.scheme != svmp::FE::timestepping::SchemeKind::BackwardEuler &&
      opts.scheme != svmp::FE::timestepping::SchemeKind::BDF2 &&
      opts.scheme != svmp::FE::timestepping::SchemeKind::VSVO_BDF &&
      opts.scheme != svmp::FE::timestepping::SchemeKind::GeneralizedAlpha) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic contact-stage provenance currently supports backward Euler, BDF2, VSVO BDF, and generalized-alpha time integration only.");
  }
  ConservativePhaseCandidateResult pending_phase_candidate;
  ConservativePhaseCandidateResult accepted_phase_candidate;
  LevelSetMaintenancePublicationState
      pending_phase_publication_state{
          LevelSetMaintenancePublicationState::Staged};
  bool pending_phase_geometry_commit_required{false};
  bool pending_phase_geometry_commit_returned{false};
  bool pending_phase_ledger_transaction_inactive{false};
  svmp::FE::level_set::LevelSetMaintenanceWorkMark pending_phase_work_mark{};
  const auto clear_pending_contact_provenance = [&] {
    contact_stage.clearPendingProvenance();
    energy.clearPendingAccount();
  };
  const auto reject_pending_phase_work = [&] {
    if (!level_set_maintenance_work.transactionActive()) {
      return;
    }
    level_set_maintenance_work.rejectTransaction();
    logLevelSetMaintenanceWorkPublishedSince(
        level_set_maintenance_work,
        pending_phase_work_mark,
        application::core::LevelSetMaintenanceWorkStatus::Rejected);
  };
  const auto commit_pending_phase_work = [&] {
    if (!level_set_maintenance_work.transactionActive()) {
      return;
    }
    level_set_maintenance_work.commitTransaction();
    logLevelSetMaintenanceWorkPublishedSince(
        level_set_maintenance_work,
        pending_phase_work_mark,
        application::core::LevelSetMaintenanceWorkStatus::Accepted);
  };
  const auto rollback_and_reject_pending_phase =
      [&](svmp::FE::timestepping::TimeHistory& h,
          std::string_view reason) {
        const bool publication_began =
            pending_phase_publication_state !=
            LevelSetMaintenancePublicationState::Staged;
        if (publication_began) {
          svmp::FE::logRecord(
              svmp::FE::LogRecord("LevelSetMaintenance", "maintenance transaction")
                  .diagnostic("level_set_maintenance_transaction")
                  .rootOnly()
                  .value("step", h.stepIndex() + 1)
                  .quoted("reason", reason)
                  .text("outcome", "publication_failed")
                  .value("publication_state", levelSetMaintenancePublicationStateName(pending_phase_publication_state))
                  .boolean("rollback_claimed", false)
                  .boolean("candidate_geometry_and_state_restored", false)
                  .boolean("ledger_transaction_inactive", !level_set_maintenance_work.transactionActive()));
          throw std::runtime_error(
              "[svMultiPhysics::Application] Conservative phase "
              "publication had already begun; candidate discard is "
              "fail-stop and cannot roll back geometry, algebraic state, "
              "or the maintenance ledger.");
        }
        std::string recovery_failure;
        const auto record_failure =
            [&](std::string_view resource,
                std::string_view diagnostic) {
              if (!recovery_failure.empty()) {
                recovery_failure += "; ";
              }
              recovery_failure += std::string(resource);
              recovery_failure += ": ";
              recovery_failure += std::string(diagnostic);
            };
        try {
          rollbackConservativePhaseCandidate(
              h, pending_phase_candidate);
        } catch (const std::exception& error) {
          record_failure("candidate_geometry_or_state", error.what());
        } catch (...) {
          record_failure(
              "candidate_geometry_or_state", "unknown failure");
        }
        try {
          reject_pending_phase_work();
        } catch (const std::exception& error) {
          record_failure("maintenance_ledger", error.what());
        } catch (...) {
          record_failure("maintenance_ledger", "unknown failure");
        }
        const bool candidate_restored =
            !pending_phase_candidate.geometry_transaction &&
            !pending_phase_candidate.changed;
        const bool ledger_rejected =
            !level_set_maintenance_work.transactionActive();
        const bool restored =
            recovery_failure.empty() &&
            candidate_restored &&
            ledger_rejected;
        svmp::FE::logRecord(
            svmp::FE::LogRecord("LevelSetMaintenance", "maintenance transaction")
                .diagnostic("level_set_maintenance_transaction")
                .rootOnly()
                .value("step", h.stepIndex() + 1)
                .quoted("reason", reason)
                .text("outcome", restored ? "rolled_back" : "rollback_failed")
                .value("publication_state", levelSetMaintenancePublicationStateName(pending_phase_publication_state))
                .boolean("rollback_claimed", restored)
                .boolean("candidate_geometry_and_state_restored", candidate_restored)
                .boolean("ledger_rejected", ledger_rejected));
        if (!restored) {
          if (recovery_failure.empty()) {
            recovery_failure =
                "covered rollback resources remain unresolved";
          }
          throw std::runtime_error(
              "[svMultiPhysics::Application] Conservative phase "
              "candidate rollback/rejection failed: " +
              recovery_failure);
        }
      };
  callbacks.on_before_step_accept =
      [&](svmp::FE::timestepping::TimeHistory& h,
          const svmp::FE::timestepping::NewtonReport& nr) {
        requireCollectiveLevelSetMaintenanceRequestSchedule(
            level_set_maintenance,
            LevelSetMaintenanceScheduleStage::
                ProspectiveAcceptedEndpoint,
            h.stepIndex() + 1,
            velocity_extension_artifact_comm);
        (void)nr;
        std::optional<ConservativePhaseCandidateStageSnapshot>
            consumed_conservative_phase_stage;
        if (has_conservative_phase) {
          const auto [minimum_pending_stage, maximum_pending_stage] =
              globalMinMaxUint64(
                  pending_conservative_phase_stage.has_value() ? 1u : 0u,
                  velocity_extension_artifact_comm);
          if (minimum_pending_stage != maximum_pending_stage ||
              maximum_pending_stage != 1u) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Conservative phase acceptance requires exactly one immutable endpoint-stage snapshot on every active FE rank.");
          }
          consumed_conservative_phase_stage =
              std::move(pending_conservative_phase_stage);
          pending_conservative_phase_stage.reset();
          if (consumed_conservative_phase_stage->attempt <=
              step_acceptance_attempt) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Conservative phase candidate-stage attempt provenance is stale or non-monotone for the current physical step.");
          }
          step_acceptance_attempt =
              consumed_conservative_phase_stage->attempt;
          if (step_acceptance_attempt !=
              free_surface_energy_attempt) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Conservative phase and free-surface energy attempt provenance disagree for the current physical step.");
          }
        } else {
          step_acceptance_attempt = free_surface_energy_attempt;
        }
        energy.requireStagedResidualWorkForAcceptance(
            h, step_acceptance_attempt);
        clear_pending_contact_provenance();
        if (pending_phase_publication_state !=
            LevelSetMaintenancePublicationState::Staged) {
          throw std::logic_error(
              "[svMultiPhysics::Application] A conservative phase "
              "publication remained non-staged at the start of a new "
              "acceptance attempt.");
        }
        if (pending_phase_candidate.geometry_transaction) {
          throw std::logic_error(
              "[svMultiPhysics::Application] A conservative phase candidate remained active at the start of a new acceptance attempt.");
        }
        if (level_set_maintenance_work.transactionActive()) {
          throw std::logic_error(
              "[svMultiPhysics::Application] A level-set maintenance work transaction remained active at the start of a new acceptance attempt.");
        }
        pending_phase_work_mark = level_set_maintenance_work.mark();
        pending_phase_candidate = ConservativePhaseCandidateResult{};
        try {
          std::vector<
              application::core::LevelSetAuthoritativeFunctionalValue>
              staged_phase_functionals;
          if (has_precommit_maintenance) {
            level_set_maintenance_work.beginTransaction(
                application::core::LevelSetMaintenanceWorkTransaction{
                    .transaction_id =
                        ++level_set_maintenance_transaction_sequence,
                    .step = static_cast<std::uint64_t>(
                        h.stepIndex() + 1),
                    .attempt = step_acceptance_attempt,
                    .time = h.time() + h.dt(),
                    .dt = h.dt(),
                    .declared_stage =
                        application::core::
                            LevelSetMaintenanceDeclaredStage::
                                ProspectiveAcceptedEndpoint,
                    .extension_map_revision =
                        currentLevelSetVelocityExtensionRevision(
                            *sim.fe_system, level_set_advection_velocity),
                });
            const auto staged_phase_solution =
                gatherFeOrderedSolution(
                    h.u(), velocity_extension_artifact_comm);
            staged_phase_functionals =
                maintenance_work_rows.functional_values(
                    staged_phase_solution);
          }
          const auto build_contact_stage_candidate =
              [&](std::span<const svmp::FE::Real> endpoint_solution,
                  LevelSetMaintenanceGeometryTransaction*
                      active_transaction) {
                if (!has_dynamic_contact_stage) {
                  return ConservativePhaseContactStageCandidate{};
                }
                const auto temporal_coordinates =
                    resolve_contact_stage_temporal_coordinates_collectively(
                        h,
                        dynamic_contact_solve_attempt_index,
                        "prospective_contact_stage_temporal_provenance");
                std::vector<svmp::FE::Real> previous_solution;
                std::exception_ptr local_state_preparation_failure;
                try {
                  previous_solution = gatherFeOrderedSolution(
                      h.uPrev(), velocity_extension_artifact_comm);
                  const auto expected_size = static_cast<std::size_t>(
                      sim.fe_system->dofHandler().getNumDofs());
                  if (previous_solution.size() != expected_size ||
                      endpoint_solution.size() != expected_size) {
                    throw std::runtime_error(
                        "Prospective contact-stage endpoint/history state "
                        "does not match the current FE layout.");
                  }
                } catch (...) {
                  local_state_preparation_failure =
                      std::current_exception();
                }
                requireCollectivePhasePreparation(
                    local_state_preparation_failure,
                    velocity_extension_artifact_comm,
                    "prospective_contact_stage_state_preparation");
                const auto previous_state_revision =
                    collectiveLevelSetMaintenanceAlgebraicRevision(
                        previous_solution,
                        velocity_extension_artifact_comm);
                const auto endpoint_state_revision =
                    collectiveLevelSetMaintenanceAlgebraicRevision(
                        endpoint_solution,
                        velocity_extension_artifact_comm);
                const auto* exact_operator_stage_observation =
                    temporal_coordinates.first_order_generalized_alpha
                            .has_value()
                        ? &pending_dynamic_contact_generalized_alpha_observation
                               .value()
                        : nullptr;
                return buildAcceptedFreeSurfaceContactStageCandidate(
                    sim,
                    params,
                    *cut_lifecycle,
                    *cut_refresh_cache,
                    transient_active_cut_requests,
                    temporal_coordinates.stage_time,
                    temporal_coordinates.stage_alpha_f,
                    previous_state_revision,
                    endpoint_state_revision,
                    previous_solution,
                    endpoint_solution,
                    active_transaction,
                    temporal_coordinates.first_order_generalized_alpha,
                    exact_operator_stage_observation);
              };
          ConservativePhaseContactStageBuilder contact_stage_builder;
          if (has_dynamic_contact_stage) {
            contact_stage_builder = build_contact_stage_candidate;
          }
          const LevelSetMaintenanceStageObserver observe_phase_stage =
              [&](application::core::LevelSetMaintenanceWorkSubstage
                      substage,
                  std::span<const svmp::FE::Real> before_candidate,
                  std::span<const svmp::FE::Real> after_candidate) {
                svmp::FE::level_set::stageLevelSetMaintenanceWorkRow(
                    level_set_maintenance_work,
                    maintenance_work_rows,
                    staged_phase_functionals,
                    substage,
                    before_candidate,
                    after_candidate);
              };
          if (has_conservative_phase) {
            pending_phase_candidate = applyConservativePhaseCandidates(
                sim,
                h,
                level_set_maintenance,
                params,
                *cut_lifecycle,
                *cut_refresh_cache,
                transient_active_cut_requests,
                *consumed_conservative_phase_stage,
                conservative_phase_fixed_graph_bindings,
                step_acceptance_attempt,
                contact_stage_builder,
                observe_phase_stage);
          }
          if (!pending_phase_candidate.accept_step) {
            clear_pending_contact_provenance();
            pending_dynamic_contact_generalized_alpha_observation.reset();
            rollback_and_reject_pending_phase(
                h, "candidate_contract_rejection");
            return false;
          }
          if (activePressureUpdateRejectOnTriggerEnabled() &&
              acceptedPressureUpdateDiagnosticEnabled() &&
              !accepted_pressure_update_previous_solution.empty()) {
            const auto current_solution = gatherFeOrderedSolution(
                h.u(), velocity_extension_artifact_comm);
            const bool triggered = logAcceptedPressureUpdateDiagnostic(
                *sim.fe_system,
                params,
                std::span<const svmp::FE::Real>(
                    accepted_pressure_update_previous_solution.data(),
                    accepted_pressure_update_previous_solution.size()),
                std::span<const svmp::FE::Real>(
                    current_solution.data(),
                    current_solution.size()),
                h.stepIndex() + 1,
                h.time() + h.dt(),
                h.dt(),
                "pre_commit",
                /*honor_fail_on_trigger=*/false);
            if (triggered) {
              clear_pending_contact_provenance();
              pending_dynamic_contact_generalized_alpha_observation.reset();
              rollback_and_reject_pending_phase(
                  h, "pressure_update_rejection");
              return false;
            }
          }
          if (reconcile_two_fluid_momentum) {
            const auto corrected_solution = gatherFeOrderedSolution(
                h.u(), velocity_extension_artifact_comm);
            const auto corrected_algebraic_revision =
                collectiveMeshBoundaryStateFingerprint(
                    h.u(), velocity_extension_artifact_comm);
            auto reconciliation =
                buildCurrentTwoFluidMomentumReconciliations(
                    sim,
                    level_set_maintenance,
                    corrected_solution,
                    corrected_algebraic_revision,
                    static_cast<svmp::FE::Real>(h.dt()),
                    velocity_extension_artifact_comm);
            for (const auto& record : reconciliation.records) {
              // Map entry R1.180 (tests/cases/fluid/free_surface_log_record_map.json).
              svmp::FE::logRecord(
                  svmp::FE::LogRecord("TwoFluidAcceptedStage", "momentum reconciliation")
                      .diagnostic("two_fluid_momentum_reconciliation")
                      .rootOnly()
                      .value("marker", record.interface_marker)
                      .value("raw_snapshot", record.raw_geometry_revision.snapshot_revision_key)
                      .value("corrected_snapshot", record.corrected_geometry_revision.snapshot_revision_key)
                      .value("raw_algebraic_revision", record.raw_algebraic_revision)
                      .value("corrected_algebraic_revision", record.corrected_algebraic_revision)
                      .real("tolerance", record.relative_tolerance)
                      .real("negative_delta_norm", record.negative_phase.momentum_delta_norm)
                      .real("negative_allowed_delta", record.negative_phase.allowed_momentum_delta)
                      .real("positive_delta_norm", record.positive_phase.momentum_delta_norm)
                      .real("positive_allowed_delta", record.positive_phase.allowed_momentum_delta)
                      .boolean("velocity_update_applied", false)
                      .text("status", record.satisfied ? "satisfied" : "rejected"));
            }
            if (!reconciliation.satisfied) {
              clear_pending_contact_provenance();
              pending_dynamic_contact_generalized_alpha_observation.reset();
              rollback_and_reject_pending_phase(
                  h, "momentum_reconciliation_rejection");
              return false;
            }
            svmp::FE::systems::twoFluidStageRecords(*sim.fe_system)
                .bindPendingTwoFluidMomentumReconciliations(
                    reconciliation.records);
          }
          return true;
        } catch (...) {
          const auto failure = std::current_exception();
          clear_pending_contact_provenance();
          pending_dynamic_contact_generalized_alpha_observation.reset();
          try {
            rollback_and_reject_pending_phase(
                h, "candidate_exception");
          } catch (const std::exception& recovery_error) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Conservative phase "
                "candidate recovery failed after an exception: " +
                std::string(recovery_error.what()));
          }
          std::rethrow_exception(failure);
        }
      };
  callbacks.on_step_candidate_discarded =
      [&](svmp::FE::timestepping::TimeHistory& h) {
        energy.discardCandidate();
        pending_conservative_phase_stage.reset();
        pending_dynamic_contact_generalized_alpha_observation.reset();
        two_fluid_stage.resetAggregationEvidence();
        fitted_ale_stage.discardCandidate();
        two_fluid_stage.discardCandidate();
        clear_pending_contact_provenance();
        rollback_and_reject_pending_phase(
            h, "time_loop_candidate_discarded");
        pending_phase_candidate = ConservativePhaseCandidateResult{};
      };
  callbacks.on_step_candidate_ready =
      [&](svmp::FE::timestepping::TimeHistory& h,
          const svmp::FE::timestepping::NewtonReport& newton_report)
          -> std::optional<svmp::FE::timestepping::StepRejectReason> {
        try {
          if (has_dynamic_contact_stage) {
            std::vector<svmp::FE::Real> endpoint_solution;
            std::vector<svmp::FE::Real> previous_solution;
            std::exception_ptr local_state_preparation_failure;
            try {
              endpoint_solution = gatherFeOrderedSolution(
                  h.u(), velocity_extension_artifact_comm);
              previous_solution = gatherFeOrderedSolution(
                  h.uPrev(), velocity_extension_artifact_comm);
            } catch (...) {
              local_state_preparation_failure =
                  std::current_exception();
            }
            requireCollectivePhasePreparation(
                local_state_preparation_failure,
                velocity_extension_artifact_comm,
                "final_candidate_contact_state_fingerprint");
            const auto previous_state_revision =
                collectiveLevelSetMaintenanceAlgebraicRevision(
                    previous_solution,
                    velocity_extension_artifact_comm);
            const auto endpoint_state_revision =
                collectiveLevelSetMaintenanceAlgebraicRevision(
                    endpoint_solution,
                    velocity_extension_artifact_comm);
            const auto temporal_coordinates =
                resolve_contact_stage_temporal_coordinates_collectively(
                    h,
                    dynamic_contact_solve_attempt_index,
                    "final_candidate_contact_stage_temporal_provenance");
            const auto* exact_operator_stage_observation =
                temporal_coordinates.first_order_generalized_alpha.has_value()
                    ? &pending_dynamic_contact_generalized_alpha_observation
                           .value()
                    : nullptr;
            auto contact_stage =
                buildAcceptedFreeSurfaceContactStageCandidate(
                    sim,
                    params,
                    *cut_lifecycle,
                    *cut_refresh_cache,
                    transient_active_cut_requests,
                    temporal_coordinates.stage_time,
                    temporal_coordinates.stage_alpha_f,
                    previous_state_revision,
                    endpoint_state_revision,
                    previous_solution,
                    endpoint_solution,
                    driverGeometryTransaction(
                        pending_phase_candidate.geometry_transaction.get()),
                    temporal_coordinates.first_order_generalized_alpha,
                    exact_operator_stage_observation);
            std::exception_ptr local_stage_storage_failure;
            try {
              pending_contact_stages =
                  std::move(contact_stage.stages);
              pending_contact_stage_constraints =
                  std::move(contact_stage.constraints);
              pending_contact_stage_solution =
                  std::move(contact_stage.stage_solution);
              pending_contact_stage_endpoint_algebraic_revision =
                  endpoint_state_revision;
            } catch (...) {
              local_stage_storage_failure =
                  std::current_exception();
            }
            requireCollectivePhasePreparation(
                local_stage_storage_failure,
                velocity_extension_artifact_comm,
                "final_candidate_cut_context_refresh");
            pending_dynamic_contact_generalized_alpha_observation.reset();
          }

          // The reinstall window of the endpoint/contact-stage geometry ends
          // with the contact-stage reconstruction.
          cut_refresh_cache->closeReinstallWindow();
          // Dynamic-contact reconstruction may finish with an endpoint refresh,
          // but every candidate takes this uniform final path.  In particular,
          // per-step-only generated state can skip every within-solve endpoint
          // synchronization, so the authoritative gate must observe h.u()
          // itself immediately before deciding.
          const auto final_cut_report = cut_geometry->refreshCached(
              h.u(), "final_candidate_topology_gate");
          two_fluid_stage.bindCandidateNumerics(newton_report);
          if (!track_transient_cut_topology) {
            return std::nullopt;
          }
          if (final_cut_report.request_policy_key !=
              expected_cut_request_policy_key) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Active cut request policy changed at the final transient candidate gate.");
          }
          if (!cut_topology_attempt_tracker->candidateMustReject(
                  cut_refresh_cache->topology_key)) {
            return std::nullopt;
          }

          oopCout()
              << "[svMultiPhysics::Application] Transient cut-topology candidate"
              << " diagnostic=cut_topology_acceptance_gate"
              << " step=" << h.stepIndex()
              << " action=reject"
              << " accepted_topology_key="
              << cut_topology_attempt_tracker->acceptedTopologyKey()
                     .value_or(0u)
              << " attempt_topology_key="
              << cut_topology_attempt_tracker->attemptTopologyKey()
                     .value_or(0u)
              << " final_topology_key="
              << final_cut_report.topology_key
              << " first_mismatched_topology_key="
              << cut_topology_attempt_tracker
                     ->firstMismatchedTopologyKey()
                     .value_or(0u)
              << " first_mismatch_provenance="
              << (cut_topology_attempt_tracker
                          ->firstMismatchProvenance()
                          .empty()
                      ? "unknown"
                      : cut_topology_attempt_tracker
                            ->firstMismatchProvenance())
              << std::endl;
          invalidate_transient_cut_topology_dependent_state();
          return svmp::FE::timestepping::StepRejectReason::
              CutTopologyChanged;
        } catch (...) {
          clear_pending_contact_provenance();
          pending_dynamic_contact_generalized_alpha_observation.reset();
          two_fluid_stage.resetAggregationEvidence();
          throw;
        }
      };
  callbacks.on_step_commit_ready =
      [&](svmp::FE::timestepping::TimeHistory& h) {
        energy.requireStagedResidualWorkAtCommitReady(
            h, step_acceptance_attempt);
        if (capture_dynamic_contact_generalized_alpha) {
          const auto [minimum_pending_stage, maximum_pending_stage] =
              globalMinMaxUint64(
                  pending_dynamic_contact_generalized_alpha_observation
                          .has_value()
                      ? 1u
                      : 0u,
                  velocity_extension_artifact_comm);
          if (minimum_pending_stage != maximum_pending_stage ||
              maximum_pending_stage != 0u) {
            throw std::logic_error(
                "[svMultiPhysics::Application] Dynamic-contact generalized-"
                "alpha provenance reached commit-ready without one-shot "
                "consumption.");
          }
        }
        if (has_conservative_phase) {
          const auto [minimum_pending_stage, maximum_pending_stage] =
              globalMinMaxUint64(
                  pending_conservative_phase_stage.has_value() ? 1u : 0u,
                  velocity_extension_artifact_comm);
          if (minimum_pending_stage != maximum_pending_stage ||
              maximum_pending_stage != 0u) {
            throw std::logic_error(
                "[svMultiPhysics::Application] A conservative phase operator-stage snapshot reached commit-ready without one-shot consumption.");
          }
          requireConservativePhaseFixedGraphBindingsCurrent(
              *sim.fe_system,
              level_set_maintenance,
              conservative_phase_fixed_graph_bindings,
              velocity_extension_artifact_comm,
              "commit_ready_fixed_background");
        }
        const bool local_cut_topology_commit_invalid =
            track_transient_cut_topology &&
            cut_topology_attempt_tracker->candidateMustReject(
                cut_refresh_cache->topology_key);
        if (globalAnyBool(
                local_cut_topology_commit_invalid,
                velocity_extension_artifact_comm)) {
          throw std::logic_error(
              "[svMultiPhysics::Application] A transient cut-topology "
              "candidate reached commit-ready without passing the "
              "fixed-topology gate on every active FE communicator rank.");
        }
        const bool changed = pending_phase_candidate.changed;
        if (has_precommit_maintenance) {
          std::vector<std::uint64_t> commit_state_words;
          bool local_commit_invariants_satisfied = false;
          std::exception_ptr local_commit_state_preparation_failure;
          try {
            commit_state_words.reserve(
                16u + 3u * static_cast<std::size_t>(
                                std::max(0, h.historyDepth())));
            appendMaintenanceScheduleWord(commit_state_words, 1u);
          appendMaintenanceScheduleBool(
              commit_state_words,
              static_cast<bool>(
                  pending_phase_candidate.geometry_transaction));
          appendMaintenanceScheduleBool(
              commit_state_words,
              pending_phase_candidate.geometry_transaction &&
                  pending_phase_candidate.geometry_transaction
                      ->active());
          appendMaintenanceScheduleBool(
              commit_state_words,
              pending_phase_candidate.accept_step);
          appendMaintenanceScheduleBool(
              commit_state_words,
              pending_phase_candidate.changed);
          const auto endpoint_solution =
              gatherFeOrderedSolution(
                  h.u(), velocity_extension_artifact_comm);
          // Content is bound by the algebraic revision of the gathered
          // vector.  GenericVector::valueRevision() counts rank-local
          // modifications and differs between ranks, so it must not enter
          // the cross-rank commit words.
          appendMaintenanceScheduleWord(
              commit_state_words,
              levelSetMaintenanceAlgebraicRevision(
                  endpoint_solution));
          appendMaintenanceScheduleWord(
              commit_state_words,
              static_cast<std::uint64_t>(
                  endpoint_solution.size()));
          appendMaintenanceScheduleSigned(
              commit_state_words, h.historyDepth());
          for (int level = 1; level <= h.historyDepth(); ++level) {
            const auto history_solution =
                gatherFeOrderedSolution(
                    h.uPrevK(level),
                    velocity_extension_artifact_comm);
            appendMaintenanceScheduleWord(
                commit_state_words,
                levelSetMaintenanceAlgebraicRevision(
                    history_solution));
            appendMaintenanceScheduleWord(
                commit_state_words,
                static_cast<std::uint64_t>(
                    history_solution.size()));
          }
          const auto request_schedule =
              canonicalLevelSetMaintenanceRequestSchedule(
                  level_set_maintenance,
                  LevelSetMaintenanceScheduleStage::
                      ProspectiveAcceptedEndpoint,
                  h.stepIndex() + 1);
          appendCanonicalLevelSetMaintenanceScheduleSection(
              commit_state_words, request_schedule);
          const auto geometry_state =
              canonicalLevelSetMaintenanceGeometryState(
                  sim, level_set_maintenance);
          appendCanonicalLevelSetMaintenanceGeometrySection(
              commit_state_words, geometry_state);
          appendMaintenanceScheduleWord(
              commit_state_words,
              static_cast<std::uint64_t>(
                  pending_phase_candidate.maintenance_ledgers.size()));
          bool local_split_stage_provenance_satisfied =
              !has_conservative_phase ||
              pending_phase_candidate.maintenance_ledgers.size() ==
                  level_set_maintenance.size();
          if (has_conservative_phase &&
              pending_phase_candidate.maintenance_ledgers.size() ==
                  level_set_maintenance.size()) {
            for (std::size_t index = 0u;
                 index < level_set_maintenance.size(); ++index) {
              const auto& request = level_set_maintenance[index];
              const auto& ledger =
                  pending_phase_candidate.maintenance_ledgers[index];
              appendMaintenanceScheduleBool(
                  commit_state_words,
                  request.configuration->transport.conservative_phase.enabled);
              appendConservativePhaseSplitStageProvenanceWords(
                  commit_state_words,
                  ledger.split_stage_provenance);
              if (!request.configuration->transport.conservative_phase.enabled) {
                continue;
              }
              const auto live_flux_ledger_digest =
                  svmp::FE::level_set::levelSetP1PhaseFluxLedgerDigest(
                      ledger.transport_stage);
              const auto live_velocity_revision =
                  svmp::FE::level_set::
                      levelSetP1PhaseVelocityContentRevision(
                          ledger.transport_stage
                              .sampled_nodal_velocity);
              appendMaintenanceScheduleWord(
                  commit_state_words, live_flux_ledger_digest);
              appendMaintenanceScheduleWord(
                  commit_state_words, live_velocity_revision);
              appendMaintenanceScheduleReal(
                  commit_state_words,
                  ledger.transport_stage.time_step);
              appendMaintenanceScheduleReal(
                  commit_state_words,
                  ledger.maximum_nodal_boundary_mass_transfer);
              appendMaintenanceScheduleReal(
                  commit_state_words,
                  ledger.boundary_mass_tolerance);
              appendMaintenanceScheduleEnum(
                  commit_state_words,
                  ledger.boundary_flux_policy);
              const auto* provenance =
                  ledger.split_stage_provenance.has_value()
                      ? &*ledger.split_stage_provenance
                      : nullptr;
              local_split_stage_provenance_satisfied =
                  provenance != nullptr &&
                  ledger.transport_stage.success &&
                  provenance->scheme ==
                      svmp::FE::level_set::
                          LevelSetP1PhaseSplitScheme::
                              BackwardEulerExplicitIndicatorEndpointVelocity &&
                  provenance->transport_mesh_policy ==
                      svmp::FE::level_set::
                          LevelSetP1PhaseTransportMeshPolicy::
                              FixedBackground &&
                  provenance->temporal_order == 1 &&
                  provenance->prospective_step ==
                      static_cast<std::uint64_t>(h.stepIndex() + 1) &&
                  provenance->attempt == step_acceptance_attempt &&
                  provenance->operator_state_revision != 0u &&
                  provenance->previous_q_revision != 0u &&
                  provenance->nodal_velocity_revision ==
                      live_velocity_revision &&
                  provenance->final_flux_ledger_digest ==
                      live_flux_ledger_digest &&
                  sameConservativePhaseStageRealBits(
                      provenance->step_start_time, h.time()) &&
                  sameConservativePhaseStageRealBits(
                      provenance->step_end_time,
                      h.time() + h.dt()) &&
                  sameConservativePhaseStageRealBits(
                      provenance->q_input_time, h.time()) &&
                  sameConservativePhaseStageRealBits(
                      provenance->velocity_state_time,
                      h.time() + h.dt()) &&
                  sameConservativePhaseStageRealBits(
                      provenance->time_step, h.dt()) &&
                  sameConservativePhaseStageRealBits(
                      ledger.transport_stage.time_step, h.dt()) &&
                  std::isfinite(
                      ledger.maximum_nodal_boundary_mass_transfer) &&
                  std::isfinite(ledger.boundary_mass_tolerance) &&
                  ledger.boundary_mass_tolerance >=
                      svmp::FE::Real{0.0} &&
                  ledger.boundary_flux_policy ==
                      request.configuration->transport.conservative_phase
                          .boundary_flux_policy &&
                  svmp::FE::level_set::
                      levelSetConservativePhaseBoundaryFluxSatisfied(
                          ledger.boundary_flux_policy,
                          ledger.maximum_nodal_boundary_mass_transfer,
                          ledger.transport_stage.correction
                              .total_physical_boundary_mass_transfer,
                          ledger.boundary_mass_tolerance) &&
                  sameConservativePhaseStageRealBits(
                      provenance->stage_options.invariant_tolerance,
                      ledger.transport_stage.executed_options
                          .invariant_tolerance) &&
                  sameConservativePhaseStageRealBits(
                      provenance->stage_options
                          .component_activity_tolerance,
                      ledger.transport_stage.executed_options
                          .component_activity_tolerance) &&
                  sameConservativePhaseStageRealBits(
                      provenance->stage_options.maximum_courant,
                      ledger.transport_stage.executed_options
                          .maximum_courant) &&
                  provenance->stage_options.enforce_courant_limit ==
                      ledger.transport_stage.executed_options
                          .enforce_courant_limit &&
                  provenance->stage_options
                          .require_constant_preservation ==
                      ledger.transport_stage.executed_options
                          .require_constant_preservation &&
                  sameConservativePhaseGraphIdentity(
                      provenance->previous_graph_identity,
                      conservative_phase_fixed_graph_bindings[index]
                          .identity) &&
                  sameConservativePhaseGraphIdentity(
                      provenance->operator_graph_identity,
                      conservative_phase_fixed_graph_bindings[index]
                          .identity) &&
                  local_split_stage_provenance_satisfied;
            }
          }
            local_commit_invariants_satisfied =
              pending_phase_candidate.accept_step &&
              request_schedule.supported &&
              geometry_state.supported &&
              local_split_stage_provenance_satisfied &&
              (!pending_phase_candidate.changed ||
               (pending_phase_candidate.geometry_transaction &&
                pending_phase_candidate.geometry_transaction
                    ->active())) &&
              (!pending_phase_candidate.geometry_transaction ||
               pending_phase_candidate.geometry_transaction
                   ->active());
          } catch (...) {
            local_commit_state_preparation_failure =
                std::current_exception();
          }
          requireCollectivePhasePreparation(
              local_commit_state_preparation_failure,
              velocity_extension_artifact_comm,
              "conservative_phase_commit_state_preparation");
          const auto decision =
              application::core::
                  collectiveLevelSetMaintenanceTransactionDecision(
                      level_set_maintenance_work,
                      local_commit_invariants_satisfied,
                      velocity_extension_artifact_comm,
                      commit_state_words);
          if (decision ==
              application::core::
                  LevelSetMaintenanceTransactionDecision::Reject) {
            clear_pending_contact_provenance();
            rollback_and_reject_pending_phase(
                h, "collective_consensus_rejection");
            pending_phase_candidate =
                ConservativePhaseCandidateResult{};
            throw std::runtime_error(
                "[svMultiPhysics::Application] Conservative phase maintenance transaction was rejected by collective commit consensus.");
          }
        }
        if (has_precommit_maintenance) {
          pending_phase_geometry_commit_required =
              static_cast<bool>(
                  pending_phase_candidate.geometry_transaction);
          pending_phase_geometry_commit_returned =
              !pending_phase_geometry_commit_required;
          pending_phase_ledger_transaction_inactive =
              !level_set_maintenance_work.transactionActive();
          pending_phase_publication_state =
              LevelSetMaintenancePublicationState::Publishing;
          try {
            if (pending_phase_candidate.geometry_transaction) {
              pending_phase_candidate.geometry_transaction->commit();
              pending_phase_geometry_commit_returned = true;
            }
            commit_pending_phase_work();
            pending_phase_ledger_transaction_inactive =
                !level_set_maintenance_work.transactionActive();
            svmp::FE::logRecord(
                svmp::FE::LogRecord("LevelSetConservativePhase", "candidate")
                    .diagnostic("conservative_phase_candidate")
                    .rootOnly()
                    .value("step", h.stepIndex() + 1)
                    .text("outcome", "committed")
                    .boolean("geometry_validated_before_commit", true)
                    .boolean("accepted_state_change", changed));
            pending_phase_publication_state =
                LevelSetMaintenancePublicationState::Published;
          } catch (...) {
            pending_phase_publication_state =
                LevelSetMaintenancePublicationState::PublicationFailed;
            pending_phase_ledger_transaction_inactive =
                !level_set_maintenance_work.transactionActive();
            svmp::FE::logRecord(
                svmp::FE::LogRecord("LevelSetMaintenance", "maintenance transaction")
                    .diagnostic("level_set_maintenance_transaction")
                    .rootOnly()
                    .value("step", h.stepIndex() + 1)
                    .text("outcome", "publication_failed")
                    .boolean("geometry_commit_required", pending_phase_geometry_commit_required)
                    .boolean("geometry_commit_returned", pending_phase_geometry_commit_returned)
                    .boolean("ledger_transaction_inactive", pending_phase_ledger_transaction_inactive)
                    .boolean("rollback_claimed", false));
            throw;
          }
        }
        accepted_phase_candidate = std::move(pending_phase_candidate);
        pending_phase_candidate = ConservativePhaseCandidateResult{};
  };
  double vtk_total_time = 0.0;
  callbacks.on_step_accepted = [&](svmp::FE::timestepping::TimeHistory& h) {
    requireCollectiveLevelSetMaintenanceRequestSchedule(
        level_set_maintenance,
        LevelSetMaintenanceScheduleStage::
            AcceptedEndpointPostStep,
        h.stepIndex(),
        velocity_extension_artifact_comm);
    energy.requireAcceptedResidualWorkStage(h);
    fitted_ale_stage.commitAccepted(h);
    two_fluid_stage.commitAccepted(h);
    energy.bindAcceptedEndpoint(h, contact_stage);
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_accepted step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << std::endl;
    logCutGeometryMemory("step_accepted");
    const auto accepted_post_maintenance_state_revision =
        has_conservative_phase
            ? collectiveMeshBoundaryStateFingerprint(
                  h.u(), velocity_extension_artifact_comm)
            : 0u;
    writeAcceptedConservativePhaseArtifacts(
        params,
        level_set_maintenance,
        accepted_phase_candidate,
        static_cast<std::uint64_t>(h.stepIndex()),
        static_cast<svmp::FE::Real>(h.time()),
        static_cast<svmp::FE::Real>(h.dt()),
        accepted_post_maintenance_state_revision,
        velocity_extension_artifact_comm);
    accepted_phase_candidate = ConservativePhaseCandidateResult{};
    pending_phase_publication_state =
        LevelSetMaintenancePublicationState::Staged;
    pending_phase_geometry_commit_required = false;
    pending_phase_geometry_commit_returned = false;
    pending_phase_ledger_transaction_inactive = false;
    if (acceptedPressureUpdateDiagnosticEnabled() &&
        !accepted_pressure_update_previous_solution.empty()) {
      const auto current_solution = gatherFeOrderedSolution(
          h.u(), velocity_extension_artifact_comm);
      logAcceptedPressureUpdateDiagnostic(
          *sim.fe_system,
          params,
          std::span<const svmp::FE::Real>(
              accepted_pressure_update_previous_solution.data(),
              accepted_pressure_update_previous_solution.size()),
          std::span<const svmp::FE::Real>(
              current_solution.data(),
              current_solution.size()),
          h.stepIndex(),
          h.time(),
          h.dt(),
          "post_accept",
          /*honor_fail_on_trigger=*/true);
    }
    const bool maintenance_work_due = std::any_of(
        level_set_maintenance.begin(),
        level_set_maintenance.end(),
        [&](const auto& request) {
          return !request.configuration->transport.conservative_phase.enabled &&
                 (svmp::FE::level_set::
                      shouldApplyLevelSetKinematicReconciliation(
                          request.configuration->transport
                              .kinematic_reconciliation,
                          h.stepIndex()) ||
                  svmp::FE::level_set::
                      shouldApplyLevelSetSignDefinitePatchBounds(
                          request.configuration->transport
                              .sign_definite_patch_bounds,
                          h.stepIndex()) ||
                  svmp::FE::level_set::
                      shouldApplyLevelSetInterfacePreservingRedistancing(
                          request.configuration->transport
                              .interface_preserving_redistancing,
                          h.stepIndex()));
        });
    const bool postaccept_maintenance_topology_tracking_required =
        maintenance_work_due && track_transient_cut_topology;
    std::optional<std::uint64_t>
        postaccept_maintenance_baseline_topology_key;
    std::optional<std::uint64_t>
        postaccept_maintenance_final_topology_key;
    ActiveCutContextRefreshReport cut_report{};
    if (postaccept_maintenance_topology_tracking_required) {
      // Establish the geometry of the already accepted physical endpoint
      // before any maintenance functional is evaluated or any geometry
      // checkpoint is captured.  The work ledger, geometry transaction backup,
      // and topology baseline must all describe this same authoritative
      // endpoint.  A cached refresh keeps the installed context only when its
      // content signature (level-set bytes, level-set DOF layout, mesh
      // revisions, request policy) and its snapshot source revisions prove
      // that it was generated from this endpoint; otherwise it rebuilds.
      // SVMP_DISABLE_POSTACCEPT_CUT_REUSE=1 restores the unconditional
      // rebuild.
      if (parseBoolEnv("SVMP_DISABLE_POSTACCEPT_CUT_REUSE", false)) {
        cut_refresh_cache->last_signature.reset();
        cut_refresh_cache->last_vector_signature.reset();
      }
      const auto accepted_endpoint_report =
          refreshActiveCutIntegrationContextCached(
              sim,
              params,
              h.u(),
              *cut_lifecycle,
              *cut_refresh_cache,
              "accepted_endpoint_before_postaccept_maintenance");
      const bool local_baseline_complete =
          (accepted_endpoint_report.refreshed ||
           accepted_endpoint_report.reused_installed_context) &&
          accepted_endpoint_report.request_policy_key ==
              expected_cut_request_policy_key &&
          accepted_endpoint_report.topology_key != 0u &&
          cut_refresh_cache->topology_key.has_value() &&
          *cut_refresh_cache->topology_key ==
              accepted_endpoint_report.topology_key;
      const auto [minimum_baseline_present,
                  maximum_baseline_present] =
          globalMinMaxUint64(
              local_baseline_complete ? 1u : 0u,
              velocity_extension_artifact_comm);
      const auto [minimum_baseline_key,
                  maximum_baseline_key] =
          globalMinMaxUint64(
              local_baseline_complete
                  ? accepted_endpoint_report.topology_key
                  : 0u,
              velocity_extension_artifact_comm);
      if (minimum_baseline_present != 1u ||
          maximum_baseline_present != 1u ||
          minimum_baseline_key == 0u ||
          minimum_baseline_key != maximum_baseline_key) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Postaccept level-set "
            "maintenance requires one complete communicator-consistent "
            "accepted-endpoint cut-topology baseline.");
      }
      postaccept_maintenance_baseline_topology_key =
          accepted_endpoint_report.topology_key;
      cut_report = accepted_endpoint_report;
    }
    std::vector<
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        pre_maintenance_functionals;
    std::vector<svmp::FE::Real> pre_maintenance_solution;
    if (maintenance_work_due) {
      pre_maintenance_solution =
          gatherFeOrderedSolution(
              h.u(), velocity_extension_artifact_comm);
      pre_maintenance_functionals =
          currentAcceptedFunctionalLayouts(sim);
    }
    std::vector<svmp::FE::LogRecord> maintenance_commit_logs;
    std::unique_ptr<LevelSetMaintenanceGeometryTransaction>
        maintenance_geometry_transaction;
    bool level_set_maintenance_changed = false;
    const auto maintenance_work_mark = level_set_maintenance_work.mark();
    auto staged_maintenance_functionals =
        levelSetMaintenanceFunctionalValues(
            sim,
            pre_maintenance_functionals,
            pre_maintenance_solution);
    std::vector<LevelSetMaintenanceRequest>
        maintenance_requests_before_transaction;
    std::vector<svmp::FE::Real>
        maintenance_current_before_transaction;
    std::vector<std::vector<svmp::FE::Real>>
        maintenance_history_before_transaction;
    std::optional<svmp::FE::timestepping::TimeHistory::RateStateSnapshot>
        maintenance_rate_before_transaction;
    bool maintenance_checkpoint_restored = false;
    bool maintenance_requests_checkpoint_restored = false;
    bool maintenance_current_checkpoint_restored = false;
    bool maintenance_rate_checkpoint_restored = false;
    bool maintenance_rate_publication_started = false;
    std::vector<std::uint8_t>
        maintenance_history_checkpoint_levels_restored;
    bool maintenance_checkpoint_ghosts_restored = false;
    if (maintenance_work_due) {
      const auto local_checkpoint_history_depth = h.historyDepth();
      if (globalAnyBool(
              local_checkpoint_history_depth <= 0,
              velocity_extension_artifact_comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Postaccept maintenance checkpoint "
            "requires positive history depth on every FE communicator rank.");
      }
      const auto [minimum_checkpoint_history_depth,
                  maximum_checkpoint_history_depth] =
          globalMinMaxUint64(
              static_cast<std::uint64_t>(local_checkpoint_history_depth),
              velocity_extension_artifact_comm);
      if (minimum_checkpoint_history_depth !=
          maximum_checkpoint_history_depth) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Postaccept maintenance checkpoint "
            "history depth differs across the FE communicator.");
      }
      std::vector<svmp::FE::backends::GenericVector*> checkpoint_targets;
      std::exception_ptr local_checkpoint_failure;
      try {
        if (!sim.backend) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Postaccept maintenance rate "
              "checkpoint requires an active backend factory.");
        }
        maintenance_requests_before_transaction =
            level_set_maintenance;
        checkpoint_targets.resize(
            static_cast<std::size_t>(local_checkpoint_history_depth) + 1u);
        checkpoint_targets.front() = &h.u();
        maintenance_history_before_transaction.resize(
            static_cast<std::size_t>(local_checkpoint_history_depth));
        maintenance_history_checkpoint_levels_restored.assign(
            static_cast<std::size_t>(local_checkpoint_history_depth), 0u);
        for (int level = 1; level <= local_checkpoint_history_depth;
             ++level) {
          checkpoint_targets[static_cast<std::size_t>(level)] =
              &h.uPrevK(level);
        }
      } catch (...) {
        local_checkpoint_failure = std::current_exception();
      }
      requireCollectivePhasePreparation(
          local_checkpoint_failure,
          velocity_extension_artifact_comm,
          "postaccept_maintenance_checkpoint_schedule");

      local_checkpoint_failure = nullptr;
      try {
        maintenance_rate_before_transaction.emplace(
            h.snapshotRateState(*sim.backend));
      } catch (...) {
        local_checkpoint_failure = std::current_exception();
      }
      requireCollectivePhasePreparation(
          local_checkpoint_failure,
          velocity_extension_artifact_comm,
          "postaccept_maintenance_rate_checkpoint_capture");

      auto captured_current =
          capturePostacceptMaintenanceVectorCollectively(
              *checkpoint_targets.front(),
              velocity_extension_artifact_comm);
      maintenance_current_before_transaction.swap(captured_current);
      for (int level = 1; level <= local_checkpoint_history_depth;
           ++level) {
        auto captured_history =
            capturePostacceptMaintenanceVectorCollectively(
                *checkpoint_targets[static_cast<std::size_t>(level)],
                velocity_extension_artifact_comm);
        maintenance_history_before_transaction[
            static_cast<std::size_t>(level - 1)]
            .swap(captured_history);
      }

      std::exception_ptr local_ledger_begin_failure;
      try {
        level_set_maintenance_work.beginTransaction(
            application::core::LevelSetMaintenanceWorkTransaction{
                .transaction_id =
                    ++level_set_maintenance_transaction_sequence,
                .step = static_cast<std::uint64_t>(h.stepIndex()),
                .attempt = std::max<std::uint64_t>(
                    1u, step_acceptance_attempt),
                .time = h.time(),
                .dt = h.dt(),
                .declared_stage =
                    application::core::
                        LevelSetMaintenanceDeclaredStage::
                            AcceptedEndpointPostStep,
                .extension_map_revision =
                    currentLevelSetVelocityExtensionRevision(
                        *sim.fe_system, level_set_advection_velocity),
            });
      } catch (...) {
        local_ledger_begin_failure = std::current_exception();
      }
      requireCollectivePhasePreparation(
          local_ledger_begin_failure,
          velocity_extension_artifact_comm,
          "postaccept_maintenance_ledger_begin");
    }
    const auto restore_maintenance_checkpoint = [&] {
      if (!maintenance_work_due ||
          maintenance_checkpoint_restored) {
        return;
      }
      if (!maintenance_requests_checkpoint_restored) {
        std::exception_ptr local_request_restore_failure;
        bool local_request_restore_completed = false;
        try {
          level_set_maintenance =
              maintenance_requests_before_transaction;
          local_request_restore_completed = true;
        } catch (...) {
          local_request_restore_failure = std::current_exception();
        }
        requirePostacceptMaintenanceRecoveryPhase(
            local_request_restore_failure,
            velocity_extension_artifact_comm,
            "restore_request_checkpoint");
        maintenance_requests_checkpoint_restored =
            local_request_restore_completed;
      }
      if (!maintenance_current_checkpoint_restored) {
        assignPostacceptMaintenanceVectorCollectively(
            h.u(),
            maintenance_current_before_transaction,
            velocity_extension_artifact_comm);
        maintenance_current_checkpoint_restored = true;
      }

      const bool history_depth_matches =
          maintenance_history_before_transaction.size() ==
              static_cast<std::size_t>(h.historyDepth()) &&
          maintenance_history_checkpoint_levels_restored.size() ==
              maintenance_history_before_transaction.size();
      if (!collectivePostacceptMaintenanceRecoveryPhaseReady(
              history_depth_matches,
              velocity_extension_artifact_comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set maintenance "
            "rollback history depth changed or diverged across the "
            "communicator.");
      }
      for (int level = 1; level <= h.historyDepth(); ++level) {
        const auto index =
            static_cast<std::size_t>(level - 1);
        if (maintenance_history_checkpoint_levels_restored[index] !=
            0u) {
          continue;
        }
        assignPostacceptMaintenanceVectorCollectively(
            h.uPrevK(level),
            maintenance_history_before_transaction[index],
            velocity_extension_artifact_comm);
        maintenance_history_checkpoint_levels_restored[index] = 1u;
      }
      const bool all_history_levels_restored =
          history_depth_matches &&
          std::all_of(
              maintenance_history_checkpoint_levels_restored.begin(),
              maintenance_history_checkpoint_levels_restored.end(),
              [](std::uint8_t value) { return value != 0u; });
      if (!maintenance_rate_checkpoint_restored) {
        const auto [minimum_rate_publication_started,
                    maximum_rate_publication_started] =
            globalMinMaxUint64(
                maintenance_rate_publication_started ? 1u : 0u,
                velocity_extension_artifact_comm);
        (void)minimum_rate_publication_started;
        const bool restore_rate_checkpoint =
            maximum_rate_publication_started != 0u;
        const bool local_rate_restore_ready =
            !restore_rate_checkpoint ||
            maintenance_rate_before_transaction.has_value();
        if (!collectivePostacceptMaintenanceRecoveryPhaseReady(
                local_rate_restore_ready,
                velocity_extension_artifact_comm)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set maintenance rate "
              "rollback lacks a captured RateStateSnapshot on at least one "
              "communicator rank.");
        }
        if (restore_rate_checkpoint) {
          // The explicit target/start flag is set before any uDot mutation.
          // Never infer rollback need from a backend value revision.
          h.restoreRateState(*maintenance_rate_before_transaction);
        }
        maintenance_rate_checkpoint_restored = true;
      }
      if (!maintenance_checkpoint_ghosts_restored &&
          maintenance_requests_checkpoint_restored &&
          maintenance_current_checkpoint_restored &&
          all_history_levels_restored &&
          maintenance_rate_checkpoint_restored) {
        const bool local_restore_complete =
            maintenance_requests_checkpoint_restored &&
            maintenance_current_checkpoint_restored &&
            all_history_levels_restored &&
            maintenance_rate_checkpoint_restored;
        if (!collectivePostacceptMaintenanceRecoveryPhaseReady(
                local_restore_complete,
                velocity_extension_artifact_comm)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set maintenance "
              "rollback is incomplete on another communicator rank.");
        }
        // Each generic state assignment completed its own ghost refresh.  A
        // restored RateStateSnapshot already owns the captured backend-native
        // overlap state, so no unfenced multi-vector refresh remains here.
        maintenance_checkpoint_ghosts_restored = true;
      }
      maintenance_checkpoint_restored =
          maintenance_requests_checkpoint_restored &&
          maintenance_current_checkpoint_restored &&
          all_history_levels_restored &&
          maintenance_rate_checkpoint_restored &&
          maintenance_checkpoint_ghosts_restored;
      if (!collectivePostacceptMaintenanceRecoveryPhaseReady(
              maintenance_checkpoint_restored,
              velocity_extension_artifact_comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set maintenance "
            "rollback checkpoint remains incomplete on at least one "
            "communicator rank.");
      }
    };
    bool maintenance_transaction_published = false;
    bool maintenance_transaction_rejected = false;
    bool maintenance_publication_started = false;
    bool maintenance_geometry_commit_required = false;
    bool maintenance_geometry_commit_returned = false;
    bool maintenance_ledger_commit_completed = false;
    bool maintenance_rejection_records_logged = false;
    const auto rollback_and_reject_maintenance =
        [&](PostacceptMaintenanceRollbackReason reason) {
          const auto rollback_scope =
              PostacceptMaintenanceRollbackScope::
                  AcceptedEndpointMaintenanceOnly;
          std::exception_ptr local_geometry_restore_failure;
          if (maintenance_geometry_transaction) {
            try {
              maintenance_geometry_transaction->rollback();
            } catch (...) {
              local_geometry_restore_failure =
                  std::current_exception();
            }
          }
          const bool local_geometry_restored =
              local_geometry_restore_failure == nullptr &&
              (!maintenance_geometry_transaction ||
               !maintenance_geometry_transaction->active());
          if (!collectivePostacceptMaintenanceRecoveryPhaseReady(
                  local_geometry_restored,
                  velocity_extension_artifact_comm)) {
            if (local_geometry_restore_failure != nullptr) {
              std::rethrow_exception(
                  local_geometry_restore_failure);
            }
            throw std::runtime_error(
                "[svMultiPhysics::Application] Another active FE "
                "communicator rank failed to roll back postaccept "
                "maintenance geometry/cache state.");
          }
          if (maintenance_geometry_transaction &&
              !maintenance_geometry_transaction->active()) {
            maintenance_geometry_transaction.reset();
          }

          restore_maintenance_checkpoint();
          const bool local_checkpoint_restored =
              !maintenance_work_due ||
              maintenance_checkpoint_restored;
          if (!collectivePostacceptMaintenanceRecoveryPhaseReady(
                  local_checkpoint_restored,
                  velocity_extension_artifact_comm)) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Postaccept maintenance "
                "request/current/history rollback is incomplete on at "
                "least one communicator rank.");
          }

          bool restored_topology_verified =
              !postaccept_maintenance_topology_tracking_required;
          if (postaccept_maintenance_topology_tracking_required) {
            cut_refresh_cache->last_signature.reset();
            cut_refresh_cache->last_vector_signature.reset();
            const auto restored_endpoint_report =
                refreshActiveCutIntegrationContextCached(
                    sim,
                    params,
                    h.u(),
                    *cut_lifecycle,
                    *cut_refresh_cache,
                    "postaccept_maintenance_rollback_verification");
            const bool local_restored_topology_complete =
                postaccept_maintenance_baseline_topology_key
                    .has_value() &&
                *postaccept_maintenance_baseline_topology_key != 0u &&
                restored_endpoint_report.refreshed &&
                restored_endpoint_report.request_policy_key ==
                    expected_cut_request_policy_key &&
                restored_endpoint_report.topology_key ==
                    *postaccept_maintenance_baseline_topology_key &&
                cut_refresh_cache->topology_key.has_value() &&
                *cut_refresh_cache->topology_key ==
                    *postaccept_maintenance_baseline_topology_key;
            const auto [minimum_restored_topology_key,
                        maximum_restored_topology_key] =
                globalMinMaxUint64(
                    local_restored_topology_complete
                        ? restored_endpoint_report.topology_key
                        : 0u,
                    velocity_extension_artifact_comm);
            restored_topology_verified =
                local_restored_topology_complete &&
                minimum_restored_topology_key != 0u &&
                minimum_restored_topology_key ==
                    maximum_restored_topology_key &&
                minimum_restored_topology_key ==
                    *postaccept_maintenance_baseline_topology_key;
            if (!collectivePostacceptMaintenanceRecoveryPhaseReady(
                    restored_topology_verified,
                    velocity_extension_artifact_comm)) {
              throw std::runtime_error(
                  "[svMultiPhysics::Application] Postaccept maintenance "
                  "rollback could not verify the accepted-endpoint "
                  "cut-topology baseline communicator-wide.");
            }
            cut_report = restored_endpoint_report;
          }

          const auto staged_trial_row_count =
              level_set_maintenance_work.trialRows().size();
          const auto [minimum_staged_trial_row_count,
                      maximum_staged_trial_row_count] =
              globalMinMaxUint64(
                  static_cast<std::uint64_t>(
                      staged_trial_row_count),
                  velocity_extension_artifact_comm);
          const bool local_ledger_rejection_preflight_ready =
              level_set_maintenance_work.transactionActive() &&
              level_set_maintenance_work.acceptedRows().size() ==
                  maintenance_work_mark.accepted_rows &&
              level_set_maintenance_work.rejectedRows().size() ==
                  maintenance_work_mark.rejected_rows &&
              level_set_maintenance_work.acceptedAttempts().size() ==
                  maintenance_work_mark.accepted_attempts &&
              level_set_maintenance_work.rejectedAttempts().size() ==
                  maintenance_work_mark.rejected_attempts &&
              minimum_staged_trial_row_count ==
                  maximum_staged_trial_row_count &&
              minimum_staged_trial_row_count ==
                  static_cast<std::uint64_t>(
                      staged_trial_row_count);
          const bool local_rejection_evidence_may_publish =
              postacceptMaintenanceRejectionEvidenceMayPublish(
                  local_geometry_restored,
                  local_checkpoint_restored,
                  restored_topology_verified,
                  local_ledger_rejection_preflight_ready);
          if (!collectivePostacceptMaintenanceRecoveryPhaseReady(
                  local_rejection_evidence_may_publish,
                  velocity_extension_artifact_comm)) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Postaccept maintenance "
                "rollback recovery did not reach the collective terminal "
                "ledger-rejection preflight; no rejected work evidence "
                "was published.");
          }

          std::exception_ptr local_ledger_rejection_failure;
          try {
            level_set_maintenance_work.rejectTransaction();
          } catch (...) {
            local_ledger_rejection_failure =
                std::current_exception();
          }
          requirePostacceptMaintenanceRecoveryPhase(
              local_ledger_rejection_failure,
              velocity_extension_artifact_comm,
              "terminal_maintenance_ledger_rejection");
          const bool local_terminal_rejection_complete =
              !level_set_maintenance_work.transactionActive() &&
              level_set_maintenance_work.acceptedRows().size() ==
                  maintenance_work_mark.accepted_rows &&
              level_set_maintenance_work.acceptedAttempts().size() ==
                  maintenance_work_mark.accepted_attempts &&
              level_set_maintenance_work.rejectedRows().size() ==
                  maintenance_work_mark.rejected_rows +
                      staged_trial_row_count &&
              level_set_maintenance_work.rejectedAttempts().size() ==
                  maintenance_work_mark.rejected_attempts + 1u &&
              level_set_maintenance_work.trialRows().empty();
          if (!collectivePostacceptMaintenanceRecoveryPhaseReady(
                  local_terminal_rejection_complete,
                  velocity_extension_artifact_comm)) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Postaccept maintenance "
                "terminal ledger rejection is inconsistent across the "
                "communicator; rejected rows were not logged.");
          }

          maintenance_transaction_rejected = true;
          level_set_maintenance_changed = false;
          maintenance_commit_logs.clear();

          if (!maintenance_rejection_records_logged) {
            logLevelSetMaintenanceWorkPublishedSince(
                level_set_maintenance_work,
                maintenance_work_mark,
                application::core::LevelSetMaintenanceWorkStatus::Rejected);
            maintenance_rejection_records_logged = true;
          }

          const auto topology_evidence =
              postacceptMaintenanceTopologyEvidence(
                  postaccept_maintenance_baseline_topology_key,
                  postaccept_maintenance_final_topology_key);
          svmp::FE::logRecord(
              svmp::FE::LogRecord("LevelSetMaintenance", "maintenance transaction")
                  .diagnostic("level_set_maintenance_transaction")
                  .rootOnly()
                  .value("step", h.stepIndex())
                  .value("reason_code", postacceptMaintenanceRollbackReasonName(reason))
                  .value("rollback_scope", postacceptMaintenanceRollbackScopeName(rollback_scope))
                  .text("outcome", "maintenance_transaction_rejected")
                  .boolean("physical_step_retained", true)
                  .boolean("full_time_step_rollback", false)
                  .boolean("maintenance_state_change_retained", false)
                  .boolean("geometry_state_restored", true)
                  .boolean("request_current_history_state_restored", true)
                  .boolean("restored_topology_verified", restored_topology_verified)
                  .boolean("maintenance_work_ledger_rejected", true)
                  .boolean("free_surface_energy_ledger_rejected", false)
                  .boolean("topology_baseline_present", topology_evidence.baseline_present)
                  .value("topology_baseline_key", topology_evidence.baseline_key)
                  .boolean("topology_final_present", topology_evidence.final_present)
                  .value("topology_final_key", topology_evidence.final_key)
                  .boolean("topology_keys_equal", topology_evidence.keys_equal));
        };
    try {
      const auto ensure_maintenance_geometry_transaction = [&] {
        if (!maintenance_geometry_transaction) {
          std::unique_ptr<LevelSetMaintenanceGeometryTransaction>
              prepared_transaction;
          std::exception_ptr local_transaction_prepare_failure;
          try {
            prepared_transaction =
                std::make_unique<LevelSetMaintenanceGeometryTransaction>(
                    sim,
                    *cut_lifecycle,
                    *cut_refresh_cache,
                    transient_active_cut_requests);
          } catch (...) {
            local_transaction_prepare_failure =
                std::current_exception();
          }
          requireCollectivePhasePreparation(
              local_transaction_prepare_failure,
              velocity_extension_artifact_comm,
              "postaccept_maintenance_geometry_transaction_begin");
          maintenance_geometry_transaction =
              std::move(prepared_transaction);
        }
      };
      const LevelSetMaintenanceStageObserver observe_stage =
          [&](application::core::LevelSetMaintenanceWorkSubstage
                  substage,
              std::span<const svmp::FE::Real> before_candidate,
              std::span<const svmp::FE::Real> after_candidate) {
            ensure_maintenance_geometry_transaction();
            cut_report = maintenance_geometry_transaction->refresh(
                params, after_candidate);
            svmp::FE::level_set::stageLevelSetMaintenanceWorkRow(
                level_set_maintenance_work,
                maintenance_work_rows,
                staged_maintenance_functionals,
                substage,
                before_candidate,
                after_candidate);
          };
      const LevelSetMaintenanceCandidateValidator validate_candidate =
          [&](std::span<const svmp::FE::Real> candidate) {
            ensure_maintenance_geometry_transaction();
            // The final candidate usually equals the after-candidate of the
            // last maintenance stage, whose refresh already generated its
            // geometry.  The cached refresh keeps that context only when its
            // content signature and snapshot source revisions prove it, so
            // the topology key below is the one a rebuild would produce.
            // SVMP_DISABLE_POSTACCEPT_CUT_REUSE=1 restores the forced rebuild.
            auto validator_report = maintenance_geometry_transaction->refresh(
                params,
                candidate,
                /*force_rebuild=*/parseBoolEnv(
                    "SVMP_DISABLE_POSTACCEPT_CUT_REUSE", false));
            const bool keep_generating_report =
                validator_report.reused_installed_context &&
                cut_report.refreshed &&
                cut_report.topology_key == validator_report.topology_key &&
                cut_report.evaluated_state_source_revisions ==
                    validator_report.evaluated_state_source_revisions;
            if (!keep_generating_report) {
              // Otherwise cut_report keeps the full report of the refresh
              // that generated the installed context, for the accepted-step
              // log below.
              cut_report = std::move(validator_report);
            }
            if (postaccept_maintenance_topology_tracking_required &&
                cut_report.request_policy_key ==
                    expected_cut_request_policy_key &&
                cut_report.topology_key != 0u &&
                cut_refresh_cache->topology_key.has_value() &&
                *cut_refresh_cache->topology_key ==
                    cut_report.topology_key) {
              // This validator runs once after every maintenance transform.
              // Intermediate stage/line-search refreshes deliberately do not
              // contribute to the publication decision.
              postaccept_maintenance_final_topology_key =
                  cut_report.topology_key;
            } else if (
                postaccept_maintenance_topology_tracking_required) {
              postaccept_maintenance_final_topology_key.reset();
            }
          };
      level_set_maintenance_changed = applyLevelSetMaintenance(
          sim,
          h,
          level_set_maintenance,
          pending_contact_stages,
          pending_contact_stage_constraints,
          pending_contact_stage_solution,
          validate_candidate,
          observe_stage,
          &maintenance_commit_logs,
          &maintenance_rate_publication_started,
          kinematic_baselines);
      if (postaccept_maintenance_topology_tracking_required &&
          !level_set_maintenance_changed) {
        // Even a no-op maintenance schedule receives an explicit final
        // validator refresh.  Do not infer equality merely from the absence
        // of an algebraic update.
        const auto unchanged_final_candidate =
            gatherFeOrderedSolution(
                h.u(), velocity_extension_artifact_comm);
        validate_candidate(unchanged_final_candidate);
      }
      if (maintenance_work_due) {
        const auto topology_evidence =
            postacceptMaintenanceTopologyEvidence(
                postaccept_maintenance_baseline_topology_key,
                postaccept_maintenance_final_topology_key);
        const auto local_topology_decision =
            postaccept_maintenance_topology_tracking_required
                ? classifyPostacceptMaintenanceTopologyEvidence(
                      topology_evidence)
                : PostacceptMaintenanceTopologyDecision::Commit;
        const std::array<std::uint64_t, 7>
            local_topology_evidence_words{{
                postaccept_maintenance_topology_tracking_required
                    ? 1u
                    : 0u,
                topology_evidence.baseline_present ? 1u : 0u,
                topology_evidence.baseline_key,
                topology_evidence.final_present ? 1u : 0u,
                topology_evidence.final_key,
                topology_evidence.keys_equal ? 1u : 0u,
                static_cast<std::uint64_t>(local_topology_decision),
            }};
        bool topology_evidence_identical_across_communicator =
            true;
        for (const auto word : local_topology_evidence_words) {
          const auto [minimum_word, maximum_word] =
              globalMinMaxUint64(
                  word, velocity_extension_artifact_comm);
          topology_evidence_identical_across_communicator =
              minimum_word == maximum_word &&
              topology_evidence_identical_across_communicator;
        }
        const bool communicator_complete_equal_topology =
            postaccept_maintenance_topology_tracking_required &&
            topology_evidence_identical_across_communicator &&
            local_topology_decision ==
                PostacceptMaintenanceTopologyDecision::Commit;
        const bool communicator_complete_topology_mismatch =
            postaccept_maintenance_topology_tracking_required &&
            topology_evidence_identical_across_communicator &&
            local_topology_decision ==
                PostacceptMaintenanceTopologyDecision::
                    RejectMaintenance;
        std::vector<std::uint64_t> commit_state_words;
        commit_state_words.reserve(
            27u + 3u * static_cast<std::size_t>(
                            std::max(0, h.historyDepth())));
        for (const auto word : local_topology_evidence_words) {
          appendMaintenanceScheduleWord(
              commit_state_words, word);
        }
        appendMaintenanceScheduleWord(commit_state_words, 1u);
        appendMaintenanceScheduleBool(
            commit_state_words,
            static_cast<bool>(maintenance_geometry_transaction));
        appendMaintenanceScheduleBool(
            commit_state_words,
            maintenance_geometry_transaction &&
                maintenance_geometry_transaction->active());
        appendMaintenanceScheduleBool(
            commit_state_words, level_set_maintenance_changed);
        const auto current_after_maintenance =
            gatherFeOrderedSolution(
                h.u(), velocity_extension_artifact_comm);
        // Content is bound by the algebraic revisions of the gathered
        // vectors.  GenericVector::valueRevision() counts rank-local
        // modifications and differs between ranks, so it must not enter
        // the cross-rank commit words.
        appendMaintenanceScheduleWord(
            commit_state_words,
            levelSetMaintenanceAlgebraicRevision(
                current_after_maintenance));
        appendMaintenanceScheduleSigned(
            commit_state_words, h.historyDepth());
        bool local_transaction_invariants_satisfied =
            maintenance_history_before_transaction.size() ==
                static_cast<std::size_t>(h.historyDepth()) &&
            current_after_maintenance.size() ==
                maintenance_current_before_transaction.size() &&
            (!level_set_maintenance_changed ||
             (maintenance_geometry_transaction &&
              maintenance_geometry_transaction->active())) &&
            (!maintenance_geometry_transaction ||
             maintenance_geometry_transaction->active()) &&
            topology_evidence_identical_across_communicator &&
            (!postaccept_maintenance_topology_tracking_required ||
             local_topology_decision !=
                 PostacceptMaintenanceTopologyDecision::
                     InvariantFailure);
        for (int level = 1; level <= h.historyDepth(); ++level) {
          const auto history_after_maintenance =
              gatherFeOrderedSolution(
                  h.uPrevK(level),
                  velocity_extension_artifact_comm);
          appendMaintenanceScheduleWord(
              commit_state_words,
              levelSetMaintenanceAlgebraicRevision(
                  history_after_maintenance));
          appendMaintenanceScheduleWord(
              commit_state_words,
              static_cast<std::uint64_t>(
                  history_after_maintenance.size()));
          const auto index =
              static_cast<std::size_t>(level - 1);
          local_transaction_invariants_satisfied =
              index <
                  maintenance_history_before_transaction.size() &&
              history_after_maintenance.size() ==
                  maintenance_history_before_transaction[index].size() &&
              local_transaction_invariants_satisfied;
        }
        const auto [minimum_u_dot_present, maximum_u_dot_present] =
            globalMinMaxUint64(
                h.hasUDotState() ? 1u : 0u,
                velocity_extension_artifact_comm);
        const auto [minimum_u_ddot_present, maximum_u_ddot_present] =
            globalMinMaxUint64(
                h.hasUDDotState() ? 1u : 0u,
                velocity_extension_artifact_comm);
        const bool u_dot_presence_agrees =
            minimum_u_dot_present == maximum_u_dot_present;
        const bool u_ddot_presence_agrees =
            minimum_u_ddot_present == maximum_u_ddot_present;
        appendMaintenanceScheduleBool(
            commit_state_words, h.hasUDotState());
        appendMaintenanceScheduleBool(
            commit_state_words, h.hasUDDotState());
        appendMaintenanceScheduleBool(
            commit_state_words,
            maintenance_rate_publication_started);
        if (u_dot_presence_agrees && maximum_u_dot_present != 0u) {
          const auto u_dot_after_maintenance =
              gatherFeOrderedSolution(
                  h.uDot(), velocity_extension_artifact_comm);
          appendMaintenanceScheduleWord(
              commit_state_words,
              levelSetMaintenanceAlgebraicRevision(
                  u_dot_after_maintenance));
          appendMaintenanceScheduleWord(
              commit_state_words,
              static_cast<std::uint64_t>(
                  u_dot_after_maintenance.size()));
          local_transaction_invariants_satisfied =
              u_dot_after_maintenance.size() ==
                  current_after_maintenance.size() &&
              local_transaction_invariants_satisfied;
        } else {
          appendMaintenanceScheduleWord(commit_state_words, 0u);
          appendMaintenanceScheduleWord(commit_state_words, 0u);
        }
        if (u_ddot_presence_agrees && maximum_u_ddot_present != 0u) {
          const auto u_ddot_after_maintenance =
              gatherFeOrderedSolution(
                  h.uDDot(), velocity_extension_artifact_comm);
          appendMaintenanceScheduleWord(
              commit_state_words,
              levelSetMaintenanceAlgebraicRevision(
                  u_ddot_after_maintenance));
          appendMaintenanceScheduleWord(
              commit_state_words,
              static_cast<std::uint64_t>(
                  u_ddot_after_maintenance.size()));
          local_transaction_invariants_satisfied =
              u_ddot_after_maintenance.size() ==
                  current_after_maintenance.size() &&
              local_transaction_invariants_satisfied;
        } else {
          appendMaintenanceScheduleWord(commit_state_words, 0u);
          appendMaintenanceScheduleWord(commit_state_words, 0u);
        }
        local_transaction_invariants_satisfied =
            u_dot_presence_agrees &&
            u_ddot_presence_agrees &&
            (!maintenance_rate_publication_started ||
             maximum_u_dot_present != 0u) &&
            local_transaction_invariants_satisfied;
        const auto request_schedule =
            canonicalLevelSetMaintenanceRequestSchedule(
                level_set_maintenance,
                LevelSetMaintenanceScheduleStage::
                    AcceptedEndpointPostStep,
                h.stepIndex());
        appendCanonicalLevelSetMaintenanceScheduleSection(
            commit_state_words, request_schedule);
        const auto geometry_state =
            canonicalLevelSetMaintenanceGeometryState(
                sim, level_set_maintenance);
        appendCanonicalLevelSetMaintenanceGeometrySection(
            commit_state_words, geometry_state);
        local_transaction_invariants_satisfied =
            request_schedule.supported &&
            geometry_state.supported &&
            local_transaction_invariants_satisfied;
        const auto non_topology_decision =
            application::core::
                collectiveLevelSetMaintenanceTransactionDecision(
                    level_set_maintenance_work,
                    local_transaction_invariants_satisfied,
                    velocity_extension_artifact_comm,
                    commit_state_words);
        if (non_topology_decision ==
            application::core::
                LevelSetMaintenanceTransactionDecision::Reject) {
          rollback_and_reject_maintenance(
              PostacceptMaintenanceRollbackReason::
                  CollectiveConsensusRejection);
          throw std::runtime_error(
              "[svMultiPhysics::Application] Postaccept level-set "
              "maintenance failed invariant/consensus validation; "
              "the accepted physical endpoint was restored and retained.");
        }
        if (communicator_complete_topology_mismatch) {
          rollback_and_reject_maintenance(
              PostacceptMaintenanceRollbackReason::
                  MaintenanceCutTopologyChanged);
        } else if (
            postaccept_maintenance_topology_tracking_required &&
            !communicator_complete_equal_topology) {
          rollback_and_reject_maintenance(
              PostacceptMaintenanceRollbackReason::
                  CollectiveConsensusRejection);
          throw std::logic_error(
              "[svMultiPhysics::Application] Postaccept maintenance "
              "topology evidence passed non-topology consensus without a "
              "complete equal or complete mismatched classification.");
        }
      }
      if (!maintenance_transaction_rejected) {
        maintenance_publication_started = true;
        maintenance_geometry_commit_required =
            static_cast<bool>(maintenance_geometry_transaction);
        maintenance_geometry_commit_returned =
            !maintenance_geometry_commit_required;
        try {
          if (maintenance_geometry_transaction) {
            maintenance_geometry_transaction->commit();
          }
          maintenance_geometry_commit_returned = true;
          if (level_set_maintenance_work.transactionActive()) {
            level_set_maintenance_work.commitTransaction();
          }
          maintenance_ledger_commit_completed = true;
          maintenance_transaction_published = true;
        } catch (...) {
          maintenance_ledger_commit_completed =
              !level_set_maintenance_work.transactionActive();
          svmp::FE::logRecord(
              svmp::FE::LogRecord("LevelSetMaintenance", "maintenance transaction")
                  .diagnostic("level_set_maintenance_transaction")
                  .rootOnly()
                  .value("step", h.stepIndex())
                  .text("outcome", "publication_failed")
                  .boolean("geometry_commit_required", maintenance_geometry_commit_required)
                  .boolean("geometry_commit_returned", maintenance_geometry_commit_returned)
                  .boolean("ledger_transaction_inactive", maintenance_ledger_commit_completed)
                  .boolean("rollback_claimed", false));
          throw;
        }
      }
      if (maintenance_transaction_published &&
          maintenance_geometry_transaction) {
        svmp::FE::logRecord(
            svmp::FE::LogRecord("LevelSetMaintenance", "maintenance transaction")
                .diagnostic("level_set_maintenance_transaction")
                .rootOnly()
                .value("step", h.stepIndex())
                .text("outcome", "committed")
                .boolean("geometry_validated_before_commit", true)
                .boolean("accepted_state_change", level_set_maintenance_changed));
      }
      if (maintenance_transaction_published) {
        logLevelSetMaintenanceWorkPublishedSince(
            level_set_maintenance_work,
            maintenance_work_mark,
            application::core::LevelSetMaintenanceWorkStatus::Accepted);
      }
    } catch (...) {
      const auto failure = std::current_exception();
      if (maintenance_transaction_published ||
          maintenance_transaction_rejected) {
        std::rethrow_exception(failure);
      }
      if (maintenance_publication_started) {
        svmp::FE::logRecord(
            svmp::FE::LogRecord("LevelSetMaintenance", "maintenance transaction")
                .diagnostic("level_set_maintenance_transaction")
                .rootOnly()
                .value("step", h.stepIndex())
                .text("outcome", "publication_failed")
                .boolean("geometry_commit_required", maintenance_geometry_commit_required)
                .boolean("geometry_commit_returned", maintenance_geometry_commit_returned)
                .boolean("ledger_transaction_inactive", maintenance_ledger_commit_completed)
                .boolean("rollback_claimed", false));
        std::rethrow_exception(failure);
      }
      try {
        rollback_and_reject_maintenance(
            PostacceptMaintenanceRollbackReason::
                CandidateException);
      } catch (const std::exception& recovery_error) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set maintenance "
            "recovery failed after candidate failure: " +
            std::string(recovery_error.what()));
      }
      std::rethrow_exception(failure);
    }
    energy.recordAcceptedStepMaintenanceAccount(
        h,
        level_set_maintenance_work,
        pending_phase_work_mark.accepted_attempts,
        pending_phase_work_mark.accepted_rows,
        step_acceptance_attempt);
    if (!level_set_maintenance_changed) {
      cut_report = refreshActiveCutIntegrationContextCached(
          sim, params, h.u(), *cut_lifecycle, *cut_refresh_cache,
          "accepted_step");
    }
    svmp::FE::logRecords(maintenance_commit_logs);
    if (parseBoolEnv("SVMP_CUT_RULE_DUMP", false)) {
      dumpActiveCutVolumeRulesForProbe(*sim.fe_system, h.stepIndex());
    }
    (void)updateLevelSetAdvectionVelocities(
        sim, h, level_set_advection_velocity);
    reimposeAcceptedMasterBearingState(h, *sim.fe_system);
    recordAcceptedMeshBoundaryProvenanceAtEndpoint(
        sim,
        static_cast<std::uint64_t>(h.stepIndex()),
        h.time(),
        h.dt());
    energy.publishAcceptedEnergy(h, pending_contact_stages);
    clear_pending_contact_provenance();
    if (level_set_maintenance_changed && cut_report.refreshed) {
      svmp::FE::logRecord(
          svmp::FE::LogRecord("LevelSetMaintenancePipeline", "refreshed cut context")
              .diagnostic("level_set_maintenance_cut_context_refresh")
              .rootOnly()
              .value("step", h.stepIndex())
              .value("cut_context_revision", cut_report.value_revision)
              .value("cell_count", cut_report.cell_count)
              .value("corner_linearized_cells", cut_report.corner_linearized_cell_count)
              .value("active_cut_cells", cut_report.active_cut_cells)
              .value("active_quadrature_points", cut_report.active_quadrature_points)
              .value("domain_interface_quadrature_point_count", cut_report.domain_interface_quadrature_point_count)
              .value("domain_volume_quadrature_point_count", cut_report.domain_volume_quadrature_point_count)
              .value("domain_total_quadrature_point_count", cut_report.domain_total_quadrature_point_count)
              .value("backend_volume_quadrature_point_count", cut_report.backend_volume_quadrature_point_count)
              .value("backend_interface_quadrature_point_count", cut_report.backend_interface_quadrature_point_count)
              .value("backend_total_quadrature_point_count", cut_report.backend_volume_quadrature_point_count + cut_report.backend_interface_quadrature_point_count)
              .value("backend_elapsed_seconds", cut_report.backend_elapsed_seconds)
              .value("generated_cell_cache_hits", cut_report.generated_cell_cache_hits)
              .value("generated_cell_cache_misses", cut_report.generated_cell_cache_misses)
              .value("generated_cell_cache_unchanged_dof_hits", cut_report.generated_cell_cache_unchanged_dof_hits)
              .value("generated_cell_refresh_candidates", cut_report.generated_cell_refresh_candidates)
              .value("generated_cell_directly_affected", cut_report.generated_cell_directly_affected)
              .value("generated_cell_affected_neighborhood", cut_report.generated_cell_affected_neighborhood)
              .value("generated_domain_cache_hits", cut_report.generated_domain_cache_hits)
              .value("process_vm_kb", cut_report.process_vm_kb)
              .value("process_rss_kb", cut_report.process_rss_kb)
              .value("basis_cache_entries", cut_report.basis_cache_entries)
              .value("negative_volume", cut_report.negative_volume)
              .value("negative_reference_volume", cut_report.negative_volume)
              .value("negative_physical_volume", cut_report.negative_physical_volume)
              .value("positive_volume", cut_report.positive_volume)
              .value("positive_reference_volume", cut_report.positive_volume)
              .value("positive_physical_volume", cut_report.positive_physical_volume)
              .value("cut_adjacent_facets", cut_report.cut_adjacent_facets));
    }
    capture_kinematic_baselines(h);
    logWetVolumeDiagnostics(
        activeCutVolumeRequests(params),
        sim.fe_system->cutIntegrationContext(),
        sim.fe_system->meshAccess(),
        sim.primary_mesh ? sim.primary_mesh->n_cells() : 0u,
        activeFESystemCommunicator(*sim.fe_system),
        h.stepIndex(),
        h.time(),
        initial_wet_volume_by_key);
    auto vtk_start = std::chrono::steady_clock::now();
    outputResults(sim, params, h.stepIndex(), h.time(), pvd);
    vtk_total_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - vtk_start).count();
  };
  callbacks.on_step_rejected = [&](const svmp::FE::timestepping::TimeHistory& h,
                                  svmp::FE::timestepping::StepRejectReason reason,
                                  const svmp::FE::timestepping::NewtonReport& nr) {
    if (nr.external_state_discontinuity) {
      if (reason != svmp::FE::timestepping::StepRejectReason::
                        CutTopologyChanged) {
        throw std::logic_error(
            "[svMultiPhysics::Application] An external-state discontinuity was not mapped to a cut-topology step rejection.");
      }
      oopCout()
          << "[svMultiPhysics::Application] Transient cut-topology attempt"
          << " diagnostic=cut_topology_early_step_rejection"
          << " accepted_topology_key="
          << cut_topology_attempt_tracker
                 ->acceptedTopologyKey()
                 .value_or(0u)
          << " first_mismatched_topology_key="
          << cut_topology_attempt_tracker
                 ->firstMismatchedTopologyKey()
                 .value_or(0u)
          << " last_observed_topology_key="
          << cut_topology_attempt_tracker
                 ->lastObservedTopologyKey()
                 .value_or(0u)
          << " restored_topology_key="
          << cut_refresh_cache->topology_key.value_or(0u)
          << " first_mismatch_provenance="
          << (cut_topology_attempt_tracker
                      ->firstMismatchProvenance()
                      .empty()
                  ? "unknown"
                  : cut_topology_attempt_tracker
                        ->firstMismatchProvenance())
          << " action=invalidate_and_retry" << std::endl;
      invalidate_transient_cut_topology_dependent_state();
    }
    energy.recordRejectedAttempt(h, reason);
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_rejected step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << " reason=" << step_reject_reason_to_string(reason)
              << " (newton: converged=" << nr.converged << " iters=" << nr.iterations
              << " outer_iters=" << nr.outer_iterations
              << " inner_iters_total=" << nr.inner_iterations_total
              << " ||r||=" << nr.residual_norm
              << " ||r_field||=" << nr.field_residual_norm
              << " ||r_aux||=" << nr.auxiliary_residual_norm << ")"
              << " outer_dynamic_relaxation_enabled="
              << nr.outer_dynamic_relaxation_enabled
              << " outer_dynamic_relaxation_updates="
              << nr.outer_dynamic_relaxation_updates
              << " outer_dynamic_relaxation_safeguards="
              << nr.outer_dynamic_relaxation_safeguards
              << " outer_dynamic_relaxation_resets="
              << nr.outer_dynamic_relaxation_resets
              << " outer_dynamic_relaxation_factor="
              << nr.outer_dynamic_relaxation_factor
              << " outer_relaxed_state_change_norm="
              << nr.outer_relaxed_state_change_norm
              << " outer_raw_contraction_ratio="
              << nr.outer_raw_contraction_ratio
              << " external_state_discontinuity="
              << nr.external_state_discontinuity
              << " external_state_discontinuity_restarts="
              << nr.external_state_discontinuity_restarts
              << std::endl;
  };
  callbacks.on_dt_updated = [&](double old_dt, double new_dt, int step_index, int attempt_index) {
    if (!oopTraceEnabled()) {
      return;
    }
    oopCout() << "[svMultiPhysics::Application] TimeLoop: dt_updated step=" << step_index
              << " attempt=" << attempt_index << " old_dt=" << old_dt << " new_dt=" << new_dt << std::endl;
  };

  // The free-surface step participants observe each converged operator
  // stage, in this order, before the conservative-phase snapshot of the
  // base callback.
  svmp::FE::timestepping::StepParticipantList free_surface_step_participants;
  free_surface_step_participants.add(contact_stage_participant);
  free_surface_step_participants.add(fitted_ale_stage_participant);
  free_surface_step_participants.add(two_fluid_stage_participant);
  free_surface_step_participants.add(energy_participant);
  callbacks = free_surface_step_participants.callbacks(std::move(callbacks));
  // The cut geometry's step participants: the entry part before each of its
  // callbacks, the exit part before the discard and after the commit.
  callbacks = cut_geometry_step_participants.callbacks(std::move(callbacks));

  capture_kinematic_baselines(*sim.time_history);
  svmp::FE::timestepping::TimeLoop loop(opts);
  oopCout() << "[svMultiPhysics::Application] TimeLoop: entering loop.run()" << std::endl;
  auto loop_start = std::chrono::steady_clock::now();
  const auto rep = loop.run(transient, *sim.backend, *sim.linear_solver, *sim.time_history, callbacks);
  double loop_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();
  oopCout() << "[svMultiPhysics::Application] TimeLoop: loop.run() returned success=" << rep.success
            << " steps_taken=" << rep.steps_taken << " final_time=" << rep.final_time
            << " message='" << rep.message << "'" << std::endl;

  // ===== PRINT TOP-LEVEL TIMING =====
  {
    const auto mpi_comm = svmp::MeshComm::world();
    if (mpi_comm.rank() == 0) {
      double solve_time = loop_total - vtk_total_time;
      fprintf(stderr,
        "\n*** TOP-LEVEL TIMING SUMMARY (rank 0) ***\n"
        "  Total time loop:      %10.6f s\n"
        "  Solve (Newton+linear):%10.6f s  (%5.1f%%)\n"
        "  VTK output:           %10.6f s  (%5.1f%%)\n"
        "*******************************************\n",
        loop_total,
        solve_time, 100.0 * solve_time / loop_total,
        vtk_total_time, 100.0 * vtk_total_time / loop_total);
    }
  }
  // ====================================

  // Whether the ghost halo was deep enough for a partition-independent
  // discretization (communicator-global totals; nonzero values mean the
  // result depends on the partition and <Ghost_layers> should grow).
  if (svmp::MeshComm::world().size() > 1 && svmp::MeshComm::world().rank() == 0) {
    oopCout() << "[svMultiPhysics::Application] Halo correctness "
              << svmp::FE::diagnostics::haloCorrectnessSummary() << std::endl;
  }
  // Small-cut aggregation candidates that had no root inside the aggregation
  // guards and received the rootless-island policy (opt-in
  // Small_cut_aggregation_rootless_fallback, D33; canonical totals).  A
  // nonzero candidates_total flags a result that relies on it.
  if (const auto guard_totals = svmp::FE::diagnostics::aggregationGuardRootlessTotals();
      (guard_totals.fallback_enabled ||
       guard_totals.root_path_guard_candidates_total +
               guard_totals.proposal_guard_candidates_total >
           0) &&
      svmp::MeshComm::world().rank() == 0) {
    oopCout() << "[svMultiPhysics::Application] Aggregation guard fallback "
              << svmp::FE::diagnostics::aggregationGuardRootlessSummary() << std::endl;
  }

  if (!rep.success) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve failed: " + rep.message +
                             ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto comm = svmp::MeshComm::world();
  if (pvd && comm.rank() == 0) {
    write_pvd_collection(*pvd);
    if (!pvd->entries.empty()) {
      oopCout() << "[svMultiPhysics::Application] Wrote PVD: " << pvd->pvd_path.string() << std::endl;
    }
  }
}

void ApplicationDriver::outputResults(const SimulationComponents& sim, const Parameters& params, int step, double time,
                                      VtkTimeSeriesCollection* pvd)
{
  if (!params.general_simulation_parameters.save_results_to_vtk_format.defined() ||
      !params.general_simulation_parameters.save_results_to_vtk_format.value()) {
    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: disabled (<Save_results_to_VTK_format>=false)." << std::endl;
    }
    return;
  }

  if (!sim.primary_mesh || !sim.fe_system || !sim.time_history) {
    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: missing mesh/system/history; skipping." << std::endl;
    }
    return;
  }

  const int save_incr = std::max(1, params.general_simulation_parameters.increment_in_saving_vtk_files.value());
  const int save_ats = std::max(0, params.general_simulation_parameters.start_saving_after_time_step.value());
  bool force_final_time_output = false;
  if (parseBoolEnv("SVMP_VTK_OUTPUT_FINAL_TIME", false) &&
      params.general_simulation_parameters.number_of_time_steps.defined() &&
      params.general_simulation_parameters.time_step_size.defined()) {
    const double final_time =
        static_cast<double>(params.general_simulation_parameters.number_of_time_steps.value()) *
        params.general_simulation_parameters.time_step_size.value();
    const double final_tol =
        100.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(final_time));
    force_final_time_output = time + final_tol >= final_time;
  }
  if (!force_final_time_output && (step < save_ats || (step % save_incr) != 0)) {
    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: skipping step=" << step
                << " (start_after=" << save_ats << " increment=" << save_incr << ")" << std::endl;
    }
    return;
  }

  const auto output_total_start = Clock::now();
  const auto setup_start = Clock::now();
  const auto comm = svmp::MeshComm::world();
  const bool is_root = (comm.rank() == 0);
  const bool mpi_parallel = (comm.size() > 1);

  std::filesystem::path out_dir = ".";
  if (params.general_simulation_parameters.save_results_in_folder.defined() &&
      !params.general_simulation_parameters.save_results_in_folder.value().empty()) {
    out_dir = params.general_simulation_parameters.save_results_in_folder.value();
  }
  std::filesystem::create_directories(out_dir);

  std::string prefix = "result";
  if (params.general_simulation_parameters.name_prefix_of_saved_vtk_files.defined() &&
      !params.general_simulation_parameters.name_prefix_of_saved_vtk_files.value().empty()) {
    prefix = params.general_simulation_parameters.name_prefix_of_saved_vtk_files.value();
  }

  std::ostringstream fname;
  fname << prefix << "_" << std::setw(3) << std::setfill('0') << step << (mpi_parallel ? ".pvtu" : ".vtu");
  const auto out_path = out_dir / fname.str();
  if (oopTraceEnabled() && is_root) {
    oopCout() << "[svMultiPhysics::Application] VTK output: begin step=" << step << " time=" << time
              << " file='" << out_path.string() << "'" << std::endl;
  }

  svmp::FE::systems::SystemStateView state;
  state.time = time;
  state.dt = sim.time_history->dt();
  state.dt_prev = sim.time_history->dtPrev();
  state.u = sim.time_history->uSpan();
  state.u_prev = sim.time_history->uPrevSpan();
  state.u_prev2 = sim.time_history->uPrev2Span();
  state.u_vector = &sim.time_history->u();
  state.u_prev_vector = &sim.time_history->uPrev();
  state.u_prev2_vector = &sim.time_history->uPrev2();
  state.u_history = sim.time_history->uHistorySpans();
  state.dt_history = sim.time_history->dtHistory();

  auto& mesh = *sim.primary_mesh;
  const int mesh_dim = mesh.dim();
  const auto& coords = mesh.X_ref();

  const auto ensure_point_field = [&](const std::string& name, std::size_t components) -> svmp::FieldHandle {
    if (mesh.has_field(svmp::EntityKind::Vertex, name)) {
      auto h = mesh.field_handle(svmp::EntityKind::Vertex, name);
      if (mesh.field_type(h) == svmp::FieldScalarType::Float64 && mesh.field_components(h) == components) {
        return h;
      }
      mesh.remove_field(h);
    }
    return mesh.attach_field(svmp::EntityKind::Vertex, name, svmp::FieldScalarType::Float64, components);
  };

  const double setup_seconds = secondsSince(setup_start);

  const auto output_fields = sim.fe_system->unknownFieldIdsInDofMapOrder();
  double primary_field_seconds = 0.0;
  std::vector<std::pair<std::string, double>> field_timings;
  std::vector<std::pair<std::string, bool>> field_fast_paths;
  field_timings.reserve(output_fields.size());
  field_fast_paths.reserve(output_fields.size());
  for (const auto field_id : output_fields) {
    const auto field_start = Clock::now();
    const auto& rec = sim.fe_system->fieldRecord(field_id);
    const auto ncomp = static_cast<std::size_t>(std::max(1, rec.components));

    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: evaluating field '" << rec.name
                << "' components=" << ncomp << std::endl;
    }

    auto h = ensure_point_field(rec.name, ncomp);
    auto* data = static_cast<double*>(mesh.field_data(h));
    if (!data) {
      throw std::runtime_error("[svMultiPhysics::Application] Failed to allocate VTK field '" + rec.name + "'.");
    }

    const auto nv = static_cast<svmp::FE::GlobalIndex>(mesh.n_vertices());
    const bool fast = sim.fe_system->evaluateFieldAtVertices(
        field_id, state, nv, std::span<double>(data, static_cast<std::size_t>(nv) * ncomp));

    if (!fast) {
      // Fallback: per-vertex spatial search + basis evaluation
      for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
        std::array<svmp::FE::Real, 3> p{0.0, 0.0, 0.0};
        for (int d = 0; d < mesh_dim; ++d) {
          p[static_cast<std::size_t>(d)] = static_cast<svmp::FE::Real>(coords[v * static_cast<std::size_t>(mesh_dim) +
                                                                             static_cast<std::size_t>(d)]);
        }

        const auto val = sim.fe_system->evaluateFieldAtPoint(field_id, state, p);
        if (!val) {
          throw std::runtime_error("[svMultiPhysics::Application] Failed to evaluate field '" + rec.name +
                                   "' at a mesh vertex for VTK output.");
        }

        for (std::size_t c = 0; c < ncomp; ++c) {
          data[v * ncomp + c] = static_cast<double>((*val)[c]);
        }
      }
    }

    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: field '" << rec.name << "' done." << std::endl;
    }
    const double field_seconds = secondsSince(field_start);
    primary_field_seconds += field_seconds;
    field_timings.emplace_back(rec.name, field_seconds);
    field_fast_paths.emplace_back(rec.name, fast);
  }

  const auto derived = sim.fe_system->derivedResults();
  double derived_seconds = 0.0;
  std::vector<std::pair<std::string, double>> derived_timings;
  derived_timings.reserve(derived.size());
  if (!derived.empty()) {
    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: evaluating "
                << derived.size() << " derived result field(s)." << std::endl;
      for (const auto& def : derived) {
        oopCout() << "[svMultiPhysics::Application] VTK output: derived field '" << def.name
                  << "' scope=" << svmp::FE::post::toString(def.scope)
                  << " policy=" << svmp::FE::post::toString(def.policy)
                  << " components=" << svmp::FE::post::componentCount(def.shape)
                  << std::endl;
      }
    }
    svmp::FE::post::DerivedResultEvaluator derived_evaluator(*sim.fe_system, state);
    for (const auto& def : derived) {
      const auto derived_start = Clock::now();
      derived_evaluator.evaluateToMeshField(mesh.local_mesh(), def);
      const double field_seconds = secondsSince(derived_start);
      derived_seconds += field_seconds;
      derived_timings.emplace_back(def.name, field_seconds);
    }
    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: derived result fields done." << std::endl;
    }
  }

  const auto active_output_requests = activeCutVolumeRequests(params);
  if (const auto* cut_context = sim.fe_system->cutIntegrationContext()) {
    cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
        sim.fe_system->meshAccess());
  }
  if (!active_output_requests.empty() && is_root) {
    oopCout()
        << "[svMultiPhysics::Application] VTK output: ActiveFluid is a vertex-sign visualization indicator; "
        << "WetVolumeFraction is the generated cut-volume active-domain diagnostic."
        << std::endl;
  }
  const auto wet_fraction_fields = writeWetVolumeFractionOutput(
      mesh,
      active_output_requests,
      sim.fe_system->cutIntegrationContext());
  if (wet_fraction_fields > 0u && is_root) {
    oopCout() << "[svMultiPhysics::Application] VTK output: wrote "
              << wet_fraction_fields
              << " wet volume diagnostic cell field(s) from generated cut metadata."
              << std::endl;
  } else if (!active_output_requests.empty() && is_root) {
    oopCout()
        << "[svMultiPhysics::Application] WARNING VTK output did not write "
        << "WetVolumeFraction from generated cut metadata"
        << " step=" << step
        << " time=" << time
        << " requests=" << active_output_requests.size()
        << " has_cut_context="
        << (sim.fe_system->cutIntegrationContext() != nullptr ? "true" : "false")
        << " diagnostic=missing_wet_volume_fraction_output"
        << std::endl;
  }
  logActiveFluidWetFractionDisagreementWarnings(
      mesh,
      active_output_requests,
      sim.fe_system->cutIntegrationContext(),
      step,
      time);
  const auto active_fluid_report = writeActiveFluidVisualizationOutput(
      mesh,
      active_output_requests);
  if (!active_output_requests.empty() &&
      active_fluid_report.total_vertices > 0u &&
      active_fluid_report.active_vertices == 0u &&
      is_root) {
    oopCout()
        << "[svMultiPhysics::Application] WARNING ActiveFluid output indicator has "
        << "zero active vertices"
        << " step=" << step
        << " time=" << time
        << " total_vertices=" << active_fluid_report.total_vertices
        << " dry_vertices=" << active_fluid_report.dry_vertices
        << " diagnostic=active_fluid_vertex_indicator_empty"
        << std::endl;
  }

  svmp::MeshIOOptions io{};
  io.format = mpi_parallel ? "pvtu" : "vtu";
  io.path = out_path.string();
  io.kv["binary"] = "true";
  io.kv["streaming"] = "true";
  const auto save_start = Clock::now();
  mesh.save_parallel(io);
  const double save_seconds = secondsSince(save_start);

  const auto pvd_start = Clock::now();
  if (is_root) {
    oopCout() << "[svMultiPhysics::Application] Wrote VTK: " << out_path.string() << std::endl;
    if (pvd && !pvd->pvd_path.empty()) {
      std::error_code ec;
      auto rel = std::filesystem::relative(out_path, pvd->pvd_path.parent_path(), ec);
      if (ec) {
        rel = out_path.filename();
      }
      const std::string rel_file = rel.generic_string();
      if (pvd->entries.empty() || pvd->entries.back().file != rel_file) {
        pvd->entries.push_back(VtkTimeSeriesCollection::Entry{time, rel_file});
      }
    }
  }
  const double pvd_seconds = secondsSince(pvd_start);

  const double output_total_seconds = secondsSince(output_total_start);
  const double accounted_seconds = setup_seconds + primary_field_seconds + derived_seconds +
                                   save_seconds + pvd_seconds;
  const double other_seconds = std::max(0.0, output_total_seconds - accounted_seconds);

  const auto total_stats = reduceOutputTiming(output_total_seconds, comm);
  const auto setup_stats = reduceOutputTiming(setup_seconds, comm);
  const auto primary_stats = reduceOutputTiming(primary_field_seconds, comm);
  const auto derived_stats = reduceOutputTiming(derived_seconds, comm);
  const auto save_stats = reduceOutputTiming(save_seconds, comm);
  const auto pvd_stats = reduceOutputTiming(pvd_seconds, comm);
  const auto other_stats = reduceOutputTiming(other_seconds, comm);

  std::vector<OutputTimingStats> field_stats;
  field_stats.reserve(field_timings.size());
  for (const auto& [_, seconds] : field_timings) {
    field_stats.push_back(reduceOutputTiming(seconds, comm));
  }

  std::vector<OutputTimingStats> derived_stats_by_field;
  derived_stats_by_field.reserve(derived_timings.size());
  for (const auto& [_, seconds] : derived_timings) {
    derived_stats_by_field.push_back(reduceOutputTiming(seconds, comm));
  }

  if (is_root) {
    std::fprintf(stderr,
                 "\n*** VTK OUTPUT SUB-TIMING step=%d time=%.16e (rank 0) ***\n",
                 step,
                 time);
    printOutputTimingLine("Total", total_stats, mpi_parallel, output_total_seconds);
    printOutputTimingLine("Setup/state", setup_stats, mpi_parallel, output_total_seconds);
    printOutputTimingLine("Primary fields", primary_stats, mpi_parallel, output_total_seconds);
    for (std::size_t i = 0; i < field_timings.size(); ++i) {
      const auto& [name, _] = field_timings[i];
      const auto fast = i < field_fast_paths.size() && field_fast_paths[i].second;
      const std::string label = "field " + name + (fast ? " [direct]" : " [fallback]");
      printOutputTimingLine(label.c_str(), field_stats[i], mpi_parallel, output_total_seconds);
    }
    printOutputTimingLine("Derived fields", derived_stats, mpi_parallel, output_total_seconds);
    for (std::size_t i = 0; i < derived_timings.size(); ++i) {
      const auto& [name, _] = derived_timings[i];
      const std::string label = "derived " + name;
      printOutputTimingLine(label.c_str(), derived_stats_by_field[i], mpi_parallel, output_total_seconds);
    }
    printOutputTimingLine("Mesh save_parallel", save_stats, mpi_parallel, output_total_seconds);
    printOutputTimingLine("PVD bookkeeping", pvd_stats, mpi_parallel, output_total_seconds);
    printOutputTimingLine("Other", other_stats, mpi_parallel, output_total_seconds);
    std::fprintf(stderr, "*******************************************************\n");
  }

  if (oopTraceEnabled() && is_root) {
    oopCout() << "[svMultiPhysics::Application] VTK output: done step=" << step << " time=" << time << std::endl;
  }
}

} // namespace core
} // namespace application

#include "Application/Core/ApplicationDriver.h"

#include "Application/Core/ActiveDomainOutput.h"
#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/LevelSetMaintenanceHistory.h"
#include "Application/Core/NearestPointIndex.h"
#include "Application/Core/OopMpiLog.h"
#include "Application/Core/SimulationBuilder.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Basis/BasisCache.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/LevelSet/LevelSetCurvatureProjection.h"
#include "FE/LevelSet/LevelSetImplicitCutQuadratureBackend.h"
#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"
#include "FE/LevelSet/LevelSetReinitialization.h"
#include "FE/LevelSet/LevelSetVolume.h"
#include "FE/PostProcessing/DerivedResultTypes.h"
#include "FE/PostProcessing/DerivedResultEvaluator.h"
#include "FE/Systems/CutIntegrationInvalidation.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/TimeStepping/NewtonSolver.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Core/MeshComm.h"
#include "Mesh/Topology/CellTopology.h"
#include "Parameters.h"
#include "tinyxml2.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <thread>
#include <vector>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using application::core::ActiveCutVolumeRequest;
using application::core::ActiveCutVolumeRequestOrigin;
using application::core::LevelSetActiveSide;
using application::core::NearestPointIndex;
using application::core::NearestPointRecord;
using application::core::activeCutVolumeRequests;

double secondsSince(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename CoordinateContainer>
std::array<double, 3> meshVertexPoint(const CoordinateContainer& coords,
                                      int mesh_dim,
                                      std::size_t vertex)
{
  std::array<double, 3> point{0.0, 0.0, 0.0};
  for (int d = 0; d < mesh_dim; ++d) {
    point[static_cast<std::size_t>(d)] =
        static_cast<double>(
            coords[vertex * static_cast<std::size_t>(mesh_dim) +
                   static_cast<std::size_t>(d)]);
  }
  return point;
}

struct OutputTimingStats {
  double local{0.0};
  double min{0.0};
  double mean{0.0};
  double max{0.0};
};

struct ProcessMemorySnapshot {
  long vm_kb{-1};
  long rss_kb{-1};
};

ProcessMemorySnapshot readProcessMemorySnapshot()
{
  ProcessMemorySnapshot snapshot;
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    std::istringstream fields(line);
    std::string key;
    long value = -1;
    std::string unit;
    if (!(fields >> key >> value >> unit)) {
      continue;
    }
    if (key == "VmSize:") {
      snapshot.vm_kb = value;
    } else if (key == "VmRSS:") {
      snapshot.rss_kb = value;
    }
  }
  return snapshot;
}

OutputTimingStats reduceOutputTiming(double local, const svmp::MeshComm& comm)
{
  OutputTimingStats stats;
  stats.local = local;
  stats.min = local;
  stats.mean = local;
  stats.max = local;

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    double sum = 0.0;
    MPI_Allreduce(&local, &stats.min, 1, MPI_DOUBLE, MPI_MIN, comm.native());
    MPI_Allreduce(&local, &stats.max, 1, MPI_DOUBLE, MPI_MAX, comm.native());
    MPI_Allreduce(&local, &sum, 1, MPI_DOUBLE, MPI_SUM, comm.native());
    stats.mean = sum / static_cast<double>(comm.size());
  }
#else
  (void)comm;
#endif

  return stats;
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
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, comm.rank(),
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
  std::ostringstream oss;
  oss << "[svMultiPhysics::Application] state_vector diagnostic=state_vector_fields"
      << " label='" << label << "'";
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
    oss << " [" << rec.name
        << " dofs=" << count
        << " norm=" << std::sqrt(std::max(0.0, sq_norm))
        << " mean=" << mean
        << " min=" << (count > 0 ? min_value : 0.0)
        << " max=" << (count > 0 ? max_value : 0.0)
        << "]";
  }
  application::core::oopCout() << oss.str() << std::endl;
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

svmp::FE::level_set::LevelSetReinitializationMethod
parseLevelSetReinitializationMethod(const std::string& raw)
{
  const auto value = normalized_token(raw);
  using Method = svmp::FE::level_set::LevelSetReinitializationMethod;
  if (value == "projection" || value == "signeddistanceprojection" ||
      value == "repairprojection") {
    return Method::Projection;
  }
  if (value == "hamiltonjacobi" || value == "hamiltonjacobipde" ||
      value == "pde") {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Reinitialization_method=HamiltonJacobiPDE "
        "is reserved until runtime Hamilton-Jacobi reinitialization is implemented; "
        "use 'Projection'.");
  }
  if (value == "fastmarching" || value == "fastmarchingmethod" ||
      value == "fmm") {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Reinitialization_method=FastMarching "
        "is reserved until runtime fast-marching reinitialization is implemented; "
        "use 'Projection'.");
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Reinitialization_method currently supports "
      "'Projection' only.");
}

svmp::FE::level_set::LevelSetTransportForm
parseLevelSetTransportForm(const std::string& raw)
{
  const auto value = normalized_token(raw);
  using Form = svmp::FE::level_set::LevelSetTransportForm;
  if (value == "advective" || value == "classical" || value == "standard") {
    return Form::Advective;
  }
  if (value == "conservative" || value == "conservativedivergence" ||
      value == "divergence" || value == "divergenceform") {
    return Form::ConservativeDivergence;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Level-set Transport_form must be one of "
      "'advective' or 'conservative_divergence'.");
}

std::string step_reject_reason_to_string(svmp::FE::timestepping::StepRejectReason r)
{
  using svmp::FE::timestepping::StepRejectReason;
  switch (r) {
    case StepRejectReason::NonlinearSolveFailed:
      return "NonlinearSolveFailed";
    case StepRejectReason::ErrorTooLarge:
      return "ErrorTooLarge";
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

struct LevelSetMaintenanceRequest {
  std::string level_set_field_name{"level_set"};
  double isovalue{0.0};
  svmp::FE::level_set::LevelSetTransportForm transport_form{
      svmp::FE::level_set::LevelSetTransportForm::Advective};
  svmp::FE::level_set::LevelSetReinitializationOptions reinitialization{};
  svmp::FE::level_set::LevelSetVolumeCorrectionOptions volume_correction{};
  bool curvature_projection_enabled{false};
  std::string curvature_field_name{};
  int curvature_projection_cadence_steps{1};
  svmp::FE::level_set::LevelSetCurvatureProjectionOptions curvature_projection{};
  bool volume_target_initialized{false};
  svmp::FE::Real volume_target{0.0};
};

struct CurvatureProjectionCacheEntry {
  bool valid{false};
  std::uint64_t signature{0};
  bool fast_valid{false};
  std::uint64_t fast_signature{0};
  svmp::FE::level_set::LevelSetCurvatureProjectionWorkspace workspace{};
  svmp::FE::level_set::LevelSetCurvatureProjectionResult last_result{};
};

struct CurvatureProjectionCache {
  std::map<std::string, CurvatureProjectionCacheEntry> entries;
};

struct LevelSetAdvectionVelocityRequest {
  std::string level_set_field_name{"level_set"};
  std::string source_velocity_field_name{"Velocity"};
  std::string target_velocity_field_name{"LevelSetAdvectionVelocity"};
  std::string extension_method{"nearest_active_vertex"};
  double isovalue{0.0};
  LevelSetActiveSide active_side{LevelSetActiveSide::Negative};
};

struct ActiveCutContextRefreshReport {
  bool refreshed{false};
  std::uint64_t topology_key{0};
  std::uint64_t request_policy_key{0};
  std::uint64_t value_revision{0};
  std::size_t cell_count{0};
  std::size_t corner_linearized_cell_count{0};
  std::size_t interface_fragments{0};
  std::size_t active_volume_regions{0};
  std::size_t active_cut_cells{0};
  std::size_t active_quadrature_points{0};
  std::size_t domain_interface_quadrature_point_count{0};
  std::size_t domain_volume_quadrature_point_count{0};
  std::size_t domain_total_quadrature_point_count{0};
  std::size_t backend_volume_quadrature_point_count{0};
  std::size_t backend_interface_quadrature_point_count{0};
  std::size_t cut_adjacent_facets{0};
  std::size_t basis_cache_entries{0};
  std::size_t generated_cell_cache_hits{0};
  std::size_t generated_cell_cache_misses{0};
  std::size_t generated_domain_cache_hits{0};
  std::size_t linear_full_cell_fast_path_count{0};
  double backend_elapsed_seconds{0.0};
  long process_vm_kb{-1};
  long process_rss_kb{-1};
  svmp::FE::Real negative_volume{0.0};
  svmp::FE::Real positive_volume{0.0};
  svmp::FE::Real negative_physical_volume{0.0};
  svmp::FE::Real positive_physical_volume{0.0};
};

struct WetVolumeDiagnostic {
  std::string level_set_field_name{};
  std::string domain_id{};
  int marker{-1};
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

struct ActiveSideRegionSummary {
  svmp::FE::Real active_volume{0.0};
  svmp::FE::Real pruned_volume{0.0};
  std::size_t active_volume_regions{0};
  std::size_t pruned_volume_regions{0};
  std::size_t active_quadrature_points{0};
  std::size_t active_wet_cells{0};
  std::size_t cut_cell_count{0};
  std::size_t full_wet_cell_count{0};
  std::size_t full_dry_cell_count{0};
  std::size_t nonfinite_measure_regions{0};
  std::size_t negative_measure_regions{0};
  std::size_t empty_quadrature_regions{0};
  svmp::FE::Real min_volume_fraction{std::numeric_limits<svmp::FE::Real>::infinity()};
  svmp::FE::Real max_volume_fraction{-std::numeric_limits<svmp::FE::Real>::infinity()};
};

struct CutAdjacentFacetScaleSummary {
  std::size_t metadata_count{0};
  std::size_t zero_scale_count{0};
  std::size_t nonfinite_scale_count{0};
  std::size_t capped_scale_count{0};
  svmp::FE::Real min_scale{std::numeric_limits<svmp::FE::Real>::infinity()};
  svmp::FE::Real max_scale{-std::numeric_limits<svmp::FE::Real>::infinity()};
  svmp::FE::Real mean_scale{0.0};
};

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
    const bool enabled =
        first_defined_bool_parameter(
            eq_params,
            {"Use_wet_extension_advection_velocity",
             "UseWetExtensionAdvectionVelocity",
             "Update_advection_velocity_from_wet_region",
             "UpdateAdvectionVelocityFromWetRegion"})
            .value_or(false);
    const auto source_field =
        first_defined_parameter(
            eq_params,
            {"Advection_velocity_from_field",
             "AdvectionVelocityFromField",
             "Source_velocity_field_name",
             "SourceVelocityFieldName",
             "Physical_velocity_field_name",
             "PhysicalVelocityFieldName"});
    if (!enabled && !source_field.has_value()) {
      continue;
    }

    LevelSetAdvectionVelocityRequest request{};
    if (const auto field =
            first_defined_parameter(eq_params, {"Level_set_field_name",
                                               "LevelSetFieldName",
                                               "Level_set_field",
                                               "LevelSetField",
                                               "Field_name"})) {
      request.level_set_field_name = trim_copy(*field);
    }
    if (source_field.has_value()) {
      request.source_velocity_field_name = trim_copy(*source_field);
    }
    if (const auto target =
            first_defined_parameter(eq_params, {"Velocity_field_name",
                                               "VelocityFieldName",
                                               "Advection_velocity_field",
                                               "AdvectionVelocityField"})) {
      request.target_velocity_field_name = trim_copy(*target);
    }
    if (request.target_velocity_field_name.empty()) {
      request.target_velocity_field_name = "LevelSetAdvectionVelocity";
    }
    if (request.target_velocity_field_name == request.source_velocity_field_name) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wet-extension level-set advection requires "
          "Velocity_field_name to be distinct from the physical source velocity field.");
    }

    const auto velocity_source =
        first_defined_parameter(eq_params, {"Velocity_source", "VelocitySource"});
    if (!velocity_source.has_value() ||
        normalized_token(*velocity_source) != "prescribeddata") {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wet-extension level-set advection requires "
          "<Velocity_source>prescribed_data</Velocity_source> so the transport equation uses "
          "the generated prescribed advection field.");
    }

    if (const auto extension_method =
            first_defined_parameter(eq_params, {"Wet_extension_advection_velocity_method",
                                               "WetExtensionAdvectionVelocityMethod",
                                               "Advection_velocity_extension_method",
                                               "AdvectionVelocityExtensionMethod"})) {
      const auto token = normalized_token(*extension_method);
      if (token != "nearestactivevertex" &&
          token != "nearestactive" &&
          token != "nearestvertex" &&
          token != "nearestinterfacepoint" &&
          token != "nearestinterfacevertex" &&
          token != "nearestinterface" &&
          token != "closestinterfacepoint") {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Unsupported wet-extension level-set "
            "advection velocity method '" + trim_copy(*extension_method) +
            "'. Implemented methods are nearest_active_vertex and nearest_interface_point.");
      }
      if (token == "nearestinterfacepoint" ||
          token == "nearestinterfacevertex" ||
          token == "nearestinterface" ||
          token == "closestinterfacepoint") {
        request.extension_method = "nearest_interface_point";
      } else {
        request.extension_method = "nearest_active_vertex";
      }
    }

    if (const auto isovalue =
            first_defined_double_parameter(eq_params, {"Level_set_isovalue",
                                                      "LevelSetIsovalue",
                                                      "Interface_isovalue",
                                                      "InterfaceIsovalue"})) {
      request.isovalue = *isovalue;
    }

    for (const auto& active_request : active_requests) {
      if (active_request.level_set_field_name != request.level_set_field_name) {
        continue;
      }
      request.isovalue = active_request.isovalue;
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

bool activeSideContains(double phi, const LevelSetAdvectionVelocityRequest& request)
{
  return request.active_side == LevelSetActiveSide::Negative
             ? phi <= request.isovalue
             : phi >= request.isovalue;
}

const char* activeSideName(LevelSetActiveSide side) noexcept
{
  return side == LevelSetActiveSide::Negative
             ? "LevelSetNegative"
             : "LevelSetPositive";
}

svmp::FE::geometry::CutIntegrationSide cutIntegrationSide(
    LevelSetActiveSide side) noexcept
{
  return side == LevelSetActiveSide::Negative
             ? svmp::FE::geometry::CutIntegrationSide::Negative
             : svmp::FE::geometry::CutIntegrationSide::Positive;
}

const char* fieldSourceKindName(
    svmp::FE::systems::FieldSourceKind kind) noexcept
{
  switch (kind) {
    case svmp::FE::systems::FieldSourceKind::Unknown:
      return "Unknown";
    case svmp::FE::systems::FieldSourceKind::PrescribedData:
      return "PrescribedData";
    case svmp::FE::systems::FieldSourceKind::DerivedFromUnknown:
      return "DerivedFromUnknown";
  }
  return "Unknown";
}

std::vector<const svmp::FE::geometry::CutQuadratureRule*>
retainedVolumeRulePointersForSide(
    const std::vector<svmp::FE::geometry::CutQuadratureRule>& rules,
    svmp::FE::geometry::CutIntegrationSide side)
{
  std::vector<const svmp::FE::geometry::CutQuadratureRule*> retained;
  if (side == svmp::FE::geometry::CutIntegrationSide::Interface) {
    return retained;
  }
  retained.reserve(rules.size());
  for (const auto& rule : rules) {
    if (rule.kind != svmp::FE::geometry::CutQuadratureKind::Volume ||
        rule.side != side ||
        svmp::FE::assembly::CutIntegrationContext::
            shouldPruneGeneratedVolumeRule(rule)) {
      continue;
    }
    retained.push_back(&rule);
  }
  return retained;
}

void validateEquationLevelCutVolumeConsumer(
    const svmp::FE::systems::FESystem& system,
    const ActiveCutVolumeRequest& request,
    int resolved_marker)
{
  if (request.origin != ActiveCutVolumeRequestOrigin::Equation) {
    return;
  }
  const auto side = cutIntegrationSide(request.active_side);
  if (system.cutVolumeKernelCount(resolved_marker, side) > 0u) {
    const auto phi_field = system.findFieldByName(request.level_set_field_name);
    if (phi_field != svmp::FE::INVALID_FIELD_ID) {
      const auto& phi_record = system.fieldRecord(phi_field);
      if (phi_record.source_kind !=
          svmp::FE::systems::FieldSourceKind::PrescribedData) {
        static std::set<std::string> warned;
        const std::string warning_key =
            request.equation_type + "|" + request.domain_id + "|" +
            request.level_set_field_name + "|" +
            std::to_string(resolved_marker) + "|" +
            activeSideName(request.active_side);
        if (warned.insert(warning_key).second) {
          application::core::oopCout()
              << "[svMultiPhysics::Application] WARNING equation-level "
              << "level-set cut-domain uses a moving level-set field with "
              << "only a first-order Hadamard cut-volume shape tangent; "
              << "full differentiated cut quadrature remains unavailable"
              << " equation_type='" << request.equation_type << "'"
              << " field='" << request.level_set_field_name << "'"
              << " field_source="
              << fieldSourceKindName(phi_record.source_kind)
              << " domain_id='" << request.domain_id << "'"
              << " marker=" << resolved_marker
              << " active_side=" << activeSideName(request.active_side)
              << " geometry_tangent_policy="
              << svmp::FE::level_set::geometryTangentPolicyName(
                     request.geometry_tangent_policy)
              << " diagnostic=equation_level_cut_domain_hadamard_shape_tangent"
              << std::endl;
        }
      }
    }
    return;
  }

  throw std::runtime_error(
      "[svMultiPhysics::Application] Equation-level level-set cut-domain "
      "request for equation_type='" + request.equation_type +
      "' field='" + request.level_set_field_name + "' domain_id='" +
      request.domain_id + "' resolved marker=" +
      std::to_string(resolved_marker) + " active_side=" +
      activeSideName(request.active_side) +
      " has no matching dCutVolume(...) form consumer. "
      "Add volume terms restricted with dCutVolume(marker, side), remove the "
      "equation-level cut-domain request, or keep this as an unfitted "
      "free-surface boundary request owned by Navier-Stokes.");
}

ActiveSideRegionSummary summarizeActiveSideRegions(
    const svmp::FE::interfaces::LevelSetInterfaceDomain& domain,
    LevelSetActiveSide active_side,
    std::size_t n_cells)
{
  ActiveSideRegionSummary summary;
  std::vector<double> wet_fraction(n_cells, 0.0);
  const auto side = cutIntegrationSide(active_side);
  for (const auto& region : domain.volumeRegions()) {
    if (!region.active() || region.side != side) {
      continue;
    }
    if (!region.full_cell_equivalent &&
        std::isfinite(region.volume_fraction) &&
        region.volume_fraction > svmp::FE::Real{0.0} &&
        region.volume_fraction <
            svmp::FE::assembly::CutIntegrationContext::
                minGeneratedCutVolumeFraction()) {
      ++summary.pruned_volume_regions;
      if (std::isfinite(region.measure) &&
          region.measure > svmp::FE::Real{0.0}) {
        summary.pruned_volume += region.measure;
      }
      continue;
    }
    ++summary.active_volume_regions;
    summary.active_volume += region.measure;
    summary.active_quadrature_points += region.quadrature_points.empty()
        ? 1u
        : region.quadrature_points.size();
    if (!std::isfinite(region.measure)) {
      ++summary.nonfinite_measure_regions;
    }
    if (region.measure < svmp::FE::Real{0.0}) {
      ++summary.negative_measure_regions;
    }
    if (region.quadrature_points.empty()) {
      ++summary.empty_quadrature_regions;
    }
    if (std::isfinite(region.volume_fraction)) {
      summary.min_volume_fraction =
          std::min(summary.min_volume_fraction, region.volume_fraction);
      summary.max_volume_fraction =
          std::max(summary.max_volume_fraction, region.volume_fraction);
    }
    const auto cell = region.parent_cell;
    if (cell >= 0 && static_cast<std::size_t>(cell) < wet_fraction.size()) {
      wet_fraction[static_cast<std::size_t>(cell)] = std::clamp(
          wet_fraction[static_cast<std::size_t>(cell)] +
              static_cast<double>(region.volume_fraction),
          0.0,
          1.0);
    }
  }

  constexpr double fraction_tol = 1.0e-12;
  for (const auto fraction : wet_fraction) {
    if (fraction <= fraction_tol) {
      ++summary.full_dry_cell_count;
    } else if (fraction >= 1.0 - fraction_tol) {
      ++summary.full_wet_cell_count;
      ++summary.active_wet_cells;
    } else {
      ++summary.cut_cell_count;
      ++summary.active_wet_cells;
    }
  }
  if (!std::isfinite(summary.min_volume_fraction)) {
    summary.min_volume_fraction = svmp::FE::Real{0.0};
  }
  if (!std::isfinite(summary.max_volume_fraction)) {
    summary.max_volume_fraction = svmp::FE::Real{0.0};
  }
  return summary;
}

CutAdjacentFacetScaleSummary summarizeCutAdjacentFacetScales(
    const svmp::FE::assembly::CutFacetSetHandle& handle)
{
  CutAdjacentFacetScaleSummary summary;
  summary.metadata_count = handle.facet_metadata.size();
  svmp::FE::Real sum = svmp::FE::Real{0.0};
  for (const auto& metadata : handle.facet_metadata) {
    const auto scale = metadata.stabilization_scale;
    if (!std::isfinite(scale)) {
      ++summary.nonfinite_scale_count;
      continue;
    }
    if (scale <= svmp::FE::Real{0.0}) {
      ++summary.zero_scale_count;
    }
    if (scale >=
        svmp::FE::assembly::CutIntegrationContext::maxCutCellStabilizationScale()) {
      ++summary.capped_scale_count;
    }
    summary.min_scale = std::min(summary.min_scale, scale);
    summary.max_scale = std::max(summary.max_scale, scale);
    sum += scale;
  }
  if (summary.metadata_count > 0u) {
    summary.mean_scale =
        sum / static_cast<svmp::FE::Real>(summary.metadata_count);
  }
  if (!std::isfinite(summary.min_scale)) {
    summary.min_scale = svmp::FE::Real{0.0};
  }
  if (!std::isfinite(summary.max_scale)) {
    summary.max_scale = svmp::FE::Real{0.0};
  }
  return summary;
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

std::vector<svmp::FE::systems::CutInteriorFacetAdjacency>
collectInteriorFacetAdjacencies(const svmp::FE::assembly::IMeshAccess& mesh)
{
  std::vector<svmp::FE::systems::CutInteriorFacetAdjacency> adjacencies;
  adjacencies.reserve(static_cast<std::size_t>(std::max<svmp::FE::GlobalIndex>(
      0, mesh.numInteriorFaces())));
  mesh.forEachInteriorFace(
      [&](svmp::FE::GlobalIndex face_id,
          svmp::FE::GlobalIndex first_cell,
          svmp::FE::GlobalIndex second_cell) {
        adjacencies.push_back(
            svmp::FE::systems::CutInteriorFacetAdjacency{
                .facet = static_cast<svmp::FE::MeshIndex>(face_id),
                .first_cell = static_cast<svmp::FE::MeshIndex>(first_cell),
                .second_cell = static_cast<svmp::FE::MeshIndex>(second_cell)});
      });
  return adjacencies;
}

std::vector<svmp::FE::MeshIndex> activeCutCellsForMarkerAndSide(
    const svmp::FE::assembly::CutIntegrationContext& context,
    int marker,
    LevelSetActiveSide active_side)
{
  std::vector<svmp::FE::MeshIndex> cells;
  const auto metadata =
      context.generatedVolumeMetadataForMarkerAndSide(
          marker, cutIntegrationSide(active_side));
  constexpr svmp::FE::Real full_fraction_tol = svmp::FE::Real{1.0e-12};
  for (const auto* entry : metadata) {
    if (entry == nullptr ||
        entry->parent_entity < static_cast<svmp::FE::MeshIndex>(0) ||
        !std::isfinite(entry->volume_fraction) ||
        entry->volume_fraction <= svmp::FE::Real{0.0} ||
        entry->volume_fraction >= svmp::FE::Real{1.0} - full_fraction_tol) {
      continue;
    }
    cells.push_back(entry->parent_entity);
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

std::vector<svmp::FE::MeshIndex> activeSupportCellsForMarkerAndSide(
    const svmp::FE::assembly::CutIntegrationContext& context,
    int marker,
    LevelSetActiveSide active_side)
{
  std::vector<svmp::FE::MeshIndex> cells;
  const auto metadata =
      context.generatedVolumeMetadataForMarkerAndSide(
          marker, cutIntegrationSide(active_side));
  for (const auto* entry : metadata) {
    if (entry == nullptr ||
        entry->parent_entity < static_cast<svmp::FE::MeshIndex>(0) ||
        !std::isfinite(entry->volume_fraction) ||
        entry->volume_fraction <= svmp::FE::Real{0.0}) {
      continue;
    }
    cells.push_back(entry->parent_entity);
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

std::vector<svmp::FE::systems::CutAdjacentInteriorFacet>
filterCutAdjacentFacetsToActiveSupport(
    const std::vector<svmp::FE::systems::CutAdjacentInteriorFacet>& facets,
    const std::vector<svmp::FE::MeshIndex>& active_support_cells)
{
  std::vector<svmp::FE::systems::CutAdjacentInteriorFacet> active_facets;
  active_facets.reserve(facets.size());
  const auto is_active = [&active_support_cells](svmp::FE::MeshIndex cell) {
    return std::binary_search(
        active_support_cells.begin(), active_support_cells.end(), cell);
  };
  for (const auto& facet : facets) {
    if (is_active(facet.first_cell) && is_active(facet.second_cell)) {
      active_facets.push_back(facet);
    }
  }
  return active_facets;
}

svmp::FE::assembly::CutFacetSetHandle addGeneratedCutAdjacentFacetSet(
    svmp::FE::assembly::CutIntegrationContext& context,
    const svmp::FE::interfaces::LevelSetInterfaceDomain& domain,
    const svmp::FE::assembly::IMeshAccess& mesh,
    LevelSetActiveSide active_side)
{
  const auto active_cut_cells =
      activeCutCellsForMarkerAndSide(context, domain.marker(), active_side);
  const auto& cut_cells =
      active_cut_cells.empty() ? domain.cutCells() : active_cut_cells;
  const auto adjacent_facets =
      svmp::FE::systems::identifyCutAdjacentInteriorFacets(
          cut_cells, collectInteriorFacetAdjacencies(mesh));
  const auto active_support_cells =
      activeSupportCellsForMarkerAndSide(context, domain.marker(), active_side);
  const auto active_adjacent_facets =
      filterCutAdjacentFacetsToActiveSupport(
          adjacent_facets, active_support_cells);
  const auto handle =
      svmp::FE::systems::makeCutAdjacentFacetSetHandle(
          domain.marker(),
          "generated-cut-adjacent-facets",
          active_adjacent_facets);

  svmp::FE::assembly::CutFacetSetHandle stored_handle;
  stored_handle.marker = handle.marker;
  stored_handle.name = handle.name;
  stored_handle.facets = handle.facets;
  stored_handle.facet_metadata.reserve(handle.facet_metadata.size());
  for (const auto& facet : handle.facet_metadata) {
    svmp::FE::assembly::CutFacetSetFacetMetadata metadata;
    metadata.facet = facet.facet;
    metadata.first_cell = facet.first_cell;
    metadata.second_cell = facet.second_cell;
    metadata.stabilization_scale = facet.stabilization_scale;
    metadata.stable_id = facet.stable_id;
    stored_handle.facet_metadata.push_back(metadata);
  }
  stored_handle.stable_id = handle.stable_id;
  context.bindFacetStabilizationScalesForMarkerAndSide(
      stored_handle,
      domain.marker(),
      cutIntegrationSide(active_side));
  return context.addFacetSetHandle(std::move(stored_handle));
}

constexpr std::uint64_t kCutContextHashOffset = 1469598103934665603ull;
constexpr std::uint64_t kCutContextHashPrime = 1099511628211ull;

void mixCutContextHash(std::uint64_t& h, std::uint64_t value) noexcept
{
  h ^= value;
  h *= kCutContextHashPrime;
}

void mixCutContextHash(std::uint64_t& h, const std::string& value) noexcept
{
  for (const char c : value) {
    mixCutContextHash(h, static_cast<unsigned char>(c));
  }
  mixCutContextHash(h, 0xffu);
}

std::uint64_t cutContextTopologyKey(
    const svmp::FE::interfaces::LevelSetInterfaceDomain& domain) noexcept
{
  std::uint64_t h = kCutContextHashOffset;
  mixCutContextHash(h, static_cast<std::uint64_t>(domain.marker()));
  mixCutContextHash(h, domain.request().quadrature_policy_key);
  for (const auto& fragment : domain.fragments()) {
    if (!fragment.active()) {
      continue;
    }
    mixCutContextHash(h, static_cast<std::uint64_t>(fragment.parent_cell));
    mixCutContextHash(h, static_cast<std::uint64_t>(fragment.kind));
    mixCutContextHash(h, static_cast<std::uint64_t>(fragment.degeneracy));
    mixCutContextHash(h, fragment.topology_id);
  }
  for (const auto& region : domain.volumeRegions()) {
    if (!region.active()) {
      continue;
    }
    mixCutContextHash(h, static_cast<std::uint64_t>(region.parent_cell));
    mixCutContextHash(h, static_cast<std::uint64_t>(region.side));
    mixCutContextHash(h, region.full_cell_equivalent ? 1u : 0u);
    mixCutContextHash(h, region.topology_id);
  }
  return h;
}

const char* stateSyncPointName(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point) noexcept
{
  using Point = svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  switch (point) {
  case Point::AcceptedNonlinearState:
    return "accepted";
  case Point::ResidualAssembly:
    return "residual";
  case Point::JacobianAssembly:
    return "jacobian";
  case Point::JacobianAndResidualAssembly:
    return "jacobian_and_residual";
  case Point::LineSearchTrialResidual:
    return "line_search_trial";
  case Point::RestoredNonlinearState:
    return "restored";
  case Point::FinalResidualAssembly:
    return "final_residual";
  }
  return "unknown";
}

std::string highOrderGeometryTangentPolicySummary(
    const std::vector<ActiveCutVolumeRequest>& requests)
{
  std::set<std::string> policies;
  for (const auto& request : requests) {
    if (request.geometry_mode !=
        svmp::FE::level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit) {
      continue;
    }
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

void logCutTopologyChange(
    const ActiveCutContextRefreshReport& report,
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point,
    std::optional<std::uint64_t>& previous_topology_key,
    const char* solve_kind)
{
  if (!report.refreshed) {
    return;
  }
  const bool changed =
      previous_topology_key.has_value() &&
      *previous_topology_key != report.topology_key;
  if (changed && application::core::oopTraceEnabled()) {
    application::core::oopCout()
        << "[svMultiPhysics::Application] Cut topology changed during "
        << solve_kind << " nonlinear solve sync_point="
        << stateSyncPointName(point)
        << " previous_topology_key=" << *previous_topology_key
        << " topology_key=" << report.topology_key
        << " active_cut_request_policy_key=" << report.request_policy_key
        << " cut_context_revision=" << report.value_revision
        << " cell_count=" << report.cell_count
        << " corner_linearized_cells=" << report.corner_linearized_cell_count
        << " interface_fragments=" << report.interface_fragments
        << " active_volume_regions=" << report.active_volume_regions
        << " active_cut_cells=" << report.active_cut_cells
        << " active_quadrature_points=" << report.active_quadrature_points
        << " domain_interface_quadrature_point_count="
        << report.domain_interface_quadrature_point_count
        << " domain_volume_quadrature_point_count="
        << report.domain_volume_quadrature_point_count
        << " domain_total_quadrature_point_count="
        << report.domain_total_quadrature_point_count
        << " backend_volume_quadrature_point_count="
        << report.backend_volume_quadrature_point_count
        << " backend_interface_quadrature_point_count="
        << report.backend_interface_quadrature_point_count
        << " backend_total_quadrature_point_count="
        << (report.backend_volume_quadrature_point_count +
            report.backend_interface_quadrature_point_count)
        << " backend_elapsed_seconds=" << report.backend_elapsed_seconds
        << " generated_cell_cache_hits="
        << report.generated_cell_cache_hits
        << " generated_cell_cache_misses="
        << report.generated_cell_cache_misses
        << " generated_domain_cache_hits="
        << report.generated_domain_cache_hits
        << " linear_full_cell_fast_path_cells="
        << report.linear_full_cell_fast_path_count
        << " process_vm_kb=" << report.process_vm_kb
        << " process_rss_kb=" << report.process_rss_kb
        << " basis_cache_entries=" << report.basis_cache_entries
        << " cut_adjacent_facets=" << report.cut_adjacent_facets
        << " negative_volume=" << report.negative_volume
        << " negative_reference_volume=" << report.negative_volume
        << " negative_physical_volume=" << report.negative_physical_volume
        << " positive_volume=" << report.positive_volume
        << " positive_reference_volume=" << report.positive_volume
        << " positive_physical_volume=" << report.positive_physical_volume
        << std::endl;
  }
  previous_topology_key = report.topology_key;
}

void logCornerLinearizedCutWarningOnce(
    const ActiveCutVolumeRequest& request,
    const svmp::FE::level_set::LevelSetGeneratedInterfaceResult& result)
{
  if (result.corner_linearized_cell_count == 0u) {
    return;
  }

  static std::set<std::string> warned_keys;
  const std::string key = request.level_set_field_name + "|" +
                          request.domain_id + "|" +
                          std::to_string(result.interface_marker);
  if (!warned_keys.insert(key).second) {
    return;
  }

  application::core::oopCout()
      << "[svMultiPhysics::Application] WARNING generated level-set interface "
      << "uses corner-linearized cut geometry"
      << " marker=" << result.interface_marker
      << " field='" << request.level_set_field_name << "'"
      << " domain_id='" << request.domain_id << "'"
      << " corner_linearized_cells="
      << result.corner_linearized_cell_count
      << " cell_count=" << result.cell_count
      << " max_cell_node_count=" << result.max_cell_node_count
      << " max_corner_node_count=" << result.max_corner_node_count
      << " diagnostic=high_order_level_set_cut_uses_corners"
      << std::endl;
}

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

    const auto side = cutIntegrationSide(request.active_side);
    const auto rules =
        cut_context->generatedVolumeRulesForMarkerAndSide(*marker, side);
    if (rules.empty()) {
      continue;
    }

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
    std::size_t n_cells)
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
    if (rules.empty()) {
      continue;
    }

    std::vector<double> wet_fraction(n_cells, 0.0);
    WetVolumeDiagnostic diagnostic;
    diagnostic.level_set_field_name = request.level_set_field_name;
    diagnostic.domain_id = request.domain_id;
    diagnostic.marker = *marker;
    diagnostic.active_side = request.active_side;
    diagnostic.isovalue = request.isovalue;
    const auto measure_summary =
        application::core::collectCutVolumeMeasures(mesh, rules);
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
        if (cell >= 0 && static_cast<std::size_t>(cell) < wet_fraction.size()) {
          wet_fraction[static_cast<std::size_t>(cell)] = std::clamp(
              wet_fraction[static_cast<std::size_t>(cell)] +
                  static_cast<double>(rule->volume_fraction),
              0.0,
              1.0);
        }
      }
    }
    constexpr double fraction_tol = 1.0e-12;
    for (const auto fraction : wet_fraction) {
      if (fraction <= fraction_tol) {
        ++diagnostic.full_dry_cell_count;
      } else if (fraction >= 1.0 - fraction_tol) {
        ++diagnostic.full_wet_cell_count;
      } else {
        ++diagnostic.cut_cell_count;
      }
    }
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
    application::core::oopCout()
        << "[svMultiPhysics::Application] WARNING active-domain cut context "
        << "is unavailable"
        << " step=" << step
        << " time=" << time
        << " requests=" << requests.size()
        << " diagnostic=missing_active_cut_context"
        << std::endl;
    return;
  }

  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    const auto marker = generatedVolumeMarkerForRequest(
        *cut_context, request, i);
    if (!marker.has_value()) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] WARNING active-domain cut context "
          << "has no generated marker for request"
          << " step=" << step
          << " time=" << time
          << " field='" << request.level_set_field_name << "'"
          << " domain_id='" << request.domain_id << "'"
          << " requested_marker=" << request.requested_interface_marker
          << " generated_marker_count="
          << cut_context->generatedVolumeMarkers().size()
          << " diagnostic=missing_generated_cut_marker"
          << std::endl;
      continue;
    }

    const auto side = cutIntegrationSide(request.active_side);
    const auto rules =
        cut_context->generatedVolumeRulesForMarkerAndSide(*marker, side);
    if (rules.empty()) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] WARNING active-domain cut context "
          << "has no retained volume rules"
          << " step=" << step
          << " time=" << time
          << " marker=" << *marker
          << " field='" << request.level_set_field_name << "'"
          << " domain_id='" << request.domain_id << "'"
          << " active_side=" << activeSideName(request.active_side)
          << " isovalue=" << request.isovalue
          << " pruned_volume_rules="
          << cut_context->generatedPrunedVolumeRuleCount()
          << " pruned_volume="
          << cut_context->generatedPrunedVolumeMeasure()
          << " diagnostic=empty_active_cut_volume_rules"
          << std::endl;
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
    application::core::oopCout()
        << "[svMultiPhysics::Application] WARNING ActiveFluid/WetVolumeFraction "
        << "disagreement"
        << " step=" << step
        << " time=" << time
        << " field='" << request.level_set_field_name << "'"
        << " domain_id='" << request.domain_id << "'"
        << " marker=" << *marker
        << " active_side=" << activeSideName(request.active_side)
        << " isovalue=" << request.isovalue
        << " compared_cut_cell_count=" << compared_cut_cell_count
        << " disagreeing_cut_cell_count=" << disagreeing_cut_cell_count
        << " threshold=" << strong_disagreement_threshold
        << " max_abs_difference=" << max_abs_difference
        << " max_difference_cell=" << max_difference_cell
        << std::endl;
  }
}

void logWetVolumeDiagnostics(
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::assembly::CutIntegrationContext* cut_context,
    const svmp::FE::assembly::IMeshAccess& mesh,
    std::size_t n_cells,
    int step,
    double time,
    std::map<std::string, svmp::FE::Real>& initial_wet_volume_by_key)
{
  logActiveCutVolumeAvailabilityWarnings(requests, cut_context, step, time);
  const auto diagnostics =
      collectWetVolumeDiagnostics(requests, cut_context, mesh, n_cells);
  const double drift_warning_threshold =
      parseDoubleEnv("SVMP_WET_VOLUME_DRIFT_WARNING", 1.0e-3);
  for (const auto& diagnostic : diagnostics) {
    const std::string key = diagnostic.level_set_field_name + "|" +
                            diagnostic.domain_id + "|" +
                            std::to_string(diagnostic.marker);
    const auto drift = application::core::computeWetVolumeDrift(
        key, diagnostic.wet_volume, initial_wet_volume_by_key);
    application::core::oopCout()
        << "[svMultiPhysics::Application] Wet volume diagnostic"
        << " step=" << step
        << " time=" << time
        << " field='" << diagnostic.level_set_field_name << "'"
        << " domain_id='" << diagnostic.domain_id << "'"
        << " marker=" << diagnostic.marker
        << " active_side=" << activeSideName(diagnostic.active_side)
        << " isovalue=" << diagnostic.isovalue
        << " wet_volume=" << diagnostic.wet_volume
        << " wet_volume_frame=" << diagnostic.wet_volume_frame
        << " reference_wet_volume=" << diagnostic.reference_wet_volume
        << " physical_wet_volume=" << diagnostic.physical_wet_volume
        << " initial_wet_volume=" << drift.initial_wet_volume
        << " wet_volume_drift=" << drift.wet_volume_drift
        << " relative_wet_volume_drift=" << drift.relative_wet_volume_drift
        << " volume_rule_count=" << diagnostic.volume_rule_count
        << " physical_volume_rule_count="
        << diagnostic.physical_volume_rule_count
        << " skipped_physical_volume_rule_count="
        << diagnostic.skipped_physical_volume_rule_count
        << " cut_cell_count=" << diagnostic.cut_cell_count
        << " full_wet_cell_count=" << diagnostic.full_wet_cell_count
        << " full_dry_cell_count=" << diagnostic.full_dry_cell_count
        << std::endl;
    if (drift_warning_threshold > 0.0 &&
        std::abs(static_cast<double>(drift.relative_wet_volume_drift)) >
            drift_warning_threshold) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] WARNING wet-volume drift exceeds "
          << "diagnostic threshold"
          << " step=" << step
          << " time=" << time
          << " field='" << diagnostic.level_set_field_name << "'"
          << " domain_id='" << diagnostic.domain_id << "'"
          << " marker=" << diagnostic.marker
          << " active_side=" << activeSideName(diagnostic.active_side)
          << " relative_wet_volume_drift="
          << drift.relative_wet_volume_drift
          << " threshold=" << drift_warning_threshold
          << " diagnostic=nonconservative_level_set_volume_drift"
          << std::endl;
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

std::vector<LevelSetMaintenanceRequest> levelSetMaintenanceRequests(const Parameters& params)
{
  std::vector<LevelSetMaintenanceRequest> requests;
  for (auto* eq : params.equation_parameters) {
    if (eq == nullptr || !eq->type.defined()) {
      continue;
    }
    const auto type = normalized_token(eq->type.value());
    if (type != "levelset" && type != "levelsettransport") {
      continue;
    }

    auto eq_params = eq->get_parameter_list();
    LevelSetMaintenanceRequest request{};
    if (const auto field =
            first_defined_parameter(eq_params, {"Level_set_field_name",
                                               "LevelSetFieldName",
                                               "Level_set_field",
                                               "LevelSetField",
                                               "Field_name"})) {
      request.level_set_field_name = trim_copy(*field);
    }
    if (const auto isovalue =
            first_defined_double_parameter(eq_params, {"Level_set_isovalue",
                                                      "LevelSetIsovalue",
                                                      "Interface_isovalue",
                                                      "InterfaceIsovalue"})) {
      request.isovalue = *isovalue;
    }
    if (const auto form =
            first_defined_parameter(eq_params, {"Transport_form",
                                               "TransportForm",
                                               "Advection_form",
                                               "AdvectionForm",
                                               "Level_set_transport_form",
                                               "LevelSetTransportForm"})) {
      request.transport_form = parseLevelSetTransportForm(*form);
    }

    if (const auto enabled =
            first_defined_bool_parameter(eq_params, {"Enable_reinitialization",
                                                    "Enable_level_set_reinitialization",
                                                    "Reinitialization",
                                                    "Reinitialization_enabled",
                                                    "Reinitialize_level_set"})) {
      request.reinitialization.enabled = *enabled;
    }
    if (const auto method =
            first_defined_parameter(eq_params, {"Reinitialization_method",
                                               "Level_set_reinitialization_method",
                                               "ReinitializationMethod"})) {
      request.reinitialization.method =
          parseLevelSetReinitializationMethod(*method);
    }
    if (const auto cadence =
            first_defined_int_parameter(eq_params, {"Reinitialization_cadence_steps",
                                                   "Reinitialization_cadence",
                                                   "Level_set_reinitialization_cadence_steps",
                                                   "ReinitializationCadenceSteps"})) {
      request.reinitialization.cadence_steps = *cadence;
    }
    if (const auto max_it =
            first_defined_int_parameter(eq_params, {"Reinitialization_max_iterations",
                                                   "Reinitialization_iterations",
                                                   "ReinitializationMaxIterations"})) {
      request.reinitialization.max_iterations = *max_it;
    }
    if (const auto scale =
            first_defined_double_parameter(eq_params, {"Reinitialization_pseudo_time_step_scale",
                                                      "ReinitializationPseudoTimeStepScale"})) {
      request.reinitialization.pseudo_time_step_scale =
          static_cast<svmp::FE::Real>(*scale);
    }
    if (const auto band =
            first_defined_double_parameter(eq_params, {"Reinitialization_interface_band_width",
                                                      "ReinitializationInterfaceBandWidth"})) {
      request.reinitialization.interface_band_width =
          static_cast<svmp::FE::Real>(*band);
    }
    if (const auto tol =
            first_defined_double_parameter(eq_params, {"Reinitialization_signed_distance_tolerance",
                                                      "ReinitializationSignedDistanceTolerance"})) {
      request.reinitialization.signed_distance_tolerance =
          static_cast<svmp::FE::Real>(*tol);
    }

    if (const auto enabled =
            first_defined_bool_parameter(eq_params, {"Enable_volume_correction",
                                                    "Enable_level_set_volume_correction",
                                                    "Volume_correction",
                                                    "VolumeCorrection",
                                                    "Correct_level_set_volume"})) {
      request.volume_correction.enabled = *enabled;
    }
    if (const auto cadence =
            first_defined_int_parameter(eq_params, {"Volume_correction_cadence_steps",
                                                   "Volume_correction_cadence",
                                                   "Level_set_volume_correction_cadence_steps",
                                                   "VolumeCorrectionCadenceSteps"})) {
      request.volume_correction.cadence_steps = *cadence;
    }
    if (const auto use_initial =
            first_defined_bool_parameter(eq_params, {"Volume_correction_use_initial_volume",
                                                    "Use_initial_level_set_volume_as_target",
                                                    "VolumeCorrectionUseInitialVolume"})) {
      request.volume_correction.use_initial_negative_volume_as_target =
          *use_initial;
    }
    if (const auto target =
            first_defined_double_parameter(eq_params, {"Volume_correction_target_negative_volume",
                                                      "Level_set_volume_correction_target_negative_volume",
                                                      "VolumeCorrectionTargetNegativeVolume"})) {
      request.volume_correction.target_negative_volume =
          static_cast<svmp::FE::Real>(*target);
      request.volume_correction.use_initial_negative_volume_as_target = false;
    }
    if (const auto tol =
            first_defined_double_parameter(eq_params, {"Volume_correction_tolerance",
                                                      "Volume_correction_volume_tolerance",
                                                      "Level_set_volume_correction_tolerance",
                                                      "VolumeCorrectionTolerance"})) {
      request.volume_correction.volume_tolerance =
          static_cast<svmp::FE::Real>(*tol);
    }
    if (const auto max_it =
            first_defined_int_parameter(eq_params, {"Volume_correction_max_iterations",
                                                   "VolumeCorrectionMaxIterations"})) {
      request.volume_correction.max_iterations = *max_it;
    }

    if (const auto enabled =
            first_defined_bool_parameter(
                eq_params,
                {"Enable_curvature_projection",
                 "Enable_projected_curvature",
                 "Project_level_set_curvature",
                 "Maintain_projected_curvature",
                 "Curvature_projection"})) {
      request.curvature_projection_enabled = *enabled;
    }
    if (const auto curvature_field =
            first_defined_parameter(
                eq_params,
                {"Curvature_field_name",
                 "CurvatureFieldName",
                 "Curvature_field",
                 "CurvatureField",
                 "Projected_curvature_field",
                 "ProjectedCurvatureField",
                 "Free_surface_curvature_field",
                 "FreeSurfaceCurvatureField"})) {
      request.curvature_field_name = trim_copy(*curvature_field);
      request.curvature_projection_enabled = true;
    }
    if (const auto cadence =
            first_defined_int_parameter(
                eq_params,
                {"Curvature_projection_cadence_steps",
                 "CurvatureProjectionCadenceSteps",
                 "Projected_curvature_cadence_steps",
                 "ProjectedCurvatureCadenceSteps"})) {
      request.curvature_projection_cadence_steps = *cadence;
    }
    if (const auto tol =
            first_defined_double_parameter(
                eq_params,
                {"Curvature_projection_gradient_tolerance",
                 "CurvatureProjectionGradientTolerance"})) {
      request.curvature_projection.gradient_tolerance =
          static_cast<svmp::FE::Real>(*tol);
    }
    if (const auto tol =
            first_defined_double_parameter(
                eq_params,
                {"Curvature_projection_normal_equation_tolerance",
                 "CurvatureProjectionNormalEquationTolerance"})) {
      request.curvature_projection.normal_equation_tolerance =
          static_cast<svmp::FE::Real>(*tol);
    }
    if (const auto residual =
            first_defined_double_parameter(
                eq_params,
                {"Curvature_projection_max_normalized_fit_residual",
                 "CurvatureProjectionMaxNormalizedFitResidual",
                 "Projected_curvature_max_normalized_fit_residual",
                 "ProjectedCurvatureMaxNormalizedFitResidual"})) {
      request.curvature_projection.max_normalized_fit_residual =
          static_cast<svmp::FE::Real>(*residual);
    }
    if (const auto rings =
            first_defined_int_parameter(
                eq_params,
                {"Curvature_projection_neighbor_rings",
                 "CurvatureProjectionNeighborRings"})) {
      request.curvature_projection.max_neighbor_rings = *rings;
    }
    if (const auto iterations =
            first_defined_int_parameter(
                eq_params,
                {"Curvature_projection_smoothing_iterations",
                 "CurvatureProjectionSmoothingIterations",
                 "Projected_curvature_smoothing_iterations",
                 "ProjectedCurvatureSmoothingIterations"})) {
      request.curvature_projection.smoothing_iterations = *iterations;
    }
    if (const auto relaxation =
            first_defined_double_parameter(
                eq_params,
                {"Curvature_projection_smoothing_relaxation",
                 "CurvatureProjectionSmoothingRelaxation",
                 "Projected_curvature_smoothing_relaxation",
                 "ProjectedCurvatureSmoothingRelaxation"})) {
      request.curvature_projection.smoothing_relaxation =
          static_cast<svmp::FE::Real>(*relaxation);
    }
    request.curvature_projection.isovalue =
        static_cast<svmp::FE::Real>(request.isovalue);

    if (request.reinitialization.enabled ||
        request.volume_correction.enabled ||
        request.curvature_projection_enabled) {
      requests.push_back(std::move(request));
    }
  }
  return requests;
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
    if (request.reinitialization.enabled || request.volume_correction.enabled) {
      transport_maintained_fields.insert(request.level_set_field_name);
    }
    application::core::oopCout()
        << "[svMultiPhysics::Application] Level-set maintenance diagnostic"
        << " field='" << request.level_set_field_name << "'"
        << " reinitialization="
        << (request.reinitialization.enabled ? "enabled" : "disabled")
        << " volume_correction="
        << (request.volume_correction.enabled ? "enabled" : "disabled")
        << " curvature_projection="
        << (request.curvature_projection_enabled ? "enabled" : "disabled")
        << " curvature_field='"
        << request.curvature_field_name << "'"
        << " conservation_diagnostic="
        << svmp::FE::level_set::levelSetConservationDiagnosticName(
               svmp::FE::level_set::levelSetConservationDiagnostic(
                   request.transport_form,
                   request.reinitialization,
                   request.volume_correction))
        << " reinitialization_cadence="
        << request.reinitialization.cadence_steps
        << " volume_correction_cadence="
        << request.volume_correction.cadence_steps
        << " curvature_projection_cadence="
        << request.curvature_projection_cadence_steps
        << std::endl;
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
    application::core::oopCout()
        << "[svMultiPhysics::Application] WARNING unfitted free-surface "
        << "level-set has no enabled reinitialization or volume-correction request"
        << " field='" << request.level_set_field_name << "'"
        << " domain_id='" << request.domain_id << "'"
        << " active_side=" << activeSideName(request.active_side)
        << " diagnostic=plain_level_set_advection_not_conservative"
        << std::endl;
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

void scatterFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution,
    std::span<const svmp::FE::Real> values)
{
  if (static_cast<std::size_t>(solution.size()) != values.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Cannot scatter FE-ordered solution values with mismatched size.");
  }

  auto view = solution.createAssemblyView();
  if (!view) {
    auto span = solution.localSpan();
    if (span.size() != values.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Could not scatter FE-ordered solution values.");
    }
    std::copy(values.begin(), values.end(), span.begin());
    solution.updateGhosts();
    return;
  }

  constexpr svmp::FE::GlobalIndex chunk_size = 4096;
  std::vector<svmp::FE::GlobalIndex> dofs;
  dofs.reserve(static_cast<std::size_t>(std::min(solution.size(), chunk_size)));
  view->beginAssemblyPhase();
  for (svmp::FE::GlobalIndex offset = 0; offset < solution.size();
       offset += chunk_size) {
    const auto chunk =
        std::min<svmp::FE::GlobalIndex>(chunk_size, solution.size() - offset);
    dofs.resize(static_cast<std::size_t>(chunk));
    for (svmp::FE::GlobalIndex i = 0; i < chunk; ++i) {
      dofs[static_cast<std::size_t>(i)] = offset + i;
    }
    view->setVectorEntries(
        std::span<const svmp::FE::GlobalIndex>(dofs.data(), dofs.size()),
        values.subspan(static_cast<std::size_t>(offset),
                       static_cast<std::size_t>(chunk)));
  }
  view->endAssemblyPhase();
  view->finalizeAssembly();
  solution.updateGhosts();
}

double maxAbsDifference(std::span<const svmp::FE::Real> left,
                        std::span<const svmp::FE::Real> right)
{
  if (left.size() != right.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Cannot compare solution history vectors with mismatched sizes.");
  }

  double local = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    local = std::max(
        local,
        static_cast<double>(std::abs(left[i] - right[i])));
  }

  return local;
}

double globalMaxAbsDifference(std::span<const svmp::FE::Real> left,
                              std::span<const svmp::FE::Real> right,
                              const svmp::MeshComm& comm)
{
  const double local = maxAbsDifference(left, right);

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm.native());
    return global;
  }
#else
  (void)comm;
#endif

  return local;
}

double globalSumDouble(double local, const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm.native());
    return global;
  }
#else
  (void)comm;
#endif

  return local;
}

double globalMinDouble(double local, const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MIN, comm.native());
    return global;
  }
#else
  (void)comm;
#endif

  return local;
}

double globalMaxDouble(double local, const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm.native());
    return global;
  }
#else
  (void)comm;
#endif

  return local;
}

std::size_t globalSumSize(std::size_t local, const svmp::MeshComm& comm)
{
  auto local_count = static_cast<long long>(local);
#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    long long global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, comm.native());
    return static_cast<std::size_t>(std::max<long long>(0, global_count));
  }
#else
  (void)comm;
#endif

  return static_cast<std::size_t>(std::max<long long>(0, local_count));
}

bool globalAnyBool(bool local, const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    const int local_value = local ? 1 : 0;
    int global_value = 0;
    MPI_Allreduce(
        &local_value, &global_value, 1, MPI_INT, MPI_MAX, comm.native());
    return global_value != 0;
  }
#else
  (void)comm;
#endif

  return local;
}

std::string formatImplicitCutBackendCounts(
    const std::array<std::size_t, 5>& counts)
{
  using Backend = svmp::FE::level_set::ImplicitCutQuadratureBackend;
  constexpr std::array<Backend, 5> backends{
      Backend::LinearCorner,
      Backend::SayeHyperrectangle,
      Backend::HighOrderSubcell,
      Backend::MomentFit,
      Backend::Auto};

  std::ostringstream oss;
  bool first = true;
  for (std::size_t i = 0; i < backends.size(); ++i) {
    if (counts[i] == 0u) {
      continue;
    }
    if (!first) {
      oss << ",";
    }
    first = false;
    oss << svmp::FE::level_set::implicitCutQuadratureBackendName(backends[i])
        << ":" << counts[i];
  }
  return first ? std::string{"none"} : oss.str();
}

std::size_t implicitCutQualificationIndex(
    svmp::FE::level_set::ImplicitCutQuadratureBackendQualification qualification) noexcept
{
  using Qualification =
      svmp::FE::level_set::ImplicitCutQuadratureBackendQualification;
  switch (qualification) {
  case Qualification::Unavailable:
    return 0u;
  case Qualification::Experimental:
    return 1u;
  case Qualification::ProductionQualified:
    return 2u;
  }
  return 0u;
}

std::string formatImplicitCutBackendQualificationCounts(
    const std::array<std::size_t, 3>& counts)
{
  using Qualification =
      svmp::FE::level_set::ImplicitCutQuadratureBackendQualification;
  constexpr std::array<Qualification, 3> qualifications{
      Qualification::Unavailable,
      Qualification::Experimental,
      Qualification::ProductionQualified};

  std::ostringstream oss;
  bool first = true;
  for (std::size_t i = 0; i < qualifications.size(); ++i) {
    if (counts[i] == 0u) {
      continue;
    }
    if (!first) {
      oss << ",";
    }
    first = false;
    oss << svmp::FE::level_set::implicitCutQuadratureBackendQualificationName(
               qualifications[i])
        << ":" << counts[i];
  }
  return first ? std::string{"none"} : oss.str();
}

svmp::FE::level_set::ImplicitCutQuadratureBackend
selectedImplicitCutBackendForCell(
    svmp::FE::level_set::ImplicitCutQuadratureBackend requested_backend,
    svmp::FE::level_set::ImplicitCutFallbackPolicy fallback_policy,
    int mesh_dimension,
    svmp::FE::ElementType element_type) noexcept
{
  using Backend = svmp::FE::level_set::ImplicitCutQuadratureBackend;
  using Fallback = svmp::FE::level_set::ImplicitCutFallbackPolicy;

  if (requested_backend == Backend::Auto) {
    if (svmp::FE::level_set::implicitCutQuadratureBackendCapability(
            Backend::SayeHyperrectangle,
            mesh_dimension,
            element_type)
            .supports_element_type) {
      return Backend::SayeHyperrectangle;
    }
    if (svmp::FE::level_set::implicitCutQuadratureBackendCapability(
            Backend::HighOrderSubcell,
            mesh_dimension,
            element_type)
            .supports_element_type) {
      return Backend::HighOrderSubcell;
    }
    return Backend::Auto;
  }

  if (svmp::FE::level_set::implicitCutQuadratureBackendCapability(
          requested_backend,
          mesh_dimension,
          element_type)
          .supports_element_type) {
    return requested_backend;
  }

  if (fallback_policy == Fallback::LinearCorner &&
      svmp::FE::level_set::implicitCutQuadratureBackendCapability(
          Backend::LinearCorner,
          mesh_dimension,
          element_type)
          .supports_element_type) {
    return Backend::LinearCorner;
  }

  return requested_backend;
}

std::array<std::size_t, 3> localImplicitCutBackendQualificationCounts(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::level_set::ImplicitCutQuadratureBackend requested_backend,
    svmp::FE::level_set::ImplicitCutFallbackPolicy fallback_policy)
{
  std::array<std::size_t, 3> counts{};
  mesh.forEachCell([&](svmp::FE::GlobalIndex cell_id) {
    const auto element_type = mesh.getCellType(cell_id);
    const auto selected_backend =
        selectedImplicitCutBackendForCell(
            requested_backend,
            fallback_policy,
            mesh.dimension(),
            element_type);
    const auto capability =
        svmp::FE::level_set::implicitCutQuadratureBackendCapability(
            selected_backend,
            mesh.dimension(),
            element_type);
    ++counts[implicitCutQualificationIndex(capability.qualification)];
  });
  return counts;
}

void initializeLevelSetMaintenanceTargets(
    application::core::SimulationComponents& sim,
    std::vector<LevelSetMaintenanceRequest>& requests)
{
  if (!sim.fe_system || !sim.time_history || requests.empty()) {
    return;
  }

  const auto fe_solution = gatherFeOrderedSolution(sim.time_history->u());
  for (auto& request : requests) {
    if (!request.volume_correction.enabled ||
        request.volume_target_initialized) {
      continue;
    }
    if (!request.volume_correction.use_initial_negative_volume_as_target) {
      request.volume_target =
          request.volume_correction.target_negative_volume;
      request.volume_target_initialized = true;
      continue;
    }

    const auto field =
        sim.fe_system->findFieldByName(request.level_set_field_name);
    if (field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set volume correction could not find field '" +
          request.level_set_field_name + "'.");
    }

    svmp::FE::level_set::LevelSetVolumeOptions volume_options{};
    volume_options.isovalue =
        static_cast<svmp::FE::Real>(request.isovalue);
    auto volume = svmp::FE::level_set::computeLevelSetCutCellVolume(
        *sim.fe_system,
        field,
        volume_options,
        fe_solution);
    if (!volume.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Initial level-set volume calculation failed for field '" +
          request.level_set_field_name + "': " + volume.diagnostic);
    }
    request.volume_target = volume.negative_volume;
    request.volume_target_initialized = true;
    application::core::oopCout()
        << "[svMultiPhysics::Application] Level-set volume target field='"
        << request.level_set_field_name << "' negative_volume="
        << request.volume_target << std::endl;
  }
}

bool applyLevelSetMaintenance(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    std::vector<LevelSetMaintenanceRequest>& requests)
{
  if (!sim.fe_system || requests.empty()) {
    return false;
  }

  bool changed = false;
  auto fe_solution = gatherFeOrderedSolution(history.u());
  std::set<svmp::FE::FieldId> modified_level_set_fields;
  for (auto& request : requests) {
    const int completed_step = history.stepIndex();
    const bool do_reinit =
        svmp::FE::level_set::shouldReinitializeLevelSet(
            request.reinitialization,
            completed_step);
    const bool do_volume =
        svmp::FE::level_set::shouldApplyLevelSetVolumeCorrection(
            request.volume_correction,
            completed_step);
    if (!do_reinit && !do_volume) {
      continue;
    }

    const auto field =
        sim.fe_system->findFieldByName(request.level_set_field_name);
    if (field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set maintenance could not find field '" +
          request.level_set_field_name + "'.");
    }

    if (do_reinit) {
      if (request.reinitialization.method !=
          svmp::FE::level_set::LevelSetReinitializationMethod::Projection) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Runtime level-set reinitialization currently supports Projection only.");
      }
      std::vector<svmp::FE::Real> repaired;
      auto result =
          svmp::FE::level_set::repairLevelSetSignedDistanceByProjection(
              *sim.fe_system,
              field,
              request.reinitialization,
              fe_solution,
              repaired);
      if (!result.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set reinitialization failed for field '" +
            request.level_set_field_name + "': " + result.diagnostic);
      }
      fe_solution = std::move(repaired);
      changed = true;
      modified_level_set_fields.insert(field);
      application::core::oopCout()
          << "[svMultiPhysics::Application] Level-set reinitialized field='"
          << request.level_set_field_name << "' step=" << completed_step
          << " repaired_dofs=" << result.repaired_dofs
          << " interface_fragments=" << result.interface_fragments
          << " interface_displacement_samples="
          << result.interface_displacement_samples
          << " max_interface_displacement="
          << result.max_interface_displacement
          << " l2_interface_displacement="
          << result.l2_interface_displacement
          << " max_abs_update=" << result.max_abs_update << std::endl;
    }

    if (do_volume) {
      if (!request.volume_target_initialized) {
        request.volume_target =
            request.volume_correction.target_negative_volume;
        request.volume_target_initialized = true;
      }
      svmp::FE::level_set::LevelSetVolumeOptions volume_options{};
      volume_options.isovalue =
          static_cast<svmp::FE::Real>(request.isovalue);
      svmp::FE::level_set::LevelSetGlobalShiftCorrectionOptions correction_options{};
      correction_options.target_negative_volume = request.volume_target;
      correction_options.volume_tolerance =
          request.volume_correction.volume_tolerance;
      correction_options.max_iterations =
          request.volume_correction.max_iterations;

      std::vector<svmp::FE::Real> corrected;
      auto result =
          svmp::FE::level_set::applyGlobalLevelSetShiftCorrection(
              *sim.fe_system,
              field,
              volume_options,
              correction_options,
              fe_solution,
              corrected);
      if (!result.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set volume correction failed for field '" +
            request.level_set_field_name + "': " + result.diagnostic);
      }
      fe_solution = std::move(corrected);
      changed = true;
      modified_level_set_fields.insert(field);
      application::core::oopCout()
          << "[svMultiPhysics::Application] Level-set volume corrected field='"
          << request.level_set_field_name << "' step=" << completed_step
          << " target_negative_volume=" << result.target_negative_volume
          << " initial_negative_volume=" << result.initial_negative_volume
          << " initial_volume_error="
          << (result.initial_negative_volume - result.target_negative_volume)
          << " corrected_negative_volume="
          << result.corrected_negative_volume
          << " achieved_volume_error=" << result.volume_error
          << " applied_shift=" << result.applied_shift
          << " applied_shift_magnitude=" << std::abs(result.applied_shift)
          << " iterations=" << result.iterations << std::endl;
    }
  }

  if (changed) {
    std::vector<std::vector<svmp::FE::Real>> older_history_before;
    older_history_before.reserve(
        static_cast<std::size_t>(std::max(0, history.historyDepth() - 1)));
    for (int k = 2; k <= history.historyDepth(); ++k) {
      older_history_before.push_back(gatherFeOrderedSolution(history.uPrevK(k)));
    }

    std::size_t synchronized_dofs = 0u;
    auto current_solution = gatherFeOrderedSolution(history.u());
    for (const auto field : modified_level_set_fields) {
      synchronized_dofs += application::core::copyFieldDofsIntoFeOrderedSolution(
          *sim.fe_system, field, fe_solution, current_solution);
    }
    scatterFeOrderedSolution(history.u(), current_solution);

    auto previous_solution = gatherFeOrderedSolution(history.uPrev());
    for (const auto field : modified_level_set_fields) {
      (void)application::core::copyFieldDofsIntoFeOrderedSolution(
          *sim.fe_system, field, fe_solution, previous_solution);
    }
    scatterFeOrderedSolution(history.uPrev(), previous_solution);
    history.updateGhosts();
    const auto current_previous_delta = globalMaxAbsDifference(
        history.uSpan(), history.uPrevSpan(), svmp::MeshComm::world());
    double older_history_delta = 0.0;
    for (int k = 2; k <= history.historyDepth(); ++k) {
      const auto after = gatherFeOrderedSolution(history.uPrevK(k));
      older_history_delta = std::max(
          older_history_delta,
          maxAbsDifference(
              older_history_before[static_cast<std::size_t>(k - 2)],
              after));
    }
    application::core::oopCout()
        << "[svMultiPhysics::Application] Level-set maintenance synchronized"
        << " step=" << history.stepIndex()
        << " accepted_solution=modified_level_set_fields"
        << " previous_state=modified_level_set_fields"
        << " synchronized_fields=" << modified_level_set_fields.size()
        << " synchronized_dofs=" << synchronized_dofs
        << " current_previous_max_abs_delta=" << current_previous_delta
        << " older_history_max_abs_delta=" << older_history_delta
        << " history_depth=" << history.historyDepth() << std::endl;
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

std::vector<double> evaluateVertexField(
    const svmp::FE::systems::FESystem& system,
    const svmp::Mesh& mesh,
    svmp::FE::FieldId field,
    const svmp::FE::systems::SystemStateView& state,
    std::size_t components)
{
  const auto n_vertices = mesh.n_vertices();
  std::vector<double> values(n_vertices * components, 0.0);
  if (system.evaluateFieldAtVertices(
          field,
          state,
          static_cast<svmp::FE::GlobalIndex>(n_vertices),
          std::span<double>(values.data(), values.size()))) {
    return values;
  }

  const int mesh_dim = mesh.dim();
  const auto& coords = mesh.X_ref();
  for (std::size_t v = 0; v < n_vertices; ++v) {
    std::array<svmp::FE::Real, 3> point{0.0, 0.0, 0.0};
    for (int d = 0; d < mesh_dim; ++d) {
      point[static_cast<std::size_t>(d)] =
          static_cast<svmp::FE::Real>(
              coords[v * static_cast<std::size_t>(mesh_dim) +
                     static_cast<std::size_t>(d)]);
    }
    const auto value = system.evaluateFieldAtPoint(field, state, point);
    if (!value) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Could not evaluate field at mesh vertex while updating level-set advection velocity.");
    }
    for (std::size_t c = 0; c < components; ++c) {
      values[v * components + c] = static_cast<double>((*value)[c]);
    }
  }
  return values;
}

std::optional<svmp::FE::spaces::FunctionSpace::Value>
referenceCellSamplePoint(svmp::FE::ElementType type)
{
  try {
    const auto nodes = svmp::FE::basis::NodeOrdering::num_nodes(type);
    if (nodes == 0u) {
      return std::nullopt;
    }
    svmp::FE::spaces::FunctionSpace::Value point{};
    for (std::size_t i = 0; i < nodes; ++i) {
      const auto node =
          svmp::FE::basis::NodeOrdering::get_node_coords(type, i);
      point[0] += node[0];
      point[1] += node[1];
      point[2] += node[2];
    }
    const auto inv = svmp::FE::Real{1.0} / static_cast<svmp::FE::Real>(nodes);
    point[0] *= inv;
    point[1] *= inv;
    point[2] *= inv;
    return point;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::array<svmp::FE::Real, 3>> physicalCellSamplePoint(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell)
{
  std::vector<std::array<svmp::FE::Real, 3>> coords;
  mesh.getCellCoordinates(cell, coords);
  if (coords.empty()) {
    return std::nullopt;
  }
  std::array<svmp::FE::Real, 3> point{0.0, 0.0, 0.0};
  for (const auto& coord : coords) {
    point[0] += coord[0];
    point[1] += coord[1];
    point[2] += coord[2];
  }
  const auto inv =
      svmp::FE::Real{1.0} / static_cast<svmp::FE::Real>(coords.size());
  point[0] *= inv;
  point[1] *= inv;
  point[2] *= inv;
  return point;
}

std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample>
collectLevelSetCurvatureSupplementalSamples(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId field)
{
  std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample> samples;
  const auto& rec = system.fieldRecord(field);
  if (!rec.space || rec.components != 1) {
    return samples;
  }

  const auto& mesh = system.meshAccess();
  const auto& field_dofs = system.fieldDofHandler(field);
  const auto offset = system.fieldDofOffset(field);
  const bool use_prescribed =
      rec.source_kind == svmp::FE::systems::FieldSourceKind::PrescribedData;
  const auto prescribed_coefficients =
      use_prescribed ? system.prescribedFieldCoefficients(field)
                     : std::span<const svmp::FE::Real>{};

  std::unique_ptr<svmp::FE::assembly::GlobalSystemView> solution_view;
  if (!use_prescribed && state.u_vector != nullptr) {
    auto* vec = const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
    solution_view = vec->createAssemblyView();
  }

  std::vector<svmp::FE::Real> coefficients;
  mesh.forEachCell([&](svmp::FE::GlobalIndex cell) {
    if (rec.space->polynomial_order(cell) <= 1) {
      return;
    }
    const auto reference_point =
        referenceCellSamplePoint(mesh.getCellType(cell));
    const auto physical_point = physicalCellSamplePoint(mesh, cell);
    if (!reference_point.has_value() || !physical_point.has_value()) {
      return;
    }
    const auto cell_dofs = field_dofs.getCellDofs(cell);
    const auto expected = rec.space->dofs_per_element(cell);
    if (cell_dofs.size() != expected) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set curvature projection found "
          "a high-order level-set cell with incompatible DOF count.");
    }

    coefficients.clear();
    coefficients.reserve(cell_dofs.size());
    for (const auto local_dof : cell_dofs) {
      const auto dof = use_prescribed ? local_dof : local_dof + offset;
      if (dof < 0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set curvature projection found "
            "a negative level-set DOF.");
      }
      if (use_prescribed) {
        const auto idx = static_cast<std::size_t>(dof);
        if (idx >= prescribed_coefficients.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set curvature projection "
              "found prescribed level-set coefficients that are too small.");
        }
        coefficients.push_back(prescribed_coefficients[idx]);
      } else if (solution_view) {
        coefficients.push_back(solution_view->getVectorEntry(dof));
      } else {
        const auto idx = static_cast<std::size_t>(dof);
        if (idx >= state.u.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set curvature projection "
              "found a level-set DOF outside the current state vector.");
        }
        coefficients.push_back(state.u[idx]);
      }
    }

    const auto value =
        rec.space->evaluate_scalar(*reference_point, coefficients);
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set curvature projection "
          "evaluated a non-finite high-order level-set sample.");
    }
    samples.push_back(
        svmp::FE::level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = static_cast<svmp::FE::MeshIndex>(cell),
            .coordinate = *physical_point,
            .value = value});
  });

  return samples;
}

bool shouldProjectLevelSetCurvature(const LevelSetMaintenanceRequest& request,
                                    int step)
{
  if (!request.curvature_projection_enabled) {
    return false;
  }
  if (request.curvature_field_name.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set curvature projection requires Curvature_field_name.");
  }
  if (request.curvature_projection_cadence_steps <= 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Curvature_projection_cadence_steps must be positive.");
  }
  return step <= 0 || step % request.curvature_projection_cadence_steps == 0;
}

void mixCurvatureSignature(std::uint64_t& seed,
                           std::uint64_t value) noexcept
{
  seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

std::uint64_t realBitsForSignature(svmp::FE::Real value) noexcept
{
  std::uint64_t bits = 0u;
  static_assert(sizeof(value) <= sizeof(bits),
                "curvature projection signatures expect Real <= 64 bits");
  std::memcpy(&bits, &value, sizeof(value));
  return bits;
}

void mixCurvatureSignatureString(std::uint64_t& seed,
                                 std::string_view value) noexcept
{
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(value.size()));
  for (const unsigned char c : value) {
    mixCurvatureSignature(seed, static_cast<std::uint64_t>(c));
  }
}

void mixCurvatureSignatureReal(std::uint64_t& seed,
                               svmp::FE::Real value) noexcept
{
  mixCurvatureSignature(seed, realBitsForSignature(value));
}

std::string curvatureProjectionCacheKey(
    const LevelSetMaintenanceRequest& request)
{
  return request.level_set_field_name + '\n' + request.curvature_field_name;
}

std::uint64_t curvatureProjectionInputSignature(
    const LevelSetMaintenanceRequest& request,
    const svmp::FE::level_set::LevelSetCurvatureProjectionOptions& options,
    std::span<const svmp::FE::Real> phi,
    std::span<const svmp::FE::level_set::LevelSetCurvatureProjectionSample>
        supplemental_samples,
    svmp::FE::GlobalIndex mesh_vertices,
    svmp::FE::GlobalIndex mesh_cells,
    int mesh_dimension,
    bool mesh_revision_tracking_available,
    std::uint64_t mesh_geometry_revision,
    std::uint64_t mesh_topology_revision,
    std::uint64_t mesh_ownership_revision,
    std::uint64_t mesh_numbering_revision,
    std::uint64_t mesh_coordinate_configuration_key)
{
  std::uint64_t seed = 0xcbf29ce484222325ull;
  mixCurvatureSignatureString(seed, request.level_set_field_name);
  mixCurvatureSignatureString(seed, request.curvature_field_name);
  mixCurvatureSignatureReal(seed,
                            static_cast<svmp::FE::Real>(request.isovalue));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            request.curvature_projection_cadence_steps));
  mixCurvatureSignatureReal(seed, options.isovalue);
  mixCurvatureSignatureReal(seed, options.gradient_tolerance);
  mixCurvatureSignatureReal(seed, options.normal_equation_tolerance);
  mixCurvatureSignatureReal(seed, options.max_normalized_fit_residual);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.max_neighbor_rings));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.smoothing_iterations));
  mixCurvatureSignatureReal(seed, options.smoothing_relaxation);
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(mesh_vertices));
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(mesh_cells));
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(mesh_dimension));
  mixCurvatureSignature(seed, mesh_revision_tracking_available ? 1u : 0u);
  mixCurvatureSignature(seed, mesh_geometry_revision);
  mixCurvatureSignature(seed, mesh_topology_revision);
  mixCurvatureSignature(seed, mesh_ownership_revision);
  mixCurvatureSignature(seed, mesh_numbering_revision);
  mixCurvatureSignature(seed, mesh_coordinate_configuration_key);
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(phi.size()));
  for (const auto value : phi) {
    mixCurvatureSignatureReal(seed, value);
  }
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            supplemental_samples.size()));
  for (const auto& sample : supplemental_samples) {
    mixCurvatureSignature(seed,
                          static_cast<std::uint64_t>(sample.parent_cell));
    for (const auto coordinate : sample.coordinate) {
      mixCurvatureSignatureReal(seed, coordinate);
    }
    mixCurvatureSignatureReal(seed, sample.value);
  }
  return seed;
}

std::optional<std::uint64_t> curvatureProjectionFastInputSignature(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId phi_field,
    const LevelSetMaintenanceRequest& request,
    const svmp::FE::level_set::LevelSetCurvatureProjectionOptions& options,
    svmp::FE::GlobalIndex mesh_vertices,
    svmp::FE::GlobalIndex mesh_cells,
    int mesh_dimension,
    bool mesh_revision_tracking_available,
    std::uint64_t mesh_geometry_revision,
    std::uint64_t mesh_topology_revision,
    std::uint64_t mesh_ownership_revision,
    std::uint64_t mesh_numbering_revision,
    std::uint64_t mesh_field_layout_revision,
    std::uint64_t mesh_label_revision,
    std::uint64_t mesh_coordinate_configuration_key)
{
  if (!mesh_revision_tracking_available) {
    return std::nullopt;
  }

  const auto& rec = system.fieldRecord(phi_field);
  std::uint64_t source_kind = 0u;
  std::uint64_t source_revision = 0u;
  if (rec.source_kind == svmp::FE::systems::FieldSourceKind::PrescribedData) {
    source_kind = 1u;
    source_revision = system.prescribedFieldRevision(phi_field);
  } else if (state.u_vector != nullptr &&
             system.fieldParticipatesInUnknownVector(phi_field)) {
    source_kind = 2u;
    source_revision = state.u_vector->valueRevision();
  } else {
    return std::nullopt;
  }

  std::uint64_t seed = 0x6a09e667f3bcc909ull;
  mixCurvatureSignatureString(seed, request.level_set_field_name);
  mixCurvatureSignatureString(seed, request.curvature_field_name);
  mixCurvatureSignatureReal(seed,
                            static_cast<svmp::FE::Real>(request.isovalue));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            request.curvature_projection_cadence_steps));
  mixCurvatureSignatureReal(seed, options.isovalue);
  mixCurvatureSignatureReal(seed, options.gradient_tolerance);
  mixCurvatureSignatureReal(seed, options.normal_equation_tolerance);
  mixCurvatureSignatureReal(seed, options.max_normalized_fit_residual);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.max_neighbor_rings));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.smoothing_iterations));
  mixCurvatureSignatureReal(seed, options.smoothing_relaxation);
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(mesh_vertices));
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(mesh_cells));
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(mesh_dimension));
  mixCurvatureSignature(seed, mesh_geometry_revision);
  mixCurvatureSignature(seed, mesh_topology_revision);
  mixCurvatureSignature(seed, mesh_ownership_revision);
  mixCurvatureSignature(seed, mesh_numbering_revision);
  mixCurvatureSignature(seed, mesh_field_layout_revision);
  mixCurvatureSignature(seed, mesh_label_revision);
  mixCurvatureSignature(seed, mesh_coordinate_configuration_key);
  mixCurvatureSignature(seed, system.spaceRevision());
  mixCurvatureSignature(seed, system.dofLayoutRevision());
  mixCurvatureSignature(seed, system.systemLayoutRevision());
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(phi_field));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max<svmp::FE::GlobalIndex>(
                                0, system.fieldDofOffset(phi_field))));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max<svmp::FE::GlobalIndex>(
                                0,
                                system.fieldDofHandler(phi_field).getNumDofs())));
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(rec.components));
  mixCurvatureSignature(seed, source_kind);
  mixCurvatureSignature(seed, source_revision);
  return seed;
}

void logLevelSetCurvatureProjectionDiagnostic(
    const LevelSetMaintenanceRequest& request,
    int step,
    std::string_view reason,
    const svmp::FE::level_set::LevelSetCurvatureProjectionResult& result,
    std::string_view cache_state,
    bool projection_skipped,
    std::uint64_t signature)
{
  application::core::oopCout()
      << "[svMultiPhysics::Application] Level-set curvature projected"
      << " field='" << request.level_set_field_name << "'"
      << " curvature_field='" << request.curvature_field_name << "'"
      << " step=" << step
      << " reason=" << reason
      << " cache=" << cache_state
      << " projection_skipped=" << (projection_skipped ? 1 : 0)
      << " signature=" << signature
      << " fitted_vertices=" << result.fitted_vertices
      << " supplemental_samples=" << result.supplemental_samples
      << " supplemental_sample_rows=" << result.supplemental_sample_rows
      << " vertices_with_supplemental_samples="
      << result.vertices_with_supplemental_samples
      << " fallback_vertices=" << result.fallback_vertices
      << " zero_fallback_vertices=" << result.zero_fallback_vertices
      << " insufficient_stencil_vertices="
      << result.insufficient_stencil_vertices
      << " singular_stencil_vertices=" << result.singular_stencil_vertices
      << " small_gradient_vertices=" << result.small_gradient_vertices
      << " fit_residual_failure_vertices="
      << result.fit_residual_failure_vertices
      << " smoothing_iterations="
      << result.smoothing_iterations_applied
      << " smoothing_mean_abs_update="
      << result.smoothing_mean_abs_update
      << " smoothing_max_abs_update="
      << result.smoothing_max_abs_update
      << " reused_vertex_adjacency="
      << (result.reused_vertex_adjacency ? 1 : 0)
      << " reused_sample_adjacency="
      << (result.reused_sample_adjacency ? 1 : 0)
      << " vertex_adjacency_builds="
      << result.vertex_adjacency_builds
      << " sample_adjacency_builds="
      << result.sample_adjacency_builds
      << " mean_fit_rms_residual=" << result.mean_fit_rms_residual
      << " max_fit_rms_residual=" << result.max_fit_rms_residual
      << " mean_normalized_fit_residual="
      << result.mean_normalized_fit_residual
      << " max_normalized_fit_residual="
      << result.max_normalized_fit_residual
      << " min_curvature=" << result.min_curvature
      << " max_curvature=" << result.max_curvature
      << " max_abs_curvature=" << result.max_abs_curvature
      << std::endl;
}

void setScalarPrescribedVertexFieldFromValues(
    svmp::FE::systems::FESystem& system,
    const svmp::Mesh& mesh,
    svmp::FE::FieldId field,
    std::span<const svmp::FE::Real> vertex_values,
    std::string_view context)
{
  const auto n_vertices = static_cast<svmp::FE::GlobalIndex>(mesh.n_vertices());
  if (vertex_values.size() != static_cast<std::size_t>(n_vertices)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " + std::string(context) +
        " received a vertex-value buffer with the wrong size.");
  }

  const auto& rec = system.fieldRecord(field);
  if (rec.source_kind != svmp::FE::systems::FieldSourceKind::PrescribedData ||
      rec.components != 1) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " + std::string(context) +
        " target field '" + rec.name +
        "' must be a scalar PrescribedData field.");
  }

  const auto& field_dofs = system.fieldDofHandler(field);
  const auto* entity_map = field_dofs.getEntityDofMap();
  if (entity_map == nullptr ||
      entity_map->numVertices() < n_vertices) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " + std::string(context) +
        " requires scalar vertex DOF metadata for field '" + rec.name + "'.");
  }

  const auto n_field_dofs =
      static_cast<std::size_t>(field_dofs.getNumDofs());
  std::vector<svmp::FE::Real> coefficients(n_field_dofs, 0.0);
  std::vector<unsigned char> assigned(n_field_dofs, 0u);

  bool all_mesh_vertices_have_scalar_vertex_dof = true;
  for (svmp::FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
    if (entity_map->getVertexDofs(vertex).size() != 1u) {
      all_mesh_vertices_have_scalar_vertex_dof = false;
      break;
    }
  }

  if (all_mesh_vertices_have_scalar_vertex_dof) {
    for (svmp::FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
      const auto vertex_dofs = entity_map->getVertexDofs(vertex);
      const auto dof = vertex_dofs.front();
      if (dof < 0 || static_cast<std::size_t>(dof) >= coefficients.size()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] " + std::string(context) +
            " found a vertex DOF outside the target field.");
      }
      coefficients[static_cast<std::size_t>(dof)] =
          vertex_values[static_cast<std::size_t>(vertex)];
      assigned[static_cast<std::size_t>(dof)] = 1u;
    }
  } else {
    const auto& local_mesh = mesh.local_mesh();
    for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
      auto [cell_vertices, n_cell_vertices] =
          local_mesh.cell_vertices_span(cell);
      if (cell_vertices == nullptr || n_cell_vertices == 0u) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] " + std::string(context) +
            " found empty cell connectivity.");
      }
      const auto cell_dofs =
          field_dofs.getCellDofs(static_cast<svmp::FE::GlobalIndex>(cell));
      if (cell_dofs.size() != n_cell_vertices) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] " + std::string(context) +
            " currently requires a nodal scalar target field matching mesh vertices; "
            "register projected curvature as a scalar P1/mesh-node PrescribedData field.");
      }
      for (std::size_t local_node = 0; local_node < n_cell_vertices; ++local_node) {
        const auto vertex = cell_vertices[local_node];
        if (vertex < 0 ||
            static_cast<std::size_t>(vertex) >= vertex_values.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] " + std::string(context) +
              " found an out-of-range mesh vertex.");
        }
        const auto dof = cell_dofs[local_node];
        if (dof < 0 || static_cast<std::size_t>(dof) >= coefficients.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] " + std::string(context) +
              " found a cell DOF outside the target field.");
        }
        coefficients[static_cast<std::size_t>(dof)] =
            vertex_values[static_cast<std::size_t>(vertex)];
        assigned[static_cast<std::size_t>(dof)] = 1u;
      }
    }
  }

  if (std::find(assigned.begin(), assigned.end(), 0u) != assigned.end()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " + std::string(context) +
        " left one or more target field DOFs unassigned.");
  }
  system.setPrescribedFieldCoefficients(
      field,
      std::span<const svmp::FE::Real>(coefficients.data(),
                                      coefficients.size()));
}

std::size_t projectLevelSetCurvatureFieldsFromState(
    application::core::SimulationComponents& sim,
    const svmp::FE::systems::SystemStateView& state,
    const std::vector<LevelSetMaintenanceRequest>& requests,
    int step,
    std::string_view reason,
    bool honor_cadence,
    CurvatureProjectionCache* curvature_cache = nullptr,
    bool reuse_cached_on_projection_failure = false)
{
  if (!sim.fe_system || !sim.primary_mesh || requests.empty()) {
    return 0u;
  }

  auto& system = *sim.fe_system;
  const auto& mesh = *sim.primary_mesh;
  std::size_t updated_fields = 0u;

  for (const auto& request : requests) {
    if (!request.curvature_projection_enabled) {
      continue;
    }
    if (honor_cadence && !shouldProjectLevelSetCurvature(request, step)) {
      continue;
    }
    if (!honor_cadence && request.curvature_field_name.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set curvature projection requires Curvature_field_name.");
    }

    const auto phi_field = system.findFieldByName(request.level_set_field_name);
    const auto kappa_field = system.findFieldByName(request.curvature_field_name);
    if (phi_field == svmp::FE::INVALID_FIELD_ID ||
        kappa_field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set curvature projection could not find level-set field '" +
          request.level_set_field_name + "' or curvature field '" +
          request.curvature_field_name + "'.");
    }

    auto options = request.curvature_projection;
    options.isovalue = static_cast<svmp::FE::Real>(request.isovalue);
    const auto& mesh_access = system.meshAccess();
    const bool mesh_revisions_available =
        mesh_access.revisionTrackingAvailable();
    const auto mesh_vertices = mesh_access.numVertices();
    const auto mesh_cells = mesh_access.numCells();
    const auto mesh_dimension = mesh_access.dimension();
    const auto mesh_geometry_revision =
        mesh_revisions_available ? mesh_access.geometryRevision() : 0u;
    const auto mesh_topology_revision =
        mesh_revisions_available ? mesh_access.topologyRevision() : 0u;
    const auto mesh_ownership_revision =
        mesh_revisions_available ? mesh_access.ownershipRevision() : 0u;
    const auto mesh_numbering_revision =
        mesh_revisions_available ? mesh_access.numberingRevision() : 0u;
    const auto mesh_field_layout_revision =
        mesh_revisions_available ? mesh_access.fieldLayoutRevision() : 0u;
    const auto mesh_label_revision =
        mesh_revisions_available ? mesh_access.labelRevision() : 0u;
    const auto mesh_coordinate_configuration_key =
        mesh_revisions_available
            ? mesh_access.coordinateConfigurationKey()
            : 0u;

    std::optional<std::uint64_t> fast_signature;
    CurvatureProjectionCacheEntry* cache_entry = nullptr;
    if (curvature_cache != nullptr) {
      const auto key = curvatureProjectionCacheKey(request);
      auto [it, inserted] = curvature_cache->entries.try_emplace(key);
      (void)inserted;
      cache_entry = &it->second;
      fast_signature = curvatureProjectionFastInputSignature(
          system,
          state,
          phi_field,
          request,
          options,
          mesh_vertices,
          mesh_cells,
          mesh_dimension,
          mesh_revisions_available,
          mesh_geometry_revision,
          mesh_topology_revision,
          mesh_ownership_revision,
          mesh_numbering_revision,
          mesh_field_layout_revision,
          mesh_label_revision,
          mesh_coordinate_configuration_key);
      if (fast_signature.has_value() &&
          cache_entry->valid &&
          cache_entry->fast_valid &&
          cache_entry->fast_signature == *fast_signature) {
        logLevelSetCurvatureProjectionDiagnostic(
            request,
            step,
            reason,
            cache_entry->last_result,
            "fast_hit",
            /*projection_skipped=*/true,
            cache_entry->signature);
        continue;
      }
    }

    const auto phi_values =
        evaluateVertexField(system, mesh, phi_field, state, 1u);
    std::vector<svmp::FE::Real> phi(
        phi_values.begin(), phi_values.end());
    const auto supplemental_samples =
        collectLevelSetCurvatureSupplementalSamples(system, state, phi_field);
    std::vector<svmp::FE::Real> curvature;
    const auto signature = curvatureProjectionInputSignature(
        request,
        options,
        std::span<const svmp::FE::Real>(phi.data(), phi.size()),
        std::span<const svmp::FE::level_set::LevelSetCurvatureProjectionSample>(
            supplemental_samples.data(), supplemental_samples.size()),
        mesh_vertices,
        mesh_cells,
        mesh_dimension,
        mesh_revisions_available,
        mesh_geometry_revision,
        mesh_topology_revision,
        mesh_ownership_revision,
        mesh_numbering_revision,
        mesh_coordinate_configuration_key);
    if (cache_entry != nullptr) {
      if (cache_entry->valid && cache_entry->signature == signature) {
        if (fast_signature.has_value()) {
          cache_entry->fast_valid = true;
          cache_entry->fast_signature = *fast_signature;
        } else {
          cache_entry->fast_valid = false;
          cache_entry->fast_signature = 0u;
        }
        logLevelSetCurvatureProjectionDiagnostic(
            request,
            step,
            reason,
            cache_entry->last_result,
            "hit",
            /*projection_skipped=*/true,
            signature);
        continue;
      }
    }

    const auto result =
        cache_entry != nullptr
            ? svmp::FE::level_set::projectLevelSetMeanCurvatureToVertices(
                  system.meshAccess(),
                  phi,
                  supplemental_samples,
                  options,
                  curvature,
                  cache_entry->workspace)
            : svmp::FE::level_set::projectLevelSetMeanCurvatureToVertices(
                  system.meshAccess(),
                  phi,
                  supplemental_samples,
                  options,
                  curvature);
    if (!result.success) {
      if (reuse_cached_on_projection_failure &&
          cache_entry != nullptr &&
          cache_entry->valid) {
        application::core::oopCout()
            << "[svMultiPhysics::Application] WARNING Level-set curvature projection"
            << " failed residual screening and reused cached curvature"
            << " field='" << request.level_set_field_name << "'"
            << " curvature_field='" << request.curvature_field_name << "'"
            << " step=" << step
            << " reason=" << reason
            << " diagnostic=curvature_projection_cached_after_failed_trial"
            << " failure='" << result.diagnostic << "'"
            << " previous_signature=" << cache_entry->signature
            << std::endl;
        logLevelSetCurvatureProjectionDiagnostic(
            request,
            step,
            reason,
            cache_entry->last_result,
            "stale_after_failed_trial",
            /*projection_skipped=*/true,
            cache_entry->signature);
        continue;
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set curvature projection failed for field '" +
          request.level_set_field_name + "': " + result.diagnostic);
    }

    setScalarPrescribedVertexFieldFromValues(
        system,
        mesh,
        kappa_field,
        std::span<const svmp::FE::Real>(curvature.data(), curvature.size()),
        "Level-set curvature projection");
    if (cache_entry != nullptr) {
      cache_entry->valid = true;
      cache_entry->signature = signature;
      if (fast_signature.has_value()) {
        cache_entry->fast_valid = true;
        cache_entry->fast_signature = *fast_signature;
      } else {
        cache_entry->fast_valid = false;
        cache_entry->fast_signature = 0u;
      }
      cache_entry->last_result = result;
    }
    ++updated_fields;

    logLevelSetCurvatureProjectionDiagnostic(
        request,
        step,
        reason,
        result,
        cache_entry != nullptr ? "miss" : "disabled",
        /*projection_skipped=*/false,
        signature);
  }

  return updated_fields;
}

std::size_t syncActiveLevelSetVertexFieldsFromSolution(
    application::core::SimulationComponents& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::span<const svmp::FE::Real> fe_solution)
{
  if (!sim.fe_system || !sim.primary_mesh || requests.empty()) {
    return 0u;
  }

  auto& system = *sim.fe_system;
  auto& mesh = *sim.primary_mesh;
  const auto n_vertices =
      static_cast<svmp::FE::GlobalIndex>(mesh.n_vertices());
  std::set<std::string> visited_fields;
  std::size_t changed_fields = 0u;

  for (const auto& request : requests) {
    if (!visited_fields.insert(request.level_set_field_name).second) {
      continue;
    }
    if (!mesh.has_field(svmp::EntityKind::Vertex,
                        request.level_set_field_name)) {
      continue;
    }

    const auto field = system.findFieldByName(request.level_set_field_name);
    if (field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active level-set support refresh could not find FE field '" +
          request.level_set_field_name + "'.");
    }
    const auto& field_dofs = system.fieldDofHandler(field);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr ||
        entity_map->numVertices() != n_vertices) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active level-set support refresh requires one vertex DOF map matching the mesh.");
    }

    const auto field_offset = system.fieldDofOffset(field);
    const auto n_field_dofs = field_dofs.getNumDofs();
    if (field_offset < 0 ||
        n_field_dofs < 0 ||
        static_cast<std::size_t>(field_offset + n_field_dofs) >
            fe_solution.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active level-set support refresh received an incompatible solution span.");
    }

    const auto handle = mesh.field_handle(
        svmp::EntityKind::Vertex, request.level_set_field_name);
    if (mesh.field_type(handle) != svmp::FieldScalarType::Float64 ||
        mesh.field_components(handle) < 1u ||
        mesh.field_entity_count(handle) <
            static_cast<std::size_t>(n_vertices)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active level-set support refresh requires a scalar Float64 vertex mesh field.");
    }
    auto* data = static_cast<double*>(mesh.field_data(handle));
    if (data == nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active level-set support refresh found an empty vertex mesh field.");
    }

    const auto components = mesh.field_components(handle);
    bool field_changed = false;
    bool all_mesh_vertices_have_scalar_vertex_dof = true;
    for (svmp::FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
      const auto vertex_dofs = entity_map->getVertexDofs(vertex);
      if (vertex_dofs.size() != 1u) {
        all_mesh_vertices_have_scalar_vertex_dof = false;
        break;
      }
    }

    if (all_mesh_vertices_have_scalar_vertex_dof) {
      for (svmp::FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto vertex_dofs = entity_map->getVertexDofs(vertex);
        const auto dof = field_offset + vertex_dofs.front();
        if (dof < 0 ||
            static_cast<std::size_t>(dof) >= fe_solution.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Active level-set support refresh found a vertex DOF outside the solution span.");
        }
        const auto value = static_cast<double>(
            fe_solution[static_cast<std::size_t>(dof)]);
        auto& target =
            data[static_cast<std::size_t>(vertex) * components];
        if (target != value) {
          target = value;
          field_changed = true;
        }
      }
    } else {
      const auto& local_mesh = mesh.local_mesh();
      std::vector<std::uint8_t> vertex_written(
          static_cast<std::size_t>(n_vertices),
          0u);
      for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
        auto [cell_vertices, n_cell_vertices] =
            local_mesh.cell_vertices_span(cell);
        if (cell_vertices == nullptr || n_cell_vertices == 0u) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Active level-set support refresh found empty cell connectivity.");
        }
        const auto cell_dofs = field_dofs.getCellDofs(
            static_cast<svmp::FE::GlobalIndex>(cell));
        if (cell_dofs.size() != n_cell_vertices) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Active level-set support refresh found incompatible high-order cell DOFs.");
        }

        for (std::size_t local_node = 0;
             local_node < n_cell_vertices;
             ++local_node) {
          const auto vertex = cell_vertices[local_node];
          if (vertex < 0 ||
              vertex >= static_cast<svmp::index_t>(n_vertices)) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Active level-set support refresh found an out-of-range mesh vertex.");
          }
          const auto dof = field_offset + cell_dofs[local_node];
          if (dof < 0 ||
              static_cast<std::size_t>(dof) >= fe_solution.size()) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Active level-set support refresh found a cell DOF outside the solution span.");
          }
          const auto value = static_cast<double>(
              fe_solution[static_cast<std::size_t>(dof)]);
          const auto vertex_index = static_cast<std::size_t>(vertex);
          auto& target = data[vertex_index * components];
          if (target != value) {
            target = value;
            field_changed = true;
          }
          vertex_written[vertex_index] = 1u;
        }
      }
      if (std::find(vertex_written.begin(), vertex_written.end(), 0u) !=
          vertex_written.end()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Active level-set support refresh left a mesh vertex without FE data.");
      }
    }
    if (field_changed) {
      ++changed_fields;
    }
  }

  if (changed_fields > 0u) {
    mesh.event_bus().notify(svmp::MeshEvent::FieldsChanged);
  }

  return changed_fields;
}

bool updateLevelSetAdvectionVelocitiesFromState(
    application::core::SimulationComponents& sim,
    const svmp::FE::systems::SystemStateView& state,
    const std::vector<LevelSetAdvectionVelocityRequest>& requests)
{
  if (!sim.fe_system || !sim.primary_mesh || requests.empty()) {
    return false;
  }

  auto& system = *sim.fe_system;
  const auto& mesh = *sim.primary_mesh;
  const auto n_vertices = mesh.n_vertices();
  const int mesh_dim = mesh.dim();
  const auto& coords = mesh.X_ref();
  const bool trace_updates =
      parseBoolEnv("SVMP_TRACE_LEVEL_SET_ADVECTION", false) ||
      application::core::oopTraceEnabled();

  bool updated = false;
  for (const auto& request : requests) {
    const auto phi_field = system.findFieldByName(request.level_set_field_name);
    const auto source_field =
        system.findFieldByName(request.source_velocity_field_name);
    const auto target_field =
        system.findFieldByName(request.target_velocity_field_name);
    if (phi_field == svmp::FE::INVALID_FIELD_ID ||
        source_field == svmp::FE::INVALID_FIELD_ID ||
        target_field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Could not find fields needed for wet-extension level-set advection velocity update.");
    }

    const auto& target_rec = system.fieldRecord(target_field);
    if (target_rec.source_kind !=
        svmp::FE::systems::FieldSourceKind::PrescribedData) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity field '" +
          target_rec.name + "' must be registered as PrescribedData.");
    }

    const auto& source_rec = system.fieldRecord(source_field);
    const auto source_components =
        static_cast<std::size_t>(std::max(1, source_rec.components));
    const auto target_components =
        static_cast<std::size_t>(std::max(1, target_rec.components));
    const auto copy_components =
        std::min(source_components, target_components);
    if (copy_components == 0u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity update found an empty velocity field.");
    }

    const auto phi_values =
        evaluateVertexField(system, mesh, phi_field, state, 1u);
    const auto source_values =
        evaluateVertexField(system, mesh, source_field, state, source_components);

    std::vector<std::uint8_t> active(n_vertices, 0u);
    std::vector<std::size_t> active_vertices;
    active_vertices.reserve(n_vertices);
    for (std::size_t v = 0; v < n_vertices; ++v) {
      const bool is_active = activeSideContains(phi_values[v], request);
      active[v] = is_active ? 1u : 0u;
      if (is_active) {
        active_vertices.push_back(v);
      }
    }

    std::vector<double> extended(n_vertices * target_components, 0.0);
    double max_active_speed = 0.0;
    double max_dry_extended_speed = 0.0;

    auto record_speed = [&](std::size_t v) {
      double speed2 = 0.0;
      for (std::size_t c = 0; c < copy_components; ++c) {
        const auto value = extended[v * target_components + c];
        speed2 += value * value;
      }
      const double speed = std::sqrt(speed2);
      if (active[v]) {
        max_active_speed = std::max(max_active_speed, speed);
      } else {
        max_dry_extended_speed = std::max(max_dry_extended_speed, speed);
      }
    };

    if (request.extension_method == "nearest_interface_point") {
      if (mesh_dim != 2) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] nearest_interface_point level-set "
            "advection velocity extension currently supports 2D meshes.");
      }

      struct InterfaceVelocitySample {
        std::array<double, 3> point{0.0, 0.0, 0.0};
        std::vector<double> value{};
      };
      std::vector<InterfaceVelocitySample> samples;

      auto add_interface_sample = [&](std::size_t va,
                                      std::size_t vb,
                                      double edge_t) {
        edge_t = std::clamp(edge_t, 0.0, 1.0);
        InterfaceVelocitySample sample;
        sample.value.assign(target_components, 0.0);
        for (int d = 0; d < mesh_dim; ++d) {
          const auto oa =
              va * static_cast<std::size_t>(mesh_dim) +
              static_cast<std::size_t>(d);
          const auto ob =
              vb * static_cast<std::size_t>(mesh_dim) +
              static_cast<std::size_t>(d);
          sample.point[static_cast<std::size_t>(d)] =
              (1.0 - edge_t) * coords[oa] + edge_t * coords[ob];
        }
        for (std::size_t c = 0; c < copy_components; ++c) {
          const auto a =
              source_values[va * source_components + c];
          const auto b =
              source_values[vb * source_components + c];
          const auto value = (1.0 - edge_t) * a + edge_t * b;
          sample.value[c] = std::isfinite(value) ? value : 0.0;
        }
        samples.push_back(std::move(sample));
      };

      const auto& local_mesh = mesh.local_mesh();
      constexpr double zero_tol = 1.0e-12;
      auto add_interface_segment = [&](std::size_t va, std::size_t vb) {
        if (va >= n_vertices || vb >= n_vertices) {
          return;
        }
        const double phia = phi_values[va] - request.isovalue;
        const double phib = phi_values[vb] - request.isovalue;
        const bool a_zero = std::abs(phia) <= zero_tol;
        const bool b_zero = std::abs(phib) <= zero_tol;
        if (a_zero && b_zero) {
          add_interface_sample(va, vb, 0.0);
          add_interface_sample(va, vb, 1.0);
        } else if (a_zero) {
          add_interface_sample(va, vb, 0.0);
        } else if (b_zero) {
          add_interface_sample(va, vb, 1.0);
        } else if (phia * phib < 0.0) {
          add_interface_sample(va, vb, phia / (phia - phib));
        }
      };
      for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
        auto [cell_vertices, n_cell_vertices] =
            local_mesh.cell_vertices_span(cell);
        if (cell_vertices == nullptr || n_cell_vertices < 2u) {
          continue;
        }
        const auto family = local_mesh.cell_shape(cell).family;
        if (family == svmp::CellFamily::Polygon) {
          const auto& shape = local_mesh.cell_shape(cell);
          const int polygon_corners =
              shape.num_corners > 0
                  ? std::min<int>(shape.num_corners,
                                  static_cast<int>(n_cell_vertices))
                  : static_cast<int>(n_cell_vertices);
          if (polygon_corners < 2) {
            continue;
          }
          const auto polygon_edges =
              svmp::CellTopology::get_polygon_edges_view(polygon_corners);
          for (int edge = 0; edge < polygon_edges.edge_count; ++edge) {
            auto edge_dofs = local_mesh.cell_edge_geometry_dofs(cell, edge);
            if (edge_dofs.size() < 2u) {
              const auto local_a = polygon_edges.pairs_flat[2 * edge];
              const auto local_b = polygon_edges.pairs_flat[2 * edge + 1];
              if (local_a < 0 || local_b < 0 ||
                  static_cast<std::size_t>(local_a) >= n_cell_vertices ||
                  static_cast<std::size_t>(local_b) >= n_cell_vertices) {
                continue;
              }
              edge_dofs = {
                  cell_vertices[static_cast<std::size_t>(local_a)],
                  cell_vertices[static_cast<std::size_t>(local_b)]};
            }
            for (std::size_t i = 0; i + 1u < edge_dofs.size(); ++i) {
              const auto raw_a = edge_dofs[i];
              const auto raw_b = edge_dofs[i + 1u];
              if (raw_a < 0 || raw_b < 0) {
                continue;
              }
              add_interface_segment(static_cast<std::size_t>(raw_a),
                                    static_cast<std::size_t>(raw_b));
            }
          }
          continue;
        }

        const auto edge_view = svmp::CellTopology::get_edges_view(family);
        for (int edge = 0; edge < edge_view.edge_count; ++edge) {
          auto edge_dofs = local_mesh.cell_edge_geometry_dofs(cell, edge);
          if (edge_dofs.size() < 2u) {
            const auto local_a = edge_view.pairs_flat[2 * edge];
            const auto local_b = edge_view.pairs_flat[2 * edge + 1];
            if (local_a < 0 || local_b < 0 ||
                static_cast<std::size_t>(local_a) >= n_cell_vertices ||
                static_cast<std::size_t>(local_b) >= n_cell_vertices) {
              continue;
            }
            edge_dofs = {
                cell_vertices[static_cast<std::size_t>(local_a)],
                cell_vertices[static_cast<std::size_t>(local_b)]};
          }
          for (std::size_t i = 0; i + 1u < edge_dofs.size(); ++i) {
            const auto raw_a = edge_dofs[i];
            const auto raw_b = edge_dofs[i + 1u];
            if (raw_a < 0 || raw_b < 0) {
              continue;
            }
            add_interface_segment(static_cast<std::size_t>(raw_a),
                                  static_cast<std::size_t>(raw_b));
          }
        }
      }

      if (samples.empty()) {
        if (trace_updates) {
          const auto [min_it, max_it] =
              std::minmax_element(phi_values.begin(), phi_values.end());
          application::core::oopCout()
              << "[svMultiPhysics::Application] Skipped level-set advection velocity update"
              << " field='" << request.target_velocity_field_name << "'"
              << " source='" << request.source_velocity_field_name << "'"
              << " extension_method=" << request.extension_method
              << " reason=no_interface_crossings"
              << " phi_min=" << (min_it != phi_values.end() ? *min_it : 0.0)
              << " phi_max=" << (max_it != phi_values.end() ? *max_it : 0.0)
              << std::endl;
        }
        continue;
      }

      std::vector<NearestPointRecord> sample_records;
      sample_records.reserve(samples.size());
      for (std::size_t s = 0; s < samples.size(); ++s) {
        sample_records.push_back(NearestPointRecord{samples[s].point, s});
      }
      const NearestPointIndex sample_index(mesh_dim, std::move(sample_records));

      for (std::size_t v = 0; v < n_vertices; ++v) {
        const auto nearest =
            sample_index.nearest(meshVertexPoint(coords, mesh_dim, v));
        if (!nearest.found || nearest.payload >= samples.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set advection velocity extension failed to find a nearest interface sample.");
        }
        const auto best_sample = nearest.payload;
        for (std::size_t c = 0; c < copy_components; ++c) {
          extended[v * target_components + c] =
              samples[best_sample].value[c];
        }
        record_speed(v);
      }
    } else {
      if (active_vertices.empty()) {
        if (trace_updates) {
          const auto [min_it, max_it] =
              std::minmax_element(phi_values.begin(), phi_values.end());
          application::core::oopCout()
              << "[svMultiPhysics::Application] Skipped level-set advection velocity update"
              << " field='" << request.target_velocity_field_name << "'"
              << " source='" << request.source_velocity_field_name << "'"
              << " extension_method=" << request.extension_method
              << " reason=no_active_vertices"
              << " phi_min=" << (min_it != phi_values.end() ? *min_it : 0.0)
              << " phi_max=" << (max_it != phi_values.end() ? *max_it : 0.0)
              << std::endl;
        }
        continue;
      }
      std::vector<NearestPointRecord> active_records;
      active_records.reserve(active_vertices.size());
      for (const auto candidate : active_vertices) {
        active_records.push_back(
            NearestPointRecord{meshVertexPoint(coords, mesh_dim, candidate),
                               candidate});
      }
      const NearestPointIndex active_index(mesh_dim, std::move(active_records));

      for (std::size_t v = 0; v < n_vertices; ++v) {
        std::size_t source_vertex = v;
        if (!active[v]) {
          const auto nearest =
              active_index.nearest(meshVertexPoint(coords, mesh_dim, v));
          if (!nearest.found || nearest.payload >= n_vertices) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Level-set advection velocity extension failed to find a nearest active vertex.");
          }
          source_vertex = nearest.payload;
        }

        for (std::size_t c = 0; c < copy_components; ++c) {
          const auto value =
              source_values[source_vertex * source_components + c];
          extended[v * target_components + c] =
              std::isfinite(value) ? value : 0.0;
        }
        record_speed(v);
      }
    }

    const auto& target_dofs = system.fieldDofHandler(target_field);
    const auto* entity_map = target_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity update requires vertex DOFs.");
    }
    std::vector<svmp::FE::Real> coefficients(
        static_cast<std::size_t>(target_dofs.getNumDofs()),
        svmp::FE::Real{0.0});
    std::vector<std::uint8_t> assigned(coefficients.size(), 0u);

    bool all_mesh_vertices_have_vertex_dofs = true;
    for (std::size_t v = 0; v < n_vertices; ++v) {
      const auto vertex_dofs =
          entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
      if (vertex_dofs.size() < target_components) {
        all_mesh_vertices_have_vertex_dofs = false;
        break;
      }
    }

    if (all_mesh_vertices_have_vertex_dofs) {
      for (std::size_t v = 0; v < n_vertices; ++v) {
        const auto vertex_dofs =
            entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
        for (std::size_t c = 0; c < target_components; ++c) {
          const auto dof = vertex_dofs[c];
          if (dof < 0 ||
              static_cast<std::size_t>(dof) >= coefficients.size()) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Level-set advection velocity update found an out-of-range vertex DOF.");
          }
          const auto sdof = static_cast<std::size_t>(dof);
          coefficients[sdof] =
              static_cast<svmp::FE::Real>(extended[v * target_components + c]);
          assigned[sdof] = 1u;
        }
      }
    } else {
      const auto& local_mesh = mesh.local_mesh();
      for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
        auto [cell_vertices, n_cell_vertices] =
            local_mesh.cell_vertices_span(cell);
        if (cell_vertices == nullptr || n_cell_vertices == 0u) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set advection velocity update found empty cell connectivity.");
        }
        const auto cell_dofs = target_dofs.getCellDofs(
            static_cast<svmp::FE::GlobalIndex>(cell));
        const auto expected_cell_dofs =
            n_cell_vertices * target_components;
        if (cell_dofs.size() != expected_cell_dofs) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set advection velocity update found incompatible high-order cell DOFs.");
        }

        for (std::size_t local_node = 0;
             local_node < n_cell_vertices;
             ++local_node) {
          const auto vertex = cell_vertices[local_node];
          if (vertex < 0 ||
              static_cast<std::size_t>(vertex) >= n_vertices) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Level-set advection velocity update found an out-of-range mesh vertex.");
          }
          for (std::size_t c = 0; c < target_components; ++c) {
            const auto cell_dof_position =
                c * n_cell_vertices + local_node;
            const auto dof = cell_dofs[cell_dof_position];
            if (dof < 0 ||
                static_cast<std::size_t>(dof) >= coefficients.size()) {
              throw std::runtime_error(
                  "[svMultiPhysics::Application] Level-set advection velocity update found an out-of-range cell DOF.");
            }
            const auto sdof = static_cast<std::size_t>(dof);
            coefficients[sdof] = static_cast<svmp::FE::Real>(
                extended[static_cast<std::size_t>(vertex) *
                             target_components +
                         c]);
            assigned[sdof] = 1u;
          }
        }
      }
    }

    std::size_t unassigned = 0u;
    for (const auto flag : assigned) {
      unassigned += flag ? 0u : 1u;
    }
    system.setPrescribedFieldCoefficients(
        target_field,
        std::span<const svmp::FE::Real>(coefficients.data(),
                                        coefficients.size()));
    updated = true;
    if (trace_updates) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] Updated level-set advection velocity field='"
          << request.target_velocity_field_name << "' from source='"
          << request.source_velocity_field_name << "' extension_method="
          << request.extension_method << " active_vertices="
          << active_vertices.size() << " dry_vertices="
          << (n_vertices - active_vertices.size())
          << " max_active_speed=" << max_active_speed
          << " max_dry_extended_speed=" << max_dry_extended_speed
          << " unassigned_dofs=" << unassigned << std::endl;
    }
  }

  return updated;
}

bool updateLevelSetAdvectionVelocities(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    const std::vector<LevelSetAdvectionVelocityRequest>& requests)
{
  const auto state = stateViewForHistory(history);
  return updateLevelSetAdvectionVelocitiesFromState(sim, state, requests);
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextFromSolution(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    std::span<const svmp::FE::Real> fe_solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    const char* provenance,
    const char* solution_source = nullptr)
{
  ActiveCutContextRefreshReport report{};
  if (!sim.fe_system) {
    return report;
  }

  const auto requests = activeCutVolumeRequests(params);
  if (requests.empty()) {
    return report;
  }
  report.request_policy_key = activeCutVolumeRequestPolicyKey(requests);

  const auto synchronized_level_set_fields =
      syncActiveLevelSetVertexFieldsFromSolution(sim, requests, fe_solution);

  auto context =
      std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
  report.refreshed = true;
  report.topology_key = kCutContextHashOffset;
  const auto comm = svmp::MeshComm::world();
  const auto& mesh_access = sim.fe_system->meshAccess();

  for (const auto& request : requests) {
    svmp::FE::level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = request.level_set_field_name;
    options.domain_id = request.domain_id;
    options.requested_interface_marker = request.requested_interface_marker;
    options.isovalue = static_cast<svmp::FE::Real>(request.isovalue);
    if (request.quadrature_order.has_value()) {
      options.quadrature_order = *request.quadrature_order;
    }
    if (request.interface_quadrature_order.has_value()) {
      options.interface_quadrature_order = *request.interface_quadrature_order;
    }
    if (request.volume_quadrature_order.has_value()) {
      options.volume_quadrature_order = *request.volume_quadrature_order;
    }
    if (!request.interface_quadrature_order.has_value() &&
        mesh_access.dimension() == 2 &&
        options.interface_quadrature_order < 0) {
      options.interface_quadrature_order = options.volume_quadrature_order;
    }
    options.geometry_mode = request.geometry_mode;
    options.implicit_cut_quadrature_backend = request.implicit_cut_backend;
    options.implicit_cut_fallback_policy =
        request.implicit_cut_fallback_policy;
    options.geometry_tangent_policy = request.geometry_tangent_policy;
    options.implicit_cut_root_tolerance =
        static_cast<svmp::FE::Real>(request.implicit_cut_root_tolerance);
    options.implicit_cut_root_coordinate_tolerance =
        static_cast<svmp::FE::Real>(
            request.implicit_cut_root_coordinate_tolerance);
    options.implicit_cut_root_max_iterations =
        request.implicit_cut_root_max_iterations;
    options.implicit_cut_max_subdivision_depth =
        request.implicit_cut_max_subdivision_depth;
    options.allow_corner_linearized_geometry =
        request.allow_corner_linearized_geometry;
    options.require_production_qualified_implicit_cut_backend =
        request.require_production_qualified_implicit_cut_backend;

    const auto backend_start = Clock::now();
    auto result = lifecycle.build(*sim.fe_system, options, fe_solution);
    const auto backend_timing =
        reduceOutputTiming(secondsSince(backend_start), comm);
    if (!result.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Generated active-domain interface '" +
          request.domain_id + "' for level-set field '" +
          request.level_set_field_name + "' failed: " + result.diagnostic);
    }
    validateEquationLevelCutVolumeConsumer(
        *sim.fe_system, request, result.interface_marker);

    const auto summary = result.summary;
    logCornerLinearizedCutWarningOnce(request, result);
    const auto active_summary = summarizeActiveSideRegions(
        result.domain,
        request.active_side,
        static_cast<std::size_t>(std::max<svmp::FE::GlobalIndex>(
            0, mesh_access.numCells())));
    const auto raw_active_volume =
        request.active_side == LevelSetActiveSide::Negative
            ? summary.negative_volume_measure
            : summary.positive_volume_measure;
    const auto active_volume = active_summary.active_volume;
    const auto global_raw_active_volume = static_cast<svmp::FE::Real>(
        globalSumDouble(static_cast<double>(raw_active_volume), comm));
    const auto global_active_volume = static_cast<svmp::FE::Real>(
        globalSumDouble(static_cast<double>(active_volume), comm));
    if (!(global_active_volume > svmp::FE::Real{0.0})) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active-domain cut context marker=" +
          std::to_string(result.interface_marker) + " field='" +
          request.level_set_field_name + "' domain_id='" + request.domain_id +
          "' active_side=" + activeSideName(request.active_side) +
          " isovalue=" + std::to_string(request.isovalue) +
          " has zero retained active wet volume after generated cut-volume pruning.");
    }
    const auto topology_key = cutContextTopologyKey(result.domain);
    const auto& domain_request = result.domain.request();
    mixCutContextHash(report.topology_key, topology_key);
    report.value_revision = result.value_revision;
    const auto global_interface_fragments =
        globalSumSize(summary.fragment_count, comm);
    const auto global_active_interface_fragments =
        globalSumSize(summary.active_fragment_count, comm);
    const auto global_interface_quadrature_points =
        globalSumSize(summary.quadrature_point_count, comm);
    const auto global_domain_volume_quadrature_points =
        globalSumSize(summary.volume_quadrature_point_count, comm);
    const auto global_domain_total_quadrature_points =
        globalSumSize(summary.total_quadrature_point_count, comm);
    const auto global_active_volume_regions =
        globalSumSize(active_summary.active_volume_regions, comm);
    const auto global_raw_active_volume_regions =
        globalSumSize(summary.active_volume_region_count, comm);
    const auto global_pruned_volume_regions =
        globalSumSize(active_summary.pruned_volume_regions, comm);
    const auto global_pruned_volume = static_cast<svmp::FE::Real>(
        globalSumDouble(static_cast<double>(active_summary.pruned_volume), comm));
    const auto global_negative_volume = static_cast<svmp::FE::Real>(
        globalSumDouble(static_cast<double>(summary.negative_volume_measure), comm));
    const auto global_positive_volume = static_cast<svmp::FE::Real>(
        globalSumDouble(static_cast<double>(summary.positive_volume_measure), comm));
    const auto global_cell_count =
        globalSumSize(result.cell_count, comm);
    const auto global_corner_linearized_cells =
        globalSumSize(result.corner_linearized_cell_count, comm);
    const auto global_implicit_cut_fallback_cells =
        globalSumSize(result.implicit_cut_fallback_cell_count, comm);
    const auto global_backend_volume_quadrature_points =
        globalSumSize(result.backend_volume_quadrature_point_count, comm);
    const auto global_backend_interface_quadrature_points =
        globalSumSize(result.backend_interface_quadrature_point_count, comm);
    const auto global_backend_elapsed_seconds =
        globalSumDouble(result.backend_elapsed_seconds, comm);
    const auto global_generated_cell_cache_hits =
        globalSumSize(result.cell_cache_hits, comm);
    const auto global_generated_cell_cache_misses =
        globalSumSize(result.cell_cache_misses, comm);
    const auto global_generated_domain_cache_hits =
        globalSumSize(result.domain_cache_hits, comm);
    const auto global_linear_full_cell_fast_path_cells =
        globalSumSize(result.linear_full_cell_fast_path_count, comm);
    std::array<std::size_t, 5> global_selected_backend_counts{};
    for (std::size_t i = 0; i < global_selected_backend_counts.size(); ++i) {
      global_selected_backend_counts[i] = globalSumSize(
          result.selected_implicit_cut_quadrature_backend_counts[i], comm);
    }
    const auto local_backend_qualification_counts =
        localImplicitCutBackendQualificationCounts(
            mesh_access,
            options.implicit_cut_quadrature_backend,
            options.implicit_cut_fallback_policy);
    std::array<std::size_t, 3> global_backend_qualification_counts{};
    for (std::size_t i = 0; i < global_backend_qualification_counts.size(); ++i) {
      global_backend_qualification_counts[i] =
          globalSumSize(local_backend_qualification_counts[i], comm);
    }
    report.interface_fragments += global_interface_fragments;
    report.domain_interface_quadrature_point_count +=
        global_interface_quadrature_points;
    report.domain_volume_quadrature_point_count +=
        global_domain_volume_quadrature_points;
    report.domain_total_quadrature_point_count +=
        global_domain_total_quadrature_points;
    report.cell_count += global_cell_count;
    report.corner_linearized_cell_count += global_corner_linearized_cells;
    report.active_volume_regions += global_active_volume_regions;
    report.backend_volume_quadrature_point_count +=
        global_backend_volume_quadrature_points;
    report.backend_interface_quadrature_point_count +=
        global_backend_interface_quadrature_points;
    report.backend_elapsed_seconds += global_backend_elapsed_seconds;
    report.generated_cell_cache_hits += global_generated_cell_cache_hits;
    report.generated_cell_cache_misses += global_generated_cell_cache_misses;
    report.generated_domain_cache_hits += global_generated_domain_cache_hits;
    report.linear_full_cell_fast_path_count +=
        global_linear_full_cell_fast_path_cells;
    report.negative_volume += global_negative_volume;
    report.positive_volume += global_positive_volume;
    const auto generated_pruned_count_before =
        context->generatedPrunedVolumeRuleCount();
    const auto generated_pruned_volume_before =
        context->generatedPrunedVolumeMeasure();
    const auto active_volume_side = cutIntegrationSide(request.active_side);
    const auto domain_volume_rules = result.domain.volumeQuadratureRules();
    const auto negative_domain_volume_rules =
        retainedVolumeRulePointersForSide(
            domain_volume_rules,
            svmp::FE::geometry::CutIntegrationSide::Negative);
    const auto positive_domain_volume_rules =
        retainedVolumeRulePointersForSide(
            domain_volume_rules,
            svmp::FE::geometry::CutIntegrationSide::Positive);
    context->addGeneratedInterfaceDomain(result.domain, active_volume_side);
    const auto active_measure_summary =
        application::core::collectCutVolumeMeasures(
            mesh_access,
            context->generatedVolumeRulesForMarkerAndSide(
                result.interface_marker,
                active_volume_side));
    const auto negative_measure_summary =
        application::core::collectCutVolumeMeasures(
            mesh_access, negative_domain_volume_rules);
    const auto positive_measure_summary =
        application::core::collectCutVolumeMeasures(
            mesh_access, positive_domain_volume_rules);
    const auto global_active_physical_volume =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(active_measure_summary.physical_measure),
            comm));
    const auto global_active_physical_volume_rules =
        globalSumSize(active_measure_summary.physical_rule_count, comm);
    const auto global_active_skipped_physical_volume_rules =
        globalSumSize(active_measure_summary.skipped_physical_rule_count, comm);
    const auto global_negative_physical_volume =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(negative_measure_summary.physical_measure),
            comm));
    const auto global_positive_physical_volume =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(positive_measure_summary.physical_measure),
            comm));
    const auto global_negative_skipped_physical_volume_rules =
        globalSumSize(negative_measure_summary.skipped_physical_rule_count, comm);
    const auto global_positive_skipped_physical_volume_rules =
        globalSumSize(positive_measure_summary.skipped_physical_rule_count, comm);
    report.negative_physical_volume += global_negative_physical_volume;
    report.positive_physical_volume += global_positive_physical_volume;
    const auto local_generated_pruned_volume_rules =
        context->generatedPrunedVolumeRuleCount() -
        generated_pruned_count_before;
    const auto local_generated_pruned_volume =
        context->generatedPrunedVolumeMeasure() -
        generated_pruned_volume_before;
    const auto global_generated_pruned_volume_rules =
        globalSumSize(local_generated_pruned_volume_rules, comm);
    const auto global_generated_pruned_volume = static_cast<svmp::FE::Real>(
        globalSumDouble(static_cast<double>(local_generated_pruned_volume), comm));
    const auto facet_set_handle = addGeneratedCutAdjacentFacetSet(
        *context, result.domain, mesh_access, request.active_side);
    mixCutContextHash(report.topology_key, facet_set_handle.stable_id);
    const auto facet_scale_summary =
        summarizeCutAdjacentFacetScales(facet_set_handle);
    const auto global_cut_adjacent_facets =
        globalSumSize(facet_set_handle.facets.size(), comm);
    const auto global_cut_adjacent_metadata =
        globalSumSize(facet_scale_summary.metadata_count, comm);
    report.cut_adjacent_facets += global_cut_adjacent_facets;
    const auto global_active_wet_cells =
        globalSumSize(active_summary.active_wet_cells, comm);
    const auto global_cut_cells =
        globalSumSize(active_summary.cut_cell_count, comm);
    const auto global_full_wet_cells =
        globalSumSize(active_summary.full_wet_cell_count, comm);
    const auto global_full_dry_cells =
        globalSumSize(active_summary.full_dry_cell_count, comm);
    const auto global_active_quadrature_points =
        globalSumSize(active_summary.active_quadrature_points, comm);
    report.active_cut_cells += global_cut_cells;
    report.active_quadrature_points += global_active_quadrature_points;
    const auto global_empty_quadrature_regions =
        globalSumSize(active_summary.empty_quadrature_regions, comm);
    const auto global_nonfinite_measure_regions =
        globalSumSize(active_summary.nonfinite_measure_regions, comm);
    const auto global_negative_measure_regions =
        globalSumSize(active_summary.negative_measure_regions, comm);
    const auto local_min_volume_fraction =
        active_summary.active_volume_regions > 0u
            ? static_cast<double>(active_summary.min_volume_fraction)
            : std::numeric_limits<double>::infinity();
    const auto global_min_volume_fraction_raw =
        globalMinDouble(local_min_volume_fraction, comm);
    const auto global_min_volume_fraction =
        std::isfinite(global_min_volume_fraction_raw)
            ? static_cast<svmp::FE::Real>(global_min_volume_fraction_raw)
            : svmp::FE::Real{0.0};
    const auto local_max_volume_fraction =
        active_summary.active_volume_regions > 0u
            ? static_cast<double>(active_summary.max_volume_fraction)
            : -std::numeric_limits<double>::infinity();
    const auto global_max_volume_fraction_raw =
        globalMaxDouble(local_max_volume_fraction, comm);
    const auto global_max_volume_fraction =
        std::isfinite(global_max_volume_fraction_raw)
            ? static_cast<svmp::FE::Real>(global_max_volume_fraction_raw)
            : svmp::FE::Real{0.0};
    const auto global_zero_scale_count =
        globalSumSize(facet_scale_summary.zero_scale_count, comm);
    const auto global_nonfinite_scale_count =
        globalSumSize(facet_scale_summary.nonfinite_scale_count, comm);
    const auto global_capped_scale_count =
        globalSumSize(facet_scale_summary.capped_scale_count, comm);
    const auto local_min_scale =
        facet_scale_summary.metadata_count > 0u
            ? static_cast<double>(facet_scale_summary.min_scale)
            : std::numeric_limits<double>::infinity();
    const auto global_min_scale_raw = globalMinDouble(local_min_scale, comm);
    const auto global_min_scale =
        std::isfinite(global_min_scale_raw)
            ? static_cast<svmp::FE::Real>(global_min_scale_raw)
            : svmp::FE::Real{0.0};
    const auto local_max_scale =
        facet_scale_summary.metadata_count > 0u
            ? static_cast<double>(facet_scale_summary.max_scale)
            : -std::numeric_limits<double>::infinity();
    const auto global_max_scale_raw = globalMaxDouble(local_max_scale, comm);
    const auto global_max_scale =
        std::isfinite(global_max_scale_raw)
            ? static_cast<svmp::FE::Real>(global_max_scale_raw)
            : svmp::FE::Real{0.0};
    const auto local_scale_sum =
        static_cast<double>(facet_scale_summary.mean_scale) *
        static_cast<double>(facet_scale_summary.metadata_count);
    const auto global_scale_sum = globalSumDouble(local_scale_sum, comm);
    const auto global_mean_scale =
        global_cut_adjacent_metadata > 0u
            ? static_cast<svmp::FE::Real>(
                  global_scale_sum /
                  static_cast<double>(global_cut_adjacent_metadata))
            : svmp::FE::Real{0.0};
    const auto memory = readProcessMemorySnapshot();
    report.process_vm_kb = memory.vm_kb;
    report.process_rss_kb = memory.rss_kb;
    report.basis_cache_entries = svmp::FE::basis::BasisCache::instance().size();
    const bool high_order_geometry =
        options.geometry_mode ==
        svmp::FE::level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    const bool refreshed_frozen_tangent =
        options.geometry_tangent_policy ==
        svmp::FE::level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature;
    const bool high_order_refreshed_frozen_tangent =
        high_order_geometry && refreshed_frozen_tangent;
    application::core::oopCout()
        << "[svMultiPhysics::Application] Active-domain cut context"
        << " diagnostic=cut_context_rebuild"
        << " provenance=" << (provenance != nullptr ? provenance : "unknown")
        << " solution_source="
        << (solution_source != nullptr ? solution_source : "unknown")
        << " marker="
        << result.interface_marker << " field='" << request.level_set_field_name
        << "' domain_id='" << request.domain_id
        << "' active_side=" << activeSideName(request.active_side)
        << " retained_volume_sides=active_only"
        << " interface_contract=one_sided_embedded"
        << " generated_interface_geometry="
        << svmp::FE::level_set::generatedInterfaceGeometryModeName(
               options.geometry_mode)
        << " implicit_cut_quadrature_backend="
        << svmp::FE::level_set::implicitCutQuadratureBackendName(
               options.implicit_cut_quadrature_backend)
        << " selected_implicit_cut_quadrature_backend_counts="
        << formatImplicitCutBackendCounts(global_selected_backend_counts)
        << " implicit_cut_backend_qualification_counts="
        << formatImplicitCutBackendQualificationCounts(
               global_backend_qualification_counts)
        << " required_implicit_cut_backend_qualification="
        << (options.require_production_qualified_implicit_cut_backend
                ? "ProductionQualified"
                : "none")
        << " implicit_cut_backend_seconds=" << backend_timing.local
        << " implicit_cut_backend_seconds_min=" << backend_timing.min
        << " implicit_cut_backend_seconds_mean=" << backend_timing.mean
        << " implicit_cut_backend_seconds_max=" << backend_timing.max
        << " implicit_cut_backend_internal_seconds="
        << result.backend_elapsed_seconds
        << " implicit_cut_backend_internal_seconds_total="
        << global_backend_elapsed_seconds
        << " generated_cell_cache_hits="
        << global_generated_cell_cache_hits
        << " generated_cell_cache_misses="
        << global_generated_cell_cache_misses
        << " generated_domain_cache_hits="
        << global_generated_domain_cache_hits
        << " linear_full_cell_fast_path_cells="
        << global_linear_full_cell_fast_path_cells
        << " implicit_cut_fallback_policy="
        << svmp::FE::level_set::implicitCutFallbackPolicyName(
               options.implicit_cut_fallback_policy)
        << " geometry_tangent_policy="
        << svmp::FE::level_set::geometryTangentPolicyName(
               options.geometry_tangent_policy)
        << " geometry_tangent_warning="
        << (high_order_refreshed_frozen_tangent
                ? "quadrature_sensitivities_omitted"
                : "none")
        << " implicit_cut_root_tolerance="
        << options.implicit_cut_root_tolerance
        << " implicit_cut_root_coordinate_tolerance="
        << options.implicit_cut_root_coordinate_tolerance
        << " implicit_cut_root_max_iterations="
        << options.implicit_cut_root_max_iterations
        << " implicit_cut_max_subdivision_depth="
        << options.implicit_cut_max_subdivision_depth
        << " quadrature_order=" << options.quadrature_order
        << " interface_quadrature_order="
        << options.interface_quadrature_order
        << " volume_quadrature_order="
        << options.volume_quadrature_order
        << " achieved_interface_quadrature_order="
        << result.achieved_interface_quadrature_order
        << " achieved_volume_quadrature_order="
        << result.achieved_volume_quadrature_order
        << " implicit_cut_fallback_cells="
        << global_implicit_cut_fallback_cells
        << " backend_volume_quadrature_point_count="
        << global_backend_volume_quadrature_points
        << " backend_interface_quadrature_point_count="
        << global_backend_interface_quadrature_points
        << " backend_total_quadrature_point_count="
        << (global_backend_volume_quadrature_points +
            global_backend_interface_quadrature_points)
        << " allow_corner_linearized_geometry="
        << (request.allow_corner_linearized_geometry ? "true" : "false")
        << " isovalue=" << request.isovalue
        << " cut_context_revision=" << result.value_revision
        << " cut_context_topology_key=" << topology_key
        << " active_cut_request_policy_key=" << report.request_policy_key
        << " quadrature_policy_key="
        << domain_request.quadrature_policy_key
        << " source_layout_revision="
        << domain_request.source.layout_revision
        << " source_value_revision="
        << domain_request.source.value_revision
        << " mesh_geometry_revision="
        << domain_request.mesh_geometry_revision
        << " mesh_topology_revision="
        << domain_request.mesh_topology_revision
        << " ownership_revision="
        << domain_request.ownership_revision
        << " cell_count=" << global_cell_count
        << " corner_linearized_cells="
        << global_corner_linearized_cells
        << " active_side_volume=" << global_active_volume
        << " active_side_volume_frame=reference"
        << " active_side_volume_local=" << active_volume
        << " active_side_physical_volume="
        << global_active_physical_volume
        << " active_side_physical_rule_count="
        << global_active_physical_volume_rules
        << " active_side_skipped_physical_rule_count="
        << global_active_skipped_physical_volume_rules
        << " active_side_raw_volume=" << global_raw_active_volume
        << " active_side_raw_volume_local=" << raw_active_volume
        << " interface_fragments=" << global_interface_fragments
        << " active_interface_fragments=" << global_active_interface_fragments
        << " interface_rule_count=" << global_active_interface_fragments
        << " interface_quadrature_point_count="
        << global_interface_quadrature_points
        << " domain_interface_quadrature_point_count="
        << global_interface_quadrature_points
        << " domain_volume_quadrature_point_count="
        << global_domain_volume_quadrature_points
        << " domain_total_quadrature_point_count="
        << global_domain_total_quadrature_points
        << " active_volume_regions="
        << global_active_volume_regions
        << " active_volume_rule_count="
        << global_active_volume_regions
        << " active_raw_volume_regions="
        << global_raw_active_volume_regions
        << " active_pruned_volume_regions="
        << global_pruned_volume_regions
        << " active_pruned_volume=" << global_pruned_volume
        << " generated_pruned_volume_rules="
        << global_generated_pruned_volume_rules
        << " generated_pruned_volume=" << global_generated_pruned_volume
        << " active_wet_cells=" << global_active_wet_cells
        << " active_cut_cells=" << global_cut_cells
        << " active_full_wet_cells=" << global_full_wet_cells
        << " active_full_dry_cells=" << global_full_dry_cells
        << " active_quadrature_points="
        << global_active_quadrature_points
        << " active_volume_quadrature_point_count="
        << global_active_quadrature_points
        << " active_empty_quadrature_regions="
        << global_empty_quadrature_regions
        << " active_nonfinite_measure_regions="
        << global_nonfinite_measure_regions
        << " active_negative_measure_regions="
        << global_negative_measure_regions
        << " active_min_volume_fraction="
        << global_min_volume_fraction
        << " active_max_volume_fraction="
        << global_max_volume_fraction
        << " cut_adjacent_facets=" << global_cut_adjacent_facets
        << " cut_adjacent_metadata="
        << global_cut_adjacent_metadata
        << " cut_adjacent_zero_scale="
        << global_zero_scale_count
        << " cut_adjacent_nonfinite_scale="
        << global_nonfinite_scale_count
        << " cut_adjacent_capped_scale="
        << global_capped_scale_count
        << " cut_adjacent_min_scale="
        << global_min_scale
        << " cut_adjacent_max_scale="
        << global_max_scale
        << " cut_adjacent_mean_scale="
        << global_mean_scale
        << " process_vm_kb=" << memory.vm_kb
        << " process_rss_kb=" << memory.rss_kb
        << " basis_cache_entries="
        << svmp::FE::basis::BasisCache::instance().size()
        << " negative_volume=" << global_negative_volume
        << " negative_reference_volume=" << global_negative_volume
        << " negative_physical_volume=" << global_negative_physical_volume
        << " negative_skipped_physical_rule_count="
        << global_negative_skipped_physical_volume_rules
        << " negative_volume_local=" << summary.negative_volume_measure
        << " positive_volume=" << global_positive_volume
        << " positive_reference_volume=" << global_positive_volume
        << " positive_physical_volume=" << global_positive_physical_volume
        << " positive_skipped_physical_rule_count="
        << global_positive_skipped_physical_volume_rules
        << " positive_volume_local=" << summary.positive_volume_measure << std::endl;
  }

  sim.fe_system->setCutIntegrationContext(std::move(context));
  sim.fe_system->rebuildConstraintState();
  application::core::oopCout()
      << "[svMultiPhysics::Application] Active pressure support constraint refresh"
      << " diagnostic=active_pressure_constraint_refresh"
      << " provenance=" << (provenance != nullptr ? provenance : "unknown")
      << " solution_source="
      << (solution_source != nullptr ? solution_source : "unknown")
      << " synchronized_level_set_fields="
      << synchronized_level_set_fields
      << " support_source=retained_cut_context"
      << " constraints=" << sim.fe_system->constraints().numConstraints()
      << std::endl;
  return report;
}

struct ActiveCutContextRefreshSignature {
  enum class SolutionSignatureKind : std::uint8_t {
    LevelSetFieldByteHash = 1,
    VectorValueRevision = 2
  };

  std::uint64_t request_policy_key{0};
  std::uint64_t mesh_geometry_revision{0};
  std::uint64_t mesh_topology_revision{0};
  std::uint64_t mesh_ownership_revision{0};
  std::uint64_t mesh_field_layout_revision{0};
  SolutionSignatureKind solution_signature_kind{
      SolutionSignatureKind::LevelSetFieldByteHash};
  std::uint64_t solution_hash{0};
  std::size_t solution_size{0};

  [[nodiscard]] bool operator==(
      const ActiveCutContextRefreshSignature& other) const noexcept
  {
    return request_policy_key == other.request_policy_key &&
           mesh_geometry_revision == other.mesh_geometry_revision &&
           mesh_topology_revision == other.mesh_topology_revision &&
           mesh_ownership_revision == other.mesh_ownership_revision &&
           mesh_field_layout_revision == other.mesh_field_layout_revision &&
           solution_signature_kind == other.solution_signature_kind &&
           solution_hash == other.solution_hash &&
           solution_size == other.solution_size;
  }
};

struct ActiveCutContextRefreshCache {
  std::optional<ActiveCutContextRefreshSignature> last_signature{};
  std::optional<ActiveCutContextRefreshSignature> last_vector_signature{};
};

bool activeCutLevelSetFieldLayoutIsAvailable(
    const svmp::FE::systems::FESystem& system,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::size_t solution_size)
{
  std::set<std::string> field_names;
  for (const auto& request : requests) {
    field_names.insert(request.level_set_field_name);
  }
  if (field_names.empty()) {
    return false;
  }

  for (const auto& field_name : field_names) {
    const auto field = system.findFieldByName(field_name);
    if (field == svmp::FE::INVALID_FIELD_ID) {
      return false;
    }
    const auto field_offset = system.fieldDofOffset(field);
    const auto n_field_dofs = system.fieldDofHandler(field).getNumDofs();
    if (field_offset < 0 ||
        n_field_dofs < 0 ||
        static_cast<std::size_t>(field_offset + n_field_dofs) >
            solution_size) {
      return false;
    }
  }
  return true;
}

std::optional<std::uint64_t> hashActiveCutLevelSetFieldBytes(
    const svmp::FE::systems::FESystem& system,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::span<const svmp::FE::Real> values)
{
  std::set<std::string> field_names;
  for (const auto& request : requests) {
    field_names.insert(request.level_set_field_name);
  }
  if (field_names.empty()) {
    return std::nullopt;
  }

  std::uint64_t h = kCutContextHashOffset;
  mixCutContextHash(h, static_cast<std::uint64_t>(field_names.size()));
  for (const auto& field_name : field_names) {
    const auto field = system.findFieldByName(field_name);
    if (field == svmp::FE::INVALID_FIELD_ID) {
      return std::nullopt;
    }
    const auto field_offset = system.fieldDofOffset(field);
    const auto n_field_dofs = system.fieldDofHandler(field).getNumDofs();
    if (field_offset < 0 ||
        n_field_dofs < 0 ||
        static_cast<std::size_t>(field_offset + n_field_dofs) > values.size()) {
      return std::nullopt;
    }

    mixCutContextHash(h, field_name);
    mixCutContextHash(h, static_cast<std::uint64_t>(field));
    mixCutContextHash(h, static_cast<std::uint64_t>(field_offset));
    mixCutContextHash(h, static_cast<std::uint64_t>(n_field_dofs));
    for (svmp::FE::GlobalIndex local_dof = 0; local_dof < n_field_dofs; ++local_dof) {
      const auto value =
          values[static_cast<std::size_t>(field_offset + local_dof)];
      const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
      for (std::size_t i = 0; i < sizeof(value); ++i) {
        mixCutContextHash(h, static_cast<std::uint64_t>(bytes[i]));
      }
    }
  }
  return h;
}

std::optional<ActiveCutContextRefreshSignature> activeCutContextRefreshSignature(
    const application::core::SimulationComponents& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::span<const svmp::FE::Real> fe_solution)
{
  if (!sim.fe_system || requests.empty()) {
    return std::nullopt;
  }
  const auto solution_hash =
      hashActiveCutLevelSetFieldBytes(*sim.fe_system, requests, fe_solution);
  if (!solution_hash.has_value()) {
    return std::nullopt;
  }
  const auto& mesh = sim.fe_system->meshAccess();
  ActiveCutContextRefreshSignature signature;
  signature.request_policy_key = activeCutVolumeRequestPolicyKey(requests);
  signature.mesh_geometry_revision = mesh.geometryRevision();
  signature.mesh_topology_revision = mesh.topologyRevision();
  signature.mesh_ownership_revision = mesh.ownershipRevision();
  signature.mesh_field_layout_revision = mesh.fieldLayoutRevision();
  signature.solution_signature_kind =
      ActiveCutContextRefreshSignature::SolutionSignatureKind::
          LevelSetFieldByteHash;
  signature.solution_size = fe_solution.size();
  signature.solution_hash = *solution_hash;
  return signature;
}

std::optional<ActiveCutContextRefreshSignature> activeCutContextRefreshSignature(
    const application::core::SimulationComponents& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::backends::GenericVector& fe_solution)
{
  if (!sim.fe_system || requests.empty()) {
    return std::nullopt;
  }
  if (!activeCutLevelSetFieldLayoutIsAvailable(
          *sim.fe_system,
          requests,
          static_cast<std::size_t>(fe_solution.size()))) {
    return std::nullopt;
  }

  const auto& mesh = sim.fe_system->meshAccess();
  ActiveCutContextRefreshSignature signature;
  signature.request_policy_key = activeCutVolumeRequestPolicyKey(requests);
  signature.mesh_geometry_revision = mesh.geometryRevision();
  signature.mesh_topology_revision = mesh.topologyRevision();
  signature.mesh_ownership_revision = mesh.ownershipRevision();
  signature.mesh_field_layout_revision = mesh.fieldLayoutRevision();
  signature.solution_signature_kind =
      ActiveCutContextRefreshSignature::SolutionSignatureKind::
          VectorValueRevision;
  signature.solution_size = static_cast<std::size_t>(fe_solution.size());
  signature.solution_hash = fe_solution.valueRevision();
  return signature;
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
  ActiveCutContextRefreshReport skipped_report{};
  if (requests.empty() || !sim.fe_system) {
    return skipped_report;
  }
  skipped_report.request_policy_key = activeCutVolumeRequestPolicyKey(requests);
  const auto comm = svmp::MeshComm::world();
  const auto signature =
      activeCutContextRefreshSignature(sim, requests, fe_solution);
  const bool local_can_skip =
      signature.has_value() &&
      cache.last_signature.has_value() &&
      *signature == *cache.last_signature &&
      sim.fe_system->cutIntegrationContext() != nullptr;
  if (!globalAnyBool(!local_can_skip, comm)) {
    return skipped_report;
  }

  auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      params,
      fe_solution,
      lifecycle,
      provenance,
      solution_source);
  if (report.refreshed) {
    const auto refreshed_signature =
        activeCutContextRefreshSignature(sim, requests, fe_solution);
    if (refreshed_signature.has_value()) {
      cache.last_signature = *refreshed_signature;
    } else if (signature.has_value()) {
      cache.last_signature = *signature;
    }
    cache.last_vector_signature.reset();
  }
  return report;
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContext(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    svmp::FE::backends::GenericVector& solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    const char* provenance)
{
  const auto fe_solution = gatherFeOrderedSolution(solution);
  return refreshActiveCutIntegrationContextFromSolution(
      sim,
      params,
      std::span<const svmp::FE::Real>(fe_solution.data(), fe_solution.size()),
      lifecycle,
      provenance,
      "fe_vector");
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCachedFromVector(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    svmp::FE::backends::GenericVector& solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance,
    const char* solution_source)
{
  const auto requests = activeCutVolumeRequests(params);
  ActiveCutContextRefreshReport skipped_report{};
  if (requests.empty() || !sim.fe_system) {
    return skipped_report;
  }
  skipped_report.request_policy_key = activeCutVolumeRequestPolicyKey(requests);
  const auto comm = svmp::MeshComm::world();
  const auto vector_signature =
      activeCutContextRefreshSignature(sim, requests, solution);
  const bool local_vector_can_skip =
      vector_signature.has_value() &&
      cache.last_vector_signature.has_value() &&
      *vector_signature == *cache.last_vector_signature &&
      sim.fe_system->cutIntegrationContext() != nullptr;
  if (!globalAnyBool(!local_vector_can_skip, comm)) {
    return skipped_report;
  }

  const auto fe_solution = gatherFeOrderedSolution(solution);
  const auto signature = activeCutContextRefreshSignature(
      sim,
      requests,
      std::span<const svmp::FE::Real>(fe_solution.data(), fe_solution.size()));
  const bool local_fe_can_skip =
      signature.has_value() &&
      cache.last_signature.has_value() &&
      *signature == *cache.last_signature &&
      sim.fe_system->cutIntegrationContext() != nullptr;
  if (!globalAnyBool(!local_fe_can_skip, comm)) {
    if (vector_signature.has_value()) {
      cache.last_vector_signature = *vector_signature;
    }
    return skipped_report;
  }

  auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      params,
      std::span<const svmp::FE::Real>(fe_solution.data(), fe_solution.size()),
      lifecycle,
      provenance,
      solution_source);
  if (report.refreshed) {
    const auto refreshed_signature = activeCutContextRefreshSignature(
        sim,
        requests,
        std::span<const svmp::FE::Real>(fe_solution.data(), fe_solution.size()));
    if (refreshed_signature.has_value()) {
      cache.last_signature = *refreshed_signature;
    } else if (signature.has_value()) {
      cache.last_signature = *signature;
    }
    const auto refreshed_vector_signature =
        activeCutContextRefreshSignature(sim, requests, solution);
    if (refreshed_vector_signature.has_value()) {
      cache.last_vector_signature = *refreshed_vector_signature;
    } else if (vector_signature.has_value()) {
      cache.last_vector_signature = *vector_signature;
    }
  }
  return report;
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCached(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    svmp::FE::backends::GenericVector& solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance)
{
  return refreshActiveCutIntegrationContextCachedFromVector(
      sim,
      params,
      solution,
      lifecycle,
      cache,
      provenance,
      "fe_vector");
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContext(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    const char* provenance)
{
  if (!sim.fe_system) {
    return {};
  }
  const char* solution_source =
      state.u_vector != nullptr ? "state_vector_fe_ordered"
                                : "state_span_assumed_fe_ordered";
  const auto fe_solution = gatherFeOrderedSolution(state);
  return refreshActiveCutIntegrationContextFromSolution(
      sim,
      params,
      std::span<const svmp::FE::Real>(fe_solution.data(), fe_solution.size()),
      lifecycle,
      provenance,
      solution_source);
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCached(
    application::core::SimulationComponents& sim,
    const Parameters& params,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance)
{
  if (!sim.fe_system) {
    return {};
  }
  const char* solution_source =
      state.u_vector != nullptr ? "state_vector_fe_ordered"
                                : "state_span_assumed_fe_ordered";
  if (state.u_vector != nullptr) {
    auto* solution =
        const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
    return refreshActiveCutIntegrationContextCachedFromVector(
        sim,
        params,
        *solution,
        lifecycle,
        cache,
        provenance,
        solution_source);
  }
  const auto fe_solution = gatherFeOrderedSolution(state);
  return refreshActiveCutIntegrationContextFromSolutionCached(
      sim,
      params,
      std::span<const svmp::FE::Real>(fe_solution.data(), fe_solution.size()),
      lifecycle,
      cache,
      provenance,
      solution_source);
}

class ZeroTimeDerivativeIntegrator final : public svmp::FE::systems::TimeIntegrator {
public:
  [[nodiscard]] std::string name() const override { return "ZeroTimeDerivative"; }
  [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 2; }

  [[nodiscard]] svmp::FE::assembly::TimeIntegrationContext buildContext(
      int max_time_derivative_order, const svmp::FE::systems::SystemStateView& /*state*/) const override
  {
    svmp::FE::assembly::TimeIntegrationContext ctx;
    ctx.integrator_name = name();
    if (max_time_derivative_order <= 0) {
      return ctx;
    }

    if (max_time_derivative_order >= 1) {
      svmp::FE::assembly::TimeDerivativeStencil s;
      s.order = 1;
      s.a.assign(1, static_cast<svmp::FE::Real>(0.0));
      ctx.dt1 = std::move(s);
    }
    if (max_time_derivative_order >= 2) {
      svmp::FE::assembly::TimeDerivativeStencil s;
      s.order = 2;
      s.a.assign(1, static_cast<svmp::FE::Real>(0.0));
      ctx.dt2 = std::move(s);
    }

    return ctx;
  }
};

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

void ApplicationDriver::run(const std::string& xml_file)
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
  runWithParameters(params);
}

void ApplicationDriver::runWithParameters(const Parameters& params)
{
  const auto comm = svmp::MeshComm::world();

  // Auto-configure OpenMP threads: hardware_cores / MPI_ranks_per_node.
  // Respects OMP_NUM_THREADS if the user has set it explicitly.
  configureOpenMPThreads(comm);

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

  SimulationBuilder builder(params);
  auto sim = builder.build();

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
  if (const auto* eq = primary_solver_equation(params)) {
    if (eq->max_iterations.defined()) {
      newton_opts.max_iterations = eq->max_iterations.value();
    }
    if (eq->min_iterations.defined()) {
      newton_opts.min_iterations = std::max(0, eq->min_iterations.value());
    }
    if (eq->tolerance.defined()) {
      const double tol = eq->tolerance.value();
      if (tol > 0.0) {
        // Legacy semantics: <Add_equation><Tolerance> is a *relative* tolerance.
        newton_opts.rel_tolerance = tol;
        // Preserve legacy convergence behavior for warm-started cases by
        // allowing the same XML tolerance to satisfy either absolute or
        // relative convergence.
        newton_opts.abs_tolerance = tol;
      }
    }
  }

  // Use the unified "equations" operator tag (same as transient).
  newton_opts.residual_op = "equations";
  newton_opts.jacobian_op = "equations";
  const auto steady_active_cut_requests = activeCutVolumeRequests(params);
  // Generated active-domain cuts can make full Newton trial states invalid
  // before line search has a chance to reject them.
  newton_opts.use_line_search =
      parseBoolEnv("SVMP_NEWTON_LINE_SEARCH",
                   !steady_active_cut_requests.empty());
  applyNewtonLineSearchEnvOptions(newton_opts);
  newton_opts.accept_inexact_linear_solutions =
      parseBoolEnv("SVMP_NEWTON_ACCEPT_INEXACT_LINEAR", false);

  // Modified Newton: reuse Jacobian across multiple iterations.
  // Period 1 = full Newton (default), 2 = rebuild every 2nd iteration, etc.
  if (const char* jrp = std::getenv("SVMP_JACOBIAN_REBUILD_PERIOD")) {
    newton_opts.jacobian_rebuild_period = std::atoi(jrp);
  }

  auto cut_lifecycle =
      std::make_shared<svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle>();
  auto cut_topology_key = std::make_shared<std::optional<std::uint64_t>>();
  if (hasHighOrderGeneratedInterfaceGeometry(steady_active_cut_requests)) {
    newton_opts.jacobian_check_geometry_mode =
        svmp::FE::timestepping::JacobianCheckGeometryMode::RefreshedGeometry;
    newton_opts.jacobian_check_geometry_tangent_policy =
        highOrderGeometryTangentPolicySummary(steady_active_cut_requests);
  }
  using StateSyncPoint =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  newton_opts.synchronize_state =
      [&, cut_lifecycle, cut_topology_key](
          const svmp::FE::systems::SystemStateView& state,
          StateSyncPoint point) {
        const auto report = refreshActiveCutIntegrationContext(
            sim, params, state, *cut_lifecycle, stateSyncPointName(point));
        logCutTopologyChange(report, point, *cut_topology_key, "steady");
      };

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

  (void)refreshActiveCutIntegrationContext(
      sim, params, sim.time_history->u(), *cut_lifecycle, "steady_initial");

  const double solve_time = sim.time_history->time();
  oopCout() << "[svMultiPhysics::Application] Steady solve: time=" << solve_time
            << " newton(max_it=" << newton_opts.max_iterations << ", min_it=" << newton_opts.min_iterations
            << ", abs_tol=" << newton_opts.abs_tolerance
            << ", rel_tol=" << newton_opts.rel_tolerance << ")" << std::endl;

  const auto report = newton.solveStep(transient, *sim.linear_solver, solve_time, *sim.time_history, workspace);
  oopCout() << "[svMultiPhysics::Application] Steady Newton: converged=" << report.converged
            << " iterations=" << report.iterations << " residual_norm=" << report.residual_norm
            << " field_residual_norm=" << report.field_residual_norm
            << " auxiliary_residual_norm=" << report.auxiliary_residual_norm << std::endl;

  if (!report.converged) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Steady solve did not converge (Newton reached max iterations). "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  sim.fe_system->commitTimeStep();
  (void)refreshActiveCutIntegrationContext(
      sim, params, sim.time_history->u(), *cut_lifecycle, "steady_accepted");
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

  svmp::FE::timestepping::TimeLoopOptions opts{};
  opts.t0 = 0.0;
  opts.dt = dt;
  opts.t_end = static_cast<double>(num_steps) * dt;
  opts.max_steps = num_steps;
  opts.scheme = svmp::FE::timestepping::SchemeKind::GeneralizedAlpha;
  if (params.general_simulation_parameters.spectral_radius_of_infinite_time_step.defined()) {
    opts.generalized_alpha_rho_inf = params.general_simulation_parameters.spectral_radius_of_infinite_time_step.value();
  }

  if (const auto* eq = primary_solver_equation(params)) {
    if (eq->max_iterations.defined()) {
      opts.newton.max_iterations = eq->max_iterations.value();
    }
    if (eq->min_iterations.defined()) {
      opts.newton.min_iterations = std::max(0, eq->min_iterations.value());
    }
    if (eq->tolerance.defined()) {
      const double tol = eq->tolerance.value();
      if (tol > 0.0) {
        // Legacy semantics: <Add_equation><Tolerance> is a *relative* tolerance.
        opts.newton.rel_tolerance = tol;
        // Preserve legacy convergence behavior for warm-started cases by
        // allowing the same XML tolerance to satisfy either absolute or
        // relative convergence.
        opts.newton.abs_tolerance = tol;
      }
    }
  }
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

  const auto transient_active_cut_requests = activeCutVolumeRequests(params);
  const auto level_set_advection_velocity =
      levelSetAdvectionVelocityRequests(params);
  opts.initialize_first_order_rate_from_pde =
      parseBoolEnv("SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT",
                   transient_active_cut_requests.empty() &&
                       level_set_advection_velocity.empty());
  // Generated level-set active domains need line search to reject trial states
  // that temporarily erase or severely distort the retained wet side.
  opts.newton.use_line_search =
      parseBoolEnv("SVMP_NEWTON_LINE_SEARCH",
                   !transient_active_cut_requests.empty() ||
                       !level_set_advection_velocity.empty());
  applyNewtonLineSearchEnvOptions(opts.newton);
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

  if (parseBoolEnv("SVMP_TIMELOOP_ADAPTIVE", false)) {
    svmp::FE::timestepping::SimpleStepControllerOptions controller_opts{};
    controller_opts.min_dt = parseDoubleEnv("SVMP_TIMELOOP_MIN_DT", dt * 0.0625);
    controller_opts.max_dt = parseDoubleEnv("SVMP_TIMELOOP_MAX_DT", dt);
    controller_opts.max_retries = std::max(0, parseIntEnv("SVMP_TIMELOOP_MAX_RETRIES", 8));
    controller_opts.decrease_factor = parseDoubleEnv("SVMP_TIMELOOP_DECREASE_FACTOR", 0.5);
    controller_opts.increase_factor = parseDoubleEnv("SVMP_TIMELOOP_INCREASE_FACTOR", 1.0);
    controller_opts.target_newton_iterations =
        std::max(1, parseIntEnv("SVMP_TIMELOOP_TARGET_NEWTON_ITERATIONS", 6));
    opts.step_controller =
        std::make_shared<svmp::FE::timestepping::SimpleStepController>(controller_opts);
    const int max_steps_multiplier =
        std::max(1, parseIntEnv("SVMP_TIMELOOP_MAX_STEPS_MULTIPLIER", 8));
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

  oopCout() << "[svMultiPhysics::Application] Transient solve: t0=" << opts.t0 << " dt=" << opts.dt
            << " t_end=" << opts.t_end << " max_steps=" << opts.max_steps
            << " scheme=GeneralizedAlpha rho_inf=" << opts.generalized_alpha_rho_inf
            << " pde_udot_init=" << (opts.initialize_first_order_rate_from_pde ? 1 : 0)
            << " newton(max_it=" << opts.newton.max_iterations << ", min_it=" << opts.newton.min_iterations
            << ", abs_tol=" << opts.newton.abs_tolerance
            << ", rel_tol=" << opts.newton.rel_tolerance << ")" << std::endl;

  // Ensure time-history vectors use the same backend layout as the solver workspace.
  oopCout() << "[svMultiPhysics::Application] Transient: repacking TimeHistory for backend layout." << std::endl;
  traceStateVectorFields(*sim.fe_system, sim.time_history->u(), "before_transient_repack");
  sim.time_history->repack(*sim.backend);
  traceStateVectorFields(*sim.fe_system, sim.time_history->u(), "after_transient_repack");
  oopCout() << "[svMultiPhysics::Application] Transient: TimeHistory repacked." << std::endl;

  auto bdf1 = std::make_shared<const svmp::FE::systems::BDFIntegrator>(1);
  svmp::FE::systems::TransientSystem transient(*sim.fe_system, std::move(bdf1));
  auto level_set_maintenance = levelSetMaintenanceRequests(params);
  auto curvature_projection_cache =
      std::make_shared<CurvatureProjectionCache>();
  initializeLevelSetMaintenanceTargets(sim, level_set_maintenance);
  (void)projectLevelSetCurvatureFieldsFromState(
      sim,
      stateViewForHistory(*sim.time_history),
      level_set_maintenance,
      sim.time_history->stepIndex(),
      "initial",
      /*honor_cadence=*/false,
      curvature_projection_cache.get(),
      /*reuse_cached_on_projection_failure=*/false);
  logLevelSetMaintenanceCoverageDiagnostics(
      activeCutVolumeRequests(params),
      level_set_maintenance);

  svmp::FE::timestepping::TimeLoopCallbacks callbacks{};
  callbacks.on_step_start = [&](const svmp::FE::timestepping::TimeHistory& h) {
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_start step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << std::endl;
  };
  auto cut_lifecycle =
      std::make_shared<svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle>();
  auto cut_refresh_cache =
      std::make_shared<ActiveCutContextRefreshCache>();
  auto cut_topology_key = std::make_shared<std::optional<std::uint64_t>>();
  std::map<std::string, svmp::FE::Real> initial_wet_volume_by_key;
  const bool high_order_cut_geometry =
      hasHighOrderGeneratedInterfaceGeometry(transient_active_cut_requests);
  if (high_order_cut_geometry) {
    opts.newton.jacobian_check_geometry_mode =
        svmp::FE::timestepping::JacobianCheckGeometryMode::RefreshedGeometry;
    opts.newton.jacobian_check_geometry_tangent_policy =
        highOrderGeometryTangentPolicySummary(transient_active_cut_requests);
  }
  const auto expected_cut_request_policy_key =
      activeCutVolumeRequestPolicyKey(transient_active_cut_requests);
  using TransientStateSyncPoint =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  opts.newton.synchronize_state =
      [&, cut_lifecycle, cut_refresh_cache, cut_topology_key,
       curvature_projection_cache, high_order_cut_geometry](
          const svmp::FE::systems::SystemStateView& state,
          TransientStateSyncPoint point) {
        (void)projectLevelSetCurvatureFieldsFromState(
            sim,
            state,
            level_set_maintenance,
            -1,
            stateSyncPointName(point),
            /*honor_cadence=*/false,
            curvature_projection_cache.get(),
            /*reuse_cached_on_projection_failure=*/true);
        (void)updateLevelSetAdvectionVelocitiesFromState(
            sim, state, level_set_advection_velocity);
        if (!opts.newton.use_line_search &&
            point == TransientStateSyncPoint::AcceptedNonlinearState &&
            !high_order_cut_geometry) {
          return;
        }
        const auto report = refreshActiveCutIntegrationContextCached(
            sim, params, state, *cut_lifecycle, *cut_refresh_cache,
            stateSyncPointName(point));
        if (report.refreshed &&
            report.request_policy_key != expected_cut_request_policy_key) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Active cut request policy changed during transient Newton synchronization.");
        }
        logCutTopologyChange(report, point, *cut_topology_key, "transient");
      };
  callbacks.on_before_physics_solve =
      [&](svmp::FE::timestepping::TimeHistory& h, double /*solve_time*/, double /*dt*/) {
        (void)projectLevelSetCurvatureFieldsFromState(
            sim,
            stateViewForHistory(h),
            level_set_maintenance,
            h.stepIndex(),
            "before_physics_solve",
            /*honor_cadence=*/true,
            curvature_projection_cache.get(),
            /*reuse_cached_on_projection_failure=*/true);
        (void)updateLevelSetAdvectionVelocities(
            sim, h, level_set_advection_velocity);
        (void)refreshActiveCutIntegrationContextCached(
            sim, params, h.u(), *cut_lifecycle, *cut_refresh_cache,
            "before_physics_solve");
        return true;
      };
  callbacks.on_nonlinear_done = [&](const svmp::FE::timestepping::TimeHistory& h,
                                   const svmp::FE::timestepping::NewtonReport& nr) {
    oopCout() << "[svMultiPhysics::Application] TimeLoop: nonlinear_done step=" << h.stepIndex()
              << " time=" << h.time() << " converged=" << nr.converged
              << " iters=" << nr.iterations << " ||r||=" << nr.residual_norm
              << " ||r_field||=" << nr.field_residual_norm
              << " ||r_aux||=" << nr.auxiliary_residual_norm
              << " (linear: converged=" << nr.linear.converged
              << " iters=" << nr.linear.iterations
              << " rel=" << nr.linear.relative_residual << ")" << std::endl;
  };
  double vtk_total_time = 0.0;
  callbacks.on_step_accepted = [&](svmp::FE::timestepping::TimeHistory& h) {
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_accepted step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << std::endl;
    const bool level_set_maintenance_changed =
        applyLevelSetMaintenance(sim, h, level_set_maintenance);
    (void)projectLevelSetCurvatureFieldsFromState(
        sim,
        stateViewForHistory(h),
        level_set_maintenance,
        h.stepIndex(),
        "accepted_step",
        /*honor_cadence=*/true,
        curvature_projection_cache.get(),
        /*reuse_cached_on_projection_failure=*/false);
    (void)updateLevelSetAdvectionVelocities(
        sim, h, level_set_advection_velocity);
    const auto cut_report = refreshActiveCutIntegrationContextCached(
        sim, params, h.u(), *cut_lifecycle, *cut_refresh_cache,
        "accepted_step");
    if (level_set_maintenance_changed && cut_report.refreshed) {
      oopCout()
          << "[svMultiPhysics::Application] Level-set maintenance refreshed cut context"
          << " step=" << h.stepIndex()
          << " cut_context_revision=" << cut_report.value_revision
          << " cell_count=" << cut_report.cell_count
          << " corner_linearized_cells="
          << cut_report.corner_linearized_cell_count
          << " active_cut_cells=" << cut_report.active_cut_cells
          << " active_quadrature_points="
          << cut_report.active_quadrature_points
          << " domain_interface_quadrature_point_count="
          << cut_report.domain_interface_quadrature_point_count
          << " domain_volume_quadrature_point_count="
          << cut_report.domain_volume_quadrature_point_count
          << " domain_total_quadrature_point_count="
          << cut_report.domain_total_quadrature_point_count
          << " backend_volume_quadrature_point_count="
          << cut_report.backend_volume_quadrature_point_count
          << " backend_interface_quadrature_point_count="
          << cut_report.backend_interface_quadrature_point_count
          << " backend_total_quadrature_point_count="
          << (cut_report.backend_volume_quadrature_point_count +
              cut_report.backend_interface_quadrature_point_count)
          << " backend_elapsed_seconds="
          << cut_report.backend_elapsed_seconds
          << " generated_cell_cache_hits="
          << cut_report.generated_cell_cache_hits
          << " generated_cell_cache_misses="
          << cut_report.generated_cell_cache_misses
          << " generated_domain_cache_hits="
          << cut_report.generated_domain_cache_hits
          << " process_vm_kb=" << cut_report.process_vm_kb
          << " process_rss_kb=" << cut_report.process_rss_kb
          << " basis_cache_entries=" << cut_report.basis_cache_entries
          << " negative_volume=" << cut_report.negative_volume
          << " negative_reference_volume=" << cut_report.negative_volume
          << " negative_physical_volume="
          << cut_report.negative_physical_volume
          << " positive_volume=" << cut_report.positive_volume
          << " positive_reference_volume=" << cut_report.positive_volume
          << " positive_physical_volume="
          << cut_report.positive_physical_volume
          << " cut_adjacent_facets=" << cut_report.cut_adjacent_facets
          << std::endl;
    }
    logWetVolumeDiagnostics(
        activeCutVolumeRequests(params),
        sim.fe_system->cutIntegrationContext(),
        sim.fe_system->meshAccess(),
        sim.primary_mesh ? sim.primary_mesh->n_cells() : 0u,
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
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_rejected step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << " reason=" << step_reject_reason_to_string(reason)
              << " (newton: converged=" << nr.converged << " iters=" << nr.iterations
              << " ||r||=" << nr.residual_norm
              << " ||r_field||=" << nr.field_residual_norm
              << " ||r_aux||=" << nr.auxiliary_residual_norm << ")" << std::endl;
  };
  callbacks.on_dt_updated = [&](double old_dt, double new_dt, int step_index, int attempt_index) {
    if (!oopTraceEnabled()) {
      return;
    }
    oopCout() << "[svMultiPhysics::Application] TimeLoop: dt_updated step=" << step_index
              << " attempt=" << attempt_index << " old_dt=" << old_dt << " new_dt=" << new_dt << std::endl;
  };

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

  const auto n_fields = sim.fe_system->fieldMap().numFields();
  double primary_field_seconds = 0.0;
  std::vector<std::pair<std::string, double>> field_timings;
  std::vector<std::pair<std::string, bool>> field_fast_paths;
  field_timings.reserve(n_fields);
  field_fast_paths.reserve(n_fields);
  for (std::size_t i = 0; i < n_fields; ++i) {
    const auto field_start = Clock::now();
    const auto field_id = static_cast<svmp::FE::FieldId>(i);
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

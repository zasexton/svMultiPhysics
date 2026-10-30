/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetGeometryService.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutGeometryMemoryReport.h"
#include "Assembly/CutIntegrationContext.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Basis/BasisCache.h"
#include "Core/DeterministicParallel.h"
#include "Core/LogRecord.h"
#include "Core/RankLog.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "LevelSet/LevelSetCutVolumeMeasures.h"
#include "LevelSet/LevelSetImplicitCutQuadratureBackend.h"
#include "Systems/CutIntegrationInvalidation.h"
#include "Systems/FESystem.h"
#include "Systems/SystemState.h"
#include "TimeStepping/TimeHistory.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace level_set {

namespace {

// The moved bodies keep the names of the driver's helpers.  Each
// communicator helper forwards to the FE/Systems collective that the
// driver's helper of the same name calls, so every rank issues the same MPI
// calls in the same order as before the move.
using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
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

using svmp::FE::systems::requireCollectivePhasePreparation;

svmp::FE::systems::CollectiveCommunicator activeFESystemCommunicator(
    const svmp::FE::systems::FESystem& system)
{
#if FE_HAS_MPI
  return svmp::FE::systems::CollectiveCommunicator(
      system.activeMpiCommunicator());
#else
  (void)system;
  return {};
#endif
}

double globalSumDouble(
    double local, const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveSumDouble(local, comm);
}

double globalMinDouble(
    double local, const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveMinDouble(local, comm);
}

double globalMaxDouble(
    double local, const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveMaxDouble(local, comm);
}

std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveMinMaxUint64(local, comm);
}

std::size_t globalSumSize(
    std::size_t local, const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveSumSize(local, comm);
}

bool globalAnyBool(
    bool local, const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveAnyTrue(local, comm);
}

svmp::FE::systems::CollectiveTimingStatistics reduceOutputTiming(
    double local, const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveTimingStatistics(local, comm);
}

svmp::FE::interfaces::FreeSurfaceGeometryOwnershipCollective
snapshotOwnershipCollective(
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::ownershipCollective(comm);
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::captureFeOrderedVectorCollectively(
      solution, comm);
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    const svmp::FE::systems::SystemStateView& state,
    const svmp::FE::systems::CollectiveCommunicator& comm)
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

// Host services (LevelSetGeometryHost) under the driver's names.
void validateEquationLevelCutVolumeConsumer(
    const LevelSetGeometryHost& sim,
    const svmp::FE::systems::FESystem& system,
    const ActiveCutVolumeRequest& request,
    int resolved_marker)
{
  if (sim.validate_cut_volume_consumer) {
    sim.validate_cut_volume_consumer(system, request, resolved_marker);
  }
}

bool generatedCutNeedsEveryGhostLayer(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& requests)
{
  return !sim.generated_cut_needs_every_ghost_layer ||
         sim.generated_cut_needs_every_ghost_layer(requests);
}

} // namespace

// Moved from Application/Core/ApplicationDriver.cpp with bodies, messages,
// environment names and collective order unchanged.

std::vector<int> communicatorWideBoundaryMarkers(
    const svmp::FE::assembly::IMeshAccess& mesh,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  auto markers = svmp::FE::interfaces::boundaryMarkers(mesh);

#if FE_HAS_MPI
  if (!comm.isParallel()) {
    return markers;
  }
  if (markers.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Too many local boundary markers for "
        "MPI active-cut refresh.");
  }

  const int local_count = static_cast<int>(markers.size());
  std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
  MPI_Allgather(&local_count,
                1,
                MPI_INT,
                counts.data(),
                1,
                MPI_INT,
                comm.native());

  std::vector<int> displacements(counts.size(), 0);
  std::size_t total_count = 0u;
  for (std::size_t rank = 0; rank < counts.size(); ++rank) {
    if (counts[rank] < 0 ||
        total_count >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                static_cast<std::size_t>(counts[rank])) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Communicator-wide boundary-marker "
          "union exceeds MPI count limits.");
    }
    displacements[rank] = static_cast<int>(total_count);
    total_count += static_cast<std::size_t>(counts[rank]);
  }

  std::vector<int> gathered(total_count, 0);
  MPI_Allgatherv(markers.empty() ? nullptr : markers.data(),
                 local_count,
                 MPI_INT,
                 gathered.empty() ? nullptr : gathered.data(),
                 counts.data(),
                 displacements.data(),
                 MPI_INT,
                 comm.native());
  std::sort(gathered.begin(), gathered.end());
  gathered.erase(std::unique(gathered.begin(), gathered.end()),
                 gathered.end());
  return gathered;
#else
  (void)comm;
  return markers;
#endif
}

namespace {

svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionSummary
ownedBoundaryIntersectionSummary(
    const svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionDomain&
        domain,
    const svmp::FE::assembly::IMeshAccess& mesh)
{
  svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionSummary summary;
  summary.interface_marker = domain.request().interface_marker;
  summary.boundary_marker = domain.request().boundary_marker;
  summary.intersection_marker = domain.marker();
  for (const auto& fragment : domain.fragments()) {
    const auto parent_cell =
        static_cast<svmp::FE::GlobalIndex>(fragment.parent_cell);
    if (parent_cell < 0 || parent_cell >= mesh.numCells() ||
        !mesh.isOwnedCell(parent_cell)) {
      continue;
    }
    ++summary.fragment_count;
    if (!fragment.active()) {
      ++summary.skipped_fragment_count;
      continue;
    }
    ++summary.active_fragment_count;
    summary.quadrature_point_count += fragment.quadrature_points.size();
    summary.measure += fragment.measure;
  }
  return summary;
}

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

long readProcessPeakRssKb()
{
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    std::istringstream fields(line);
    std::string key;
    long value = -1;
    if ((fields >> key >> value) && key == "VmHWM:") {
      return value;
    }
  }
  return -1;
}

} // namespace

// Opt-in (SVMP_CUT_MEMORY_REPORT=1) storage breakdown of the generated cut
// geometry, printed at the phases of an active-cut context rebuild.
bool cutGeometryMemoryReportEnabled()
{
  static const bool enabled = [] {
    const char* value = std::getenv("SVMP_CUT_MEMORY_REPORT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

void logCutGeometryMemory(
    const char* phase,
    const std::function<void(svmp::FE::LogRecord&)>& details)
{
  if (!cutGeometryMemoryReportEnabled()) {
    return;
  }
  const auto memory = readProcessMemorySnapshot();
  svmp::FE::LogRecord record("LevelSetGeometryService", "cut geometry memory");
  record.diagnostic("cut_geometry_memory")
      .text("phase", phase)
      .value("rss_kb", memory.rss_kb)
      .value("hwm_kb", readProcessPeakRssKb());
  if (details) {
    details(record);
  }
  svmp::FE::logRecord(record);
}

namespace {

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

} // namespace

const char* activeSideName(LevelSetActiveSide side) noexcept
{
  return side == LevelSetActiveSide::Negative
             ? "LevelSetNegative"
             : "LevelSetPositive";
}

namespace {

const char* cutIntegrationSideName(
    svmp::FE::geometry::CutIntegrationSide side) noexcept
{
  switch (side) {
  case svmp::FE::geometry::CutIntegrationSide::Negative:
    return "Negative";
  case svmp::FE::geometry::CutIntegrationSide::Positive:
    return "Positive";
  case svmp::FE::geometry::CutIntegrationSide::Interface:
    return "Interface";
  }
  return "Unknown";
}

svmp::FE::geometry::CutIntegrationSide oppositeCutIntegrationSide(
    svmp::FE::geometry::CutIntegrationSide side) noexcept
{
  return side == svmp::FE::geometry::CutIntegrationSide::Negative
             ? svmp::FE::geometry::CutIntegrationSide::Positive
             : svmp::FE::geometry::CutIntegrationSide::Negative;
}

const char* retainedVolumeSidesName(
    ActiveCutVolumeRetention retention) noexcept
{
  return retention == ActiveCutVolumeRetention::ActiveAndInactive
             ? "active_and_inactive"
             : "active_only";
}

ActiveSideRegionSummary summarizeActiveSideRegions(
    const svmp::FE::interfaces::LevelSetInterfaceDomain& domain,
    LevelSetActiveSide active_side,
    const svmp::FE::assembly::IMeshAccess& mesh)
{
  ActiveSideRegionSummary summary;
  const auto n_cells = static_cast<std::size_t>(
      std::max<svmp::FE::GlobalIndex>(0, mesh.numCells()));
  std::vector<double> wet_fraction(n_cells, 0.0);
  const auto side = cutIntegrationSide(active_side);
  for (const auto& region : domain.volumeRegions()) {
    const auto parent_cell =
        static_cast<svmp::FE::GlobalIndex>(region.parent_cell);
    if (!region.active() || region.side != side || parent_cell < 0 ||
        parent_cell >= mesh.numCells() || !mesh.isOwnedCell(parent_cell)) {
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
  for (std::size_t cell = 0; cell < wet_fraction.size(); ++cell) {
    if (!mesh.isOwnedCell(static_cast<svmp::FE::GlobalIndex>(cell))) {
      continue;
    }
    const auto fraction = wet_fraction[cell];
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

} // namespace

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

namespace {

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
  stored_handle.side = cutIntegrationSide(active_side);
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

} // namespace

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
  if (changed && svmp::FE::rank_log::oopTraceEnabled()) {
    svmp::FE::logRecord(
        svmp::FE::LogRecord("LevelSetGeometryService", "cut topology changed during a nonlinear solve")
            .diagnostic("cut_topology_change_nonsmooth_event")
            .value("solve_kind", solve_kind)
            .text("event_class", "nonsmooth_cut_topology_change")
            .text("newton_consistency", "not_expected")
            .text("jacobian_validity", "piecewise_smooth_topology_only")
            .value("sync_point", svmp::FE::timestepping::generatedStateSyncPointName(point))
            .value("previous_topology_key", *previous_topology_key)
            .value("topology_key", report.topology_key)
            .value("active_cut_request_policy_key", report.request_policy_key)
            .value("cut_context_revision", report.value_revision)
            .value("cell_count", report.cell_count)
            .value("corner_linearized_cells", report.corner_linearized_cell_count)
            .value("interface_fragments", report.interface_fragments)
            .value("active_volume_regions", report.active_volume_regions)
            .value("active_cut_cells", report.active_cut_cells)
            .value("active_quadrature_points", report.active_quadrature_points)
            .value("domain_interface_quadrature_point_count", report.domain_interface_quadrature_point_count)
            .value("domain_volume_quadrature_point_count", report.domain_volume_quadrature_point_count)
            .value("domain_total_quadrature_point_count", report.domain_total_quadrature_point_count)
            .value("backend_volume_quadrature_point_count", report.backend_volume_quadrature_point_count)
            .value("backend_interface_quadrature_point_count", report.backend_interface_quadrature_point_count)
            .value("backend_total_quadrature_point_count", (report.backend_volume_quadrature_point_count + report.backend_interface_quadrature_point_count))
            .value("backend_elapsed_seconds", report.backend_elapsed_seconds)
            .text("generated_cell_work_scope", "summed_rank_local_including_ghost_cells")
            .value("generated_cell_cache_hits", report.generated_cell_cache_hits)
            .value("generated_cell_cache_misses", report.generated_cell_cache_misses)
            .value("generated_cell_cache_unchanged_dof_hits", report.generated_cell_cache_unchanged_dof_hits)
            .value("generated_cell_refresh_candidates", report.generated_cell_refresh_candidates)
            .value("generated_cell_directly_affected", report.generated_cell_directly_affected)
            .value("generated_cell_affected_neighborhood", report.generated_cell_affected_neighborhood)
            .value("generated_domain_cache_hits", report.generated_domain_cache_hits)
            .value("linear_full_cell_fast_path_cells", report.linear_full_cell_fast_path_count)
            .value("process_vm_kb", report.process_vm_kb)
            .value("process_rss_kb", report.process_rss_kb)
            .value("basis_cache_entries", report.basis_cache_entries)
            .text("cut_adjacent_scope", "summed_rank_local_including_ghost_cells")
            .value("cut_adjacent_facets", report.cut_adjacent_facets)
            .value("negative_volume", report.negative_volume)
            .value("negative_reference_volume", report.negative_volume)
            .value("negative_physical_volume", report.negative_physical_volume)
            .value("positive_volume", report.positive_volume)
            .value("positive_reference_volume", report.positive_volume)
            .value("positive_physical_volume", report.positive_physical_volume)
            .rootOnly());
  }
  previous_topology_key = report.topology_key;
}

namespace {

void logCornerLinearizedCutWarningOnce(
    const ActiveCutVolumeRequest& request,
    const svmp::FE::level_set::LevelSetGeneratedInterfaceResult& result,
    std::size_t global_corner_linearized_cell_count,
    std::size_t global_cell_count)
{
  if (global_corner_linearized_cell_count == 0u) {
    return;
  }

  static std::set<std::string> warned_keys;
  const std::string key = request.level_set_field_name + "|" +
                          request.domain_id + "|" +
                          std::to_string(result.interface_marker);
  if (!warned_keys.insert(key).second) {
    return;
  }

  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "generated interface uses corner-linearized cut geometry", svmp::FE::LogLevel::WARNING)
          .diagnostic("high_order_level_set_cut_uses_corners")
          .value("marker", result.interface_marker)
          .quoted("field", request.level_set_field_name)
          .quoted("domain_id", request.domain_id)
          .value("corner_linearized_cells", global_corner_linearized_cell_count)
          .value("cell_count", global_cell_count)
          .value("max_cell_node_count", result.max_cell_node_count)
          .value("max_corner_node_count", result.max_corner_node_count)
          .rootOnly());
}

} // namespace

std::uint64_t requireActiveCutRequestConsensus(
    const svmp::FE::systems::FESystem& system,
    const std::vector<ActiveCutVolumeRequest>& requests)
{
  if (requests.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
    throw std::runtime_error(
        "Active-cut request count exceeds the collective contract range.");
  }
  const auto comm = activeFESystemCommunicator(system);
  const auto local_count = static_cast<std::uint64_t>(requests.size());
  const auto local_policy_fingerprint =
      activeCutVolumeRequestPolicyKey(requests);

  // Always execute both fixed-order reductions before diagnosing either
  // mismatch.  The count is exact; the 64-bit policy key is a bounded
  // fingerprint of request semantics, not a collision-proof proof.
  const auto [minimum_count, maximum_count] =
      globalMinMaxUint64(local_count, comm);
  const auto [minimum_policy, maximum_policy] =
      globalMinMaxUint64(local_policy_fingerprint, comm);
  if (minimum_count != maximum_count) {
    throw std::runtime_error(
        "Active-cut request counts differ across the FE communicator before refresh collective entry.");
  }
  if (minimum_policy != maximum_policy) {
    throw std::runtime_error(
        "Active-cut request policy fingerprints differ across the FE communicator before refresh collective entry.");
  }
  return local_policy_fingerprint;
}

namespace {

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
  mesh.forEachOwnedCell([&](svmp::FE::GlobalIndex cell_id) {
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

} // namespace

std::vector<double> evaluateVertexField(
    const svmp::FE::systems::FESystem& system,
    const svmp::Mesh& mesh,
    svmp::FE::FieldId field,
    const svmp::FE::systems::SystemStateView& state,
    std::size_t components,
    std::string_view action)
{
  const auto n_vertices = mesh.n_vertices();
  std::vector<double> values(n_vertices * components, 0.0);
  const bool use_vertex_fast_path =
      !parseBoolEnv("SVMP_DISABLE_VERTEX_FIELD_FAST_PATH", false);
  if (use_vertex_fast_path && system.evaluateFieldAtVertices(
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
          "[svMultiPhysics::Application] Could not evaluate field at mesh vertex while " +
          std::string(action) + ".");
    }
    for (std::size_t c = 0; c < components; ++c) {
      values[v * components + c] = static_cast<double>((*value)[c]);
    }
  }
  return values;
}

namespace {

ActiveLevelSetMeshFieldCheckpoint captureActiveLevelSetMeshFields(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& requests)
{
  ActiveLevelSetMeshFieldCheckpoint checkpoint;
  if (!sim.primary_mesh) {
    return checkpoint;
  }
  auto& mesh = *sim.primary_mesh;
  checkpoint.valid = true;
  checkpoint.revisions = mesh.event_bus().revision_state();
  std::set<std::string> visited_fields;
  for (const auto& request : requests) {
    if (!visited_fields.insert(request.level_set_field_name).second ||
        !mesh.has_field(
            svmp::EntityKind::Vertex, request.level_set_field_name)) {
      continue;
    }
    const auto handle = mesh.field_handle(
        svmp::EntityKind::Vertex, request.level_set_field_name);
    if (mesh.field_type(handle) != svmp::FieldScalarType::Float64) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Cannot checkpoint a non-Float64 active level-set mesh field.");
    }
    ActiveLevelSetMeshFieldCheckpoint::Field field;
    field.name = request.level_set_field_name;
    field.components = mesh.field_components(handle);
    field.entity_count = mesh.field_entity_count(handle);
    const auto value_count = field.components * field.entity_count;
    const auto* data = static_cast<const double*>(mesh.field_data(handle));
    if (value_count > 0u && data == nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Cannot checkpoint an empty active level-set mesh field.");
    }
    if (value_count > 0u) {
      field.values.assign(data, data + value_count);
    }
    checkpoint.fields.push_back(std::move(field));
  }
  return checkpoint;
}

void restoreActiveLevelSetMeshFields(
    const LevelSetGeometryHost& sim,
    const ActiveLevelSetMeshFieldCheckpoint& checkpoint)
{
  if (!checkpoint.valid) {
    return;
  }
  if (!sim.primary_mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Cannot restore active level-set mesh fields without the checkpointed mesh.");
  }
  auto& mesh = *sim.primary_mesh;
  for (const auto& field : checkpoint.fields) {
    if (!mesh.has_field(svmp::EntityKind::Vertex, field.name)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] A checkpointed active level-set mesh field disappeared during maintenance.");
    }
    const auto handle =
        mesh.field_handle(svmp::EntityKind::Vertex, field.name);
    if (mesh.field_type(handle) != svmp::FieldScalarType::Float64 ||
        mesh.field_components(handle) != field.components ||
        mesh.field_entity_count(handle) != field.entity_count) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] A checkpointed active level-set mesh field changed layout during maintenance.");
    }
    auto* data = static_cast<double*>(mesh.field_data(handle));
    if (!field.values.empty() && data == nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] A checkpointed active level-set mesh field has no restore storage.");
    }
    if (!field.values.empty()) {
      std::copy(field.values.begin(), field.values.end(), data);
    }
  }
  if (!checkpoint.fields.empty()) {
    mesh.event_bus().notify(svmp::MeshEvent::FieldsChanged);
  }
  mesh.event_bus().restore_revision_state(checkpoint.revisions);
}

} // namespace

std::size_t syncActiveLevelSetVertexFieldsFromSolution(
    const LevelSetGeometryHost& sim,
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
    const auto& rec = system.fieldRecord(field);
    if (rec.components != 1) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active level-set support refresh requires scalar FE field '" +
          rec.name + "'.");
    }
    const auto& field_dofs = system.fieldDofHandler(field);

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
    svmp::FE::systems::SystemStateView state_view{};
    state_view.u = fe_solution;
    const auto sampled =
        evaluateVertexField(system,
                            mesh,
                            field,
                            state_view,
                            1u,
                            "refreshing active level-set support");
    for (svmp::FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
      const auto vertex_index = static_cast<std::size_t>(vertex);
      const auto value = sampled[vertex_index];
      auto& target = data[vertex_index * components];
      if (target != value) {
        target = value;
        field_changed = true;
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

namespace {

// Full-cell snapshot records carry only their classification unless a
// consumer integrates over their volume points without the mesh: the
// two-fluid stage diagnostics.  The active-volume energy, dissipation and
// kinetic-work evaluators rematerialize them from the mesh.  The dry
// (inactive) side is always eligible; in 3D, where full cells dominate the
// snapshot, both sides are.  SVMP_KEEP_DRY_CELL_POINTS=1 keeps every point
// (debugging and memory comparisons).
bool snapshotMayStoreFullCellsClassificationOnly(
    const LevelSetGeometryHost& sim)
{
  static const bool keep_points = [] {
    const char* value = std::getenv("SVMP_KEEP_DRY_CELL_POINTS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return !keep_points &&
         !sim.two_fluid_accepted_stage_diagnostics_declared;
}

// Incremental cut-geometry rebuilds reuse, from one rebuild to the next,
// the work for cells whose inputs did not change (bitwise identical results).
// SVMP_DISABLE_INCREMENTAL_REBUILD=1 rebuilds everything;
// SVMP_INCREMENTAL_REBUILD_SELF_CHECK=1 also rebuilds everything from scratch
// and fails unless the incremental result is identical bit for bit.
bool incrementalRebuildEnabled()
{
  static const bool enabled =
      !parseBoolEnv("SVMP_DISABLE_INCREMENTAL_REBUILD", false);
  return enabled;
}

bool incrementalRebuildSelfCheckEnabled()
{
  static const bool enabled =
      incrementalRebuildEnabled() &&
      parseBoolEnv("SVMP_INCREMENTAL_REBUILD_SELF_CHECK", false);
  return enabled;
}

void addCutGeometryContextMemory(
    svmp::FE::LogRecord& record,
    const std::string& prefix,
    const svmp::FE::assembly::CutIntegrationContext* context)
{
  namespace report = svmp::FE::assembly::memory_report;
  if (context == nullptr) {
    record.integer(prefix + "_mb", 0);
    return;
  }
  report::addContext(record, prefix, report::contextStorage(*context));
  std::size_t index = 0u;
  for (const auto& snapshot : context->freeSurfaceGeometrySnapshots()) {
    if (snapshot) {
      report::addSnapshot(record,
                          prefix + "_snapshot" + std::to_string(index),
                          report::snapshotStorage(*snapshot));
    }
    ++index;
  }
}

void addCutGeometryLifecycleMemory(
    svmp::FE::LogRecord& record,
    const svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle)
{
  namespace report = svmp::FE::assembly::memory_report;
  const auto storage = lifecycle.storage();
  record.real("lifecycle_cells_mb", report::megabytes(storage.cell_cache_bytes))
      .value("lifecycle_cell_points", storage.cell_cache_region_points)
      .real("lifecycle_domain_mb", report::megabytes(storage.domain_cache_bytes))
      .value("lifecycle_domain_points", storage.domain_cache_region_points)
      .real("lifecycle_adjacency_mb", report::megabytes(storage.adjacency_bytes))
      .real("lifecycle_backup_mb",
            report::megabytes(storage.transaction_backup_bytes));
}

// ---- ghost-layer audit (diagnostics only) ----------------------------------
//
// SVMP_AUDIT_GHOST_LAYERS=1 prints, on every rank, how the generated cut
// geometry is distributed over the ghost layers (face-adjacency distance from
// the owned cells; owned cells are layer 0).
// SVMP_AUDIT_WITHHOLD_GHOST_CUT_LAYERS=k removes the generated fragments,
// volume regions and sensitivity records of ghost cells beyond layer k from
// the domain every downstream consumer reads (snapshot, active boundary,
// context, constraints, assembly, ...), to find which consumers need cut data
// how deep into the halo; it acts after the production restriction to
// ActiveCutVolumeRequest::generated_cut_ghost_layers (set that to -1 to audit
// against every layer).  It is an audit tool.
constexpr int kGhostLayerAuditBuckets = 16;

} // namespace

std::vector<int> ghostCellLayers(
    const svmp::FE::assembly::IMeshAccess& mesh)
{
  const auto n_cells = static_cast<std::size_t>(
      std::max<svmp::FE::GlobalIndex>(0, mesh.numCells()));
  std::vector<int> layer(n_cells, std::numeric_limits<int>::max());
  std::vector<std::vector<svmp::FE::GlobalIndex>> neighbors(n_cells);
  mesh.forEachInteriorFace([&](svmp::FE::GlobalIndex,
                               svmp::FE::GlobalIndex a,
                               svmp::FE::GlobalIndex b) {
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= n_cells ||
        static_cast<std::size_t>(b) >= n_cells) {
      return;
    }
    neighbors[static_cast<std::size_t>(a)].push_back(b);
    neighbors[static_cast<std::size_t>(b)].push_back(a);
  });
  std::vector<svmp::FE::GlobalIndex> frontier;
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    if (mesh.isOwnedCell(static_cast<svmp::FE::GlobalIndex>(cell))) {
      layer[cell] = 0;
      frontier.push_back(static_cast<svmp::FE::GlobalIndex>(cell));
    }
  }
  int depth = 0;
  while (!frontier.empty()) {
    ++depth;
    std::vector<svmp::FE::GlobalIndex> next;
    for (const auto cell : frontier) {
      for (const auto neighbor : neighbors[static_cast<std::size_t>(cell)]) {
        auto& value = layer[static_cast<std::size_t>(neighbor)];
        if (value == std::numeric_limits<int>::max()) {
          value = depth;
          next.push_back(neighbor);
        }
      }
    }
    frontier.swap(next);
  }
  return layer;
}

namespace {

[[nodiscard]] std::size_t ghostLayerBucket(int layer) noexcept
{
  return static_cast<std::size_t>(
      std::min(layer, kGhostLayerAuditBuckets - 1));
}

std::string ghostLayerHistogramText(
    const std::array<std::size_t, kGhostLayerAuditBuckets>& counts)
{
  std::string text = "[";
  for (std::size_t i = 0; i < counts.size(); ++i) {
    text += (i == 0 ? "" : ",") + std::to_string(counts[i]);
  }
  return text + "]";
}

} // namespace

// Removes the generated fragments, volume regions and sensitivity records of
// cells beyond face layer `keep_layers` (layers from ghostCellLayers) from
// the domain, keeping the order of the rest.
GhostCutWithholding withholdGhostCutLayersBeyond(
    svmp::FE::interfaces::LevelSetInterfaceDomain& domain,
    const std::vector<int>& layers,
    int keep_layers)
{
  const auto layer_of = [&](svmp::FE::MeshIndex cell) {
    return cell >= 0 && static_cast<std::size_t>(cell) < layers.size()
               ? layers[static_cast<std::size_t>(cell)]
               : std::numeric_limits<int>::max();
  };
  GhostCutWithholding withheld;
  svmp::FE::interfaces::LevelSetInterfaceDomain kept(domain.request());
  for (const auto& fragment : domain.fragments()) {
    if (layer_of(fragment.parent_cell) > keep_layers) {
      ++withheld.fragments;
      continue;
    }
    kept.addFragment(fragment);
  }
  for (const auto& region : domain.volumeRegions()) {
    if (layer_of(region.parent_cell) > keep_layers) {
      ++withheld.regions;
      continue;
    }
    kept.addVolumeRegion(region);
  }
  for (const auto& record : domain.sensitivityRecords()) {
    if (layer_of(record.parent_cell) > keep_layers) {
      ++withheld.sensitivity_records;
      continue;
    }
    kept.addSensitivityRecord(record);
  }
  domain = std::move(kept);
  return withheld;
}

namespace {

void auditGhostLayers(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::interfaces::LevelSetInterfaceDomain& domain,
    const svmp::FE::systems::CollectiveCommunicator& comm,
    std::string_view domain_id)
{
  const bool print = parseBoolEnv("SVMP_AUDIT_GHOST_LAYERS", false);
  const char* withhold_text =
      std::getenv("SVMP_AUDIT_WITHHOLD_GHOST_CUT_LAYERS");
  const bool withhold =
      withhold_text != nullptr && withhold_text[0] != '\0';
  if (!print && !withhold) {
    return;
  }
  const int keep_layers = withhold ? std::atoi(withhold_text) : 0;
  const auto layers = ghostCellLayers(mesh);
  const auto layer_of = [&](svmp::FE::MeshIndex cell) {
    return cell >= 0 && static_cast<std::size_t>(cell) < layers.size()
               ? layers[static_cast<std::size_t>(cell)]
               : std::numeric_limits<int>::max();
  };
  std::array<std::size_t, kGhostLayerAuditBuckets> cells{};
  std::array<std::size_t, kGhostLayerAuditBuckets> cut_cells{};
  std::array<std::size_t, kGhostLayerAuditBuckets> fragments{};
  std::array<std::size_t, kGhostLayerAuditBuckets> regions{};
  for (const auto layer : layers) {
    ++cells[ghostLayerBucket(layer)];
  }
  std::vector<std::uint8_t> is_cut(layers.size(), 0u);
  for (const auto& fragment : domain.fragments()) {
    ++fragments[ghostLayerBucket(layer_of(fragment.parent_cell))];
    if (fragment.parent_cell >= 0 &&
        static_cast<std::size_t>(fragment.parent_cell) < is_cut.size()) {
      is_cut[static_cast<std::size_t>(fragment.parent_cell)] = 1u;
    }
  }
  for (std::size_t cell = 0; cell < is_cut.size(); ++cell) {
    if (is_cut[cell] != 0u) {
      ++cut_cells[ghostLayerBucket(layers[cell])];
    }
  }
  for (const auto& region : domain.volumeRegions()) {
    ++regions[ghostLayerBucket(layer_of(region.parent_cell))];
  }
  if (print) {
    svmp::FE::logRecord(
        svmp::FE::LogRecord("LevelSetGeometryService", "ghost-layer audit")
            .diagnostic("ghost_layer_audit")
            .quoted("domain_id", domain_id)
            .value("rank", comm.rank())
            .value("cells_by_layer", ghostLayerHistogramText(cells))
            .value("cut_cells_by_layer", ghostLayerHistogramText(cut_cells))
            .value("fragments_by_layer", ghostLayerHistogramText(fragments))
            .value("regions_by_layer", ghostLayerHistogramText(regions)));
  }
  if (!withhold) {
    return;
  }
  const auto withheld =
      withholdGhostCutLayersBeyond(domain, layers, keep_layers);
  const auto withheld_fragments = withheld.fragments;
  const auto withheld_regions = withheld.regions;
  const auto withheld_sensitivity = withheld.sensitivity_records;
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "ghost-layer cut withholding")
          .diagnostic("ghost_layer_audit_withhold")
          .quoted("domain_id", domain_id)
          .value("keep_layers", keep_layers)
          .value("withheld_fragments", globalSumSize(withheld_fragments, comm))
          .value("withheld_regions", globalSumSize(withheld_regions, comm))
          .value("withheld_sensitivity_records", globalSumSize(withheld_sensitivity, comm))
          .rootOnly());
}

} // namespace

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextFromSolution(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    std::span<const svmp::FE::Real> fe_solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    const char* provenance,
    const char* solution_source,
    bool request_consensus_already_checked)
{
  ActiveCutContextRefreshReport report{};
  const auto& requests = active_cut_requests;
  if (!sim.fe_system) {
    if (!requests.empty()) {
      throw std::runtime_error(
          "Active-cut refresh has configured requests but no FE system; communicator consensus is unavailable.");
    }
    return report;
  }
  report.request_policy_key =
      request_consensus_already_checked
          ? activeCutVolumeRequestPolicyKey(requests)
          : requireActiveCutRequestConsensus(*sim.fe_system, requests);
  if (requests.empty()) {
    return report;
  }

  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  std::size_t synchronized_level_set_fields{0u};
  std::shared_ptr<svmp::FE::assembly::CutIntegrationContext> context;
  std::vector<ActiveCutTopologySnapshotBinding> topology_bindings;
  std::exception_ptr local_refresh_preparation_failure;
  try {
    synchronized_level_set_fields =
        syncActiveLevelSetVertexFieldsFromSolution(
            sim, requests, fe_solution);
    context =
        std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
    topology_bindings.reserve(requests.size());
  } catch (...) {
    local_refresh_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_refresh_preparation_failure,
      comm,
      "active_cut_source_sync_before_snapshot_build");
  report.refreshed = true;
  const auto& mesh_access = sim.fe_system->meshAccess();
  // Every rank decides alike: the assembler policy and the deck are shared.
  const bool keep_every_ghost_cut_layer =
      comm.size() <= 1 || generatedCutNeedsEveryGhostLayer(sim, requests);
  if (cutGeometryMemoryReportEnabled()) {
    logCutGeometryMemory(
        "refresh_begin", [&](svmp::FE::LogRecord& record) {
          addCutGeometryLifecycleMemory(record, lifecycle);
          addCutGeometryContextMemory(
              record, "installed", sim.fe_system->cutIntegrationContext());
        });
  }

  for (const auto& request : requests) {
    const auto backend_start = Clock::now();
    std::optional<
        svmp::FE::level_set::LevelSetGeneratedInterfaceOptions>
        options_storage;
    std::optional<
        svmp::FE::level_set::LevelSetGeneratedInterfaceResult>
        result_storage;
    std::exception_ptr local_build_failure;
    try {
      auto& options = options_storage.emplace(
          generatedInterfaceOptionsForActiveCut(
              request, mesh_access.dimension()));
      result_storage.emplace(
          lifecycle.build(*sim.fe_system, options, fe_solution));
      if (!result_storage->success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Generated active-domain "
            "interface '" + request.domain_id +
            "' for level-set field '" + request.level_set_field_name +
            "' failed: " + result_storage->diagnostic);
      }
    } catch (...) {
      local_build_failure = std::current_exception();
    }
    const auto local_backend_seconds = secondsSince(backend_start);
    requireCollectivePhasePreparation(
        local_build_failure,
        comm,
        "active_cut_lifecycle_build");
    const auto backend_timing =
        reduceOutputTiming(local_backend_seconds, comm);
    auto& options = *options_storage;
    auto& result = *result_storage;
    // Generated cut data beyond the kept ghost layers is withheld before any
    // consumer reads the domain (see generatedCutNeedsEveryGhostLayer).
    const int kept_ghost_cut_layers =
        keep_every_ghost_cut_layer ? -1 : request.generated_cut_ghost_layers;
    GhostCutWithholding ghost_cut_withheld;
    std::exception_ptr local_ghost_restriction_failure;
    try {
      if (kept_ghost_cut_layers >= 0) {
        ghost_cut_withheld = withholdGhostCutLayersBeyond(
            result.domain, ghostCellLayers(mesh_access),
            kept_ghost_cut_layers);
      }
    } catch (...) {
      local_ghost_restriction_failure = std::current_exception();
    }
    requireCollectivePhasePreparation(
        local_ghost_restriction_failure,
        comm,
        "active_cut_ghost_layer_restriction");
    const auto global_withheld_ghost_cut_fragments =
        globalSumSize(ghost_cut_withheld.fragments, comm);
    const auto global_withheld_ghost_cut_regions =
        globalSumSize(ghost_cut_withheld.regions, comm);
    auditGhostLayers(mesh_access, result.domain, comm, request.domain_id);
    if (svmp::FE::geometryThreadsSelfCheckEnabled()) {
      // Every rank reads the same environment, so the reduction is matched.
      svmp::FE::logRecord(
          svmp::FE::LogRecord("LevelSetGeometryService", "geometry-thread self-check")
              .diagnostic("geometry_threads_self_check")
              .text("stage", "generated_cells")
              .quoted("domain_id", request.domain_id)
              .text("identical", "1")
              .value("checked_cells", globalSumSize(result.thread_self_checked_cell_count, comm))
              .rootOnly());
    }
    if (cutGeometryMemoryReportEnabled()) {
      logCutGeometryMemory(
          "after_lifecycle_build", [&](svmp::FE::LogRecord& record) {
            addCutGeometryLifecycleMemory(record, lifecycle);
            svmp::FE::assembly::memory_report::addDomain(
                record,
                "result_domain",
                svmp::FE::assembly::memory_report::domainStorage(
                    result.domain));
          });
    }
    const bool local_missing_source_topology =
        std::any_of(
            result.domain.fragments().begin(),
            result.domain.fragments().end(),
            [](const auto& fragment) {
              return fragment.parent_corner_topology_key == 0u;
            }) ||
        std::any_of(
            result.domain.volumeRegions().begin(),
            result.domain.volumeRegions().end(),
            [](const auto& region) {
              return region.parent_corner_topology_key == 0u;
            });
    if (globalAnyBool(local_missing_source_topology, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Generated active-domain geometry is missing its epoch-free parent-corner topology descriptor.");
    }
    const auto [source_revision_it, source_revision_inserted] =
        report.evaluated_state_source_revisions.emplace(
            result.interface_marker, result.value_revision);
    if ((!source_revision_inserted &&
         source_revision_it->second != result.value_revision) ||
        result.value_revision == 0u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Generated active-domain geometry "
          "did not produce a unique nonzero evaluated-state source revision "
          "for its interface marker.");
    }
    validateEquationLevelCutVolumeConsumer(
        sim, *sim.fe_system, request, result.interface_marker);

    // The lifecycle retains ghost-cell rules in result.domain/result.summary
    // for distributed assembly.  Only the owned view may enter global
    // physical and topology totals.
    const auto summary = result.owned_summary;
    const auto active_summary = summarizeActiveSideRegions(
        result.domain,
        request.active_side,
        mesh_access);
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
        globalSumSize(result.owned_cell_count, comm);
    const auto global_corner_linearized_cells =
        globalSumSize(result.owned_corner_linearized_cell_count, comm);
    const auto global_implicit_cut_fallback_cells =
        globalSumSize(result.owned_implicit_cut_fallback_cell_count, comm);
    logCornerLinearizedCutWarningOnce(
        request, result, global_corner_linearized_cells, global_cell_count);
    const auto global_backend_volume_quadrature_points =
        globalSumSize(result.owned_backend_volume_quadrature_point_count, comm);
    const auto global_backend_interface_quadrature_points =
        globalSumSize(result.owned_backend_interface_quadrature_point_count,
                      comm);
    const auto global_backend_elapsed_seconds =
        globalSumDouble(result.backend_elapsed_seconds, comm);
    const auto global_generated_cell_cache_hits =
        globalSumSize(result.cell_cache_hits, comm);
    const auto global_generated_cell_cache_misses =
        globalSumSize(result.cell_cache_misses, comm);
    const auto global_generated_cell_cache_unchanged_dof_hits =
        globalSumSize(result.cell_cache_unchanged_dof_hits, comm);
    const auto global_generated_cell_refresh_candidates =
        globalSumSize(result.cell_refresh_candidate_count, comm);
    const auto global_generated_cell_directly_affected =
        globalSumSize(result.directly_affected_cell_count, comm);
    const auto global_generated_cell_affected_neighborhood =
        globalSumSize(result.affected_cell_neighborhood_count, comm);
    const auto global_generated_domain_cache_hits =
        globalSumSize(result.domain_cache_hits, comm);
    const auto global_linear_full_cell_fast_path_cells =
        globalSumSize(result.owned_linear_full_cell_fast_path_count, comm);
    std::array<std::size_t, 5> global_selected_backend_counts{};
    for (std::size_t i = 0; i < global_selected_backend_counts.size(); ++i) {
      global_selected_backend_counts[i] = globalSumSize(
          result.owned_selected_implicit_cut_quadrature_backend_counts[i],
          comm);
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
    report.generated_cell_cache_unchanged_dof_hits +=
        global_generated_cell_cache_unchanged_dof_hits;
    report.generated_cell_refresh_candidates +=
        global_generated_cell_refresh_candidates;
    report.generated_cell_directly_affected +=
        global_generated_cell_directly_affected;
    report.generated_cell_affected_neighborhood +=
        global_generated_cell_affected_neighborhood;
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
    const auto inactive_volume_side =
        oppositeCutIntegrationSide(active_volume_side);
    const auto retained_volume_sides =
        request.volume_retention ==
                ActiveCutVolumeRetention::ActiveAndInactive
            ? std::optional<svmp::FE::geometry::CutIntegrationSide>{}
            : std::optional<svmp::FE::geometry::CutIntegrationSide>{
                  active_volume_side};
    // volumeQuadratureRules() yields one volume rule per active region, so
    // count the regions instead of materializing every rule.
    const auto available_volume_rule_count =
        [&result](svmp::FE::geometry::CutIntegrationSide side) {
          return static_cast<std::size_t>(std::count_if(
              result.domain.volumeRegions().begin(),
              result.domain.volumeRegions().end(),
              [side](const auto& region) {
                return region.active() && region.side == side;
              }));
        };
    const auto local_negative_available_cut_volume_rules =
        available_volume_rule_count(
            svmp::FE::geometry::CutIntegrationSide::Negative);
    const auto local_positive_available_cut_volume_rules =
        available_volume_rule_count(
            svmp::FE::geometry::CutIntegrationSide::Positive);
    std::size_t local_boundary_intersection_fragments = 0u;
    std::size_t local_active_boundary_intersection_fragments = 0u;
    std::size_t local_skipped_boundary_intersection_fragments = 0u;
    std::size_t local_boundary_intersection_qpoints = 0u;
    svmp::FE::Real local_boundary_intersection_measure{0.0};
    // Boundary-face ownership is rank-local.  Every rank must nevertheless
    // enter the per-marker reductions below in the same order, including
    // ranks with no local faces for a marker.  The intersection builder
    // returns an empty domain for that case.
    const auto boundary_markers =
        communicatorWideBoundaryMarkers(mesh_access, comm);
    std::vector<svmp::FE::interfaces::
                    GeneratedInterfaceBoundaryIntersectionDomain>
        snapshot_contact_domains;
    std::vector<svmp::FE::interfaces::GeneratedActiveBoundaryDomain>
        snapshot_active_boundary_domains;
    snapshot_contact_domains.reserve(boundary_markers.size());
    snapshot_active_boundary_domains.reserve(2u * boundary_markers.size());
    for (const int boundary_marker : boundary_markers) {
      svmp::FE::interfaces::
          GeneratedInterfaceBoundaryIntersectionRequest intersection_request;
      intersection_request.source = domain_request.source;
      intersection_request.generated_domain_id = request.domain_id;
      intersection_request.isovalue = domain_request.isovalue;
      intersection_request.interface_marker = result.interface_marker;
      intersection_request.boundary_marker = boundary_marker;
      intersection_request.tolerance = domain_request.tolerance;
      intersection_request.quadrature_order =
          domain_request.resolvedInterfaceQuadratureOrder();
      intersection_request.frame = domain_request.frame;
      intersection_request.mesh_geometry_revision =
          domain_request.mesh_geometry_revision;
      intersection_request.mesh_topology_revision =
          domain_request.mesh_topology_revision;
      intersection_request.ownership_revision =
          domain_request.ownership_revision;
      intersection_request.quadrature_policy_key =
          domain_request.quadrature_policy_key;
      intersection_request.source_value_revision = result.value_revision;

      auto intersection_domain = svmp::FE::interfaces::
          buildGeneratedInterfaceBoundaryIntersectionDomain(
              std::move(intersection_request),
              result.domain,
              mesh_access);
      const auto intersection_provenance =
          svmp::FE::interfaces::
              validateGeneratedInterfaceBoundaryProvenance(
                  intersection_domain, result.domain);
      // Keep ghost contact rules in the local context, but reduce only the
      // owning parent cell's contribution to global validation/measure data.
      const auto intersection_summary =
          ownedBoundaryIntersectionSummary(intersection_domain, mesh_access);
      local_boundary_intersection_fragments +=
          intersection_summary.fragment_count;
      local_active_boundary_intersection_fragments +=
          intersection_summary.active_fragment_count;
      local_skipped_boundary_intersection_fragments +=
          intersection_summary.skipped_fragment_count;
      local_boundary_intersection_qpoints +=
          intersection_summary.quadrature_point_count;
      local_boundary_intersection_measure +=
          intersection_summary.measure;
      const auto global_marker_fragments =
          globalSumSize(intersection_summary.fragment_count, comm);
      const auto global_marker_active_fragments =
          globalSumSize(intersection_summary.active_fragment_count, comm);
      const auto global_marker_skipped_fragments =
          globalSumSize(intersection_summary.skipped_fragment_count, comm);
      const auto global_marker_qpoints =
          globalSumSize(intersection_summary.quadrature_point_count, comm);
      const auto global_marker_measure =
          static_cast<svmp::FE::Real>(globalSumDouble(
              static_cast<double>(intersection_summary.measure), comm));
      if (svmp::FE::rank_log::oopDiagnosticsEnabled(svmp::FE::LogLevel::DEBUG)) {
          svmp::FE::logRecord(
              svmp::FE::LogRecord("LevelSetGeometryService", "generated interface-boundary intersection")
                  .diagnostic("generated_interface_boundary_intersection_marker")
                  .value("marker", intersection_domain.marker())
                  .value("interface_marker", result.interface_marker)
                  .value("boundary_marker", boundary_marker)
                  .quoted("field", request.level_set_field_name)
                  .quoted("domain_id", request.domain_id)
                  .quoted("source", domain_request.source.identifier())
                  .value("fragments", global_marker_fragments)
                  .value("active_fragments", global_marker_active_fragments)
                  .value("skipped_fragments", global_marker_skipped_fragments)
                  .value("qpoints", global_marker_qpoints)
                  .value("measure", global_marker_measure)
                  .value("source_surface_fragments", intersection_provenance.source_surface_fragment_count)
                  .value("referenced_source_surface_fragments", intersection_provenance.referenced_source_surface_fragment_count)
                  .value("orphan_contact_fragments", intersection_provenance.orphan_contact_fragment_count)
                  .value("stale_revision_count", intersection_provenance.stale_revision_count)
                  .value("max_level_set_residual", intersection_provenance.max_level_set_residual)
                  .rootOnly());
      }
      const bool contact_marker_is_consumed =
          sim.fe_system->isGeneratedEmbeddedInterfaceMarkerRegistered(
              intersection_domain.marker());
      if (global_marker_skipped_fragments > 0u) {
        throw std::runtime_error(
            std::string("[svMultiPhysics::Application] Generated "
                        "interface-boundary intersection validation rejected "
                        "a degenerate contact fragment") +
            " marker=" + std::to_string(intersection_domain.marker()) +
            " interface_marker=" + std::to_string(result.interface_marker) +
            " boundary_marker=" + std::to_string(boundary_marker) +
            " field='" + request.level_set_field_name + "'" +
            " domain_id='" + request.domain_id + "'" +
            " fragments=" + std::to_string(global_marker_fragments) +
            " active_fragments=" +
            std::to_string(global_marker_active_fragments) +
            " skipped_fragments=" +
            std::to_string(global_marker_skipped_fragments) +
            " qpoints=" + std::to_string(global_marker_qpoints) +
            " contact_marker_consumed=" +
            (contact_marker_is_consumed ? "true" : "false"));
      }
      const auto make_active_boundary_request =
          [&](svmp::FE::geometry::CutIntegrationSide side) {
            svmp::FE::interfaces::GeneratedActiveBoundaryRequest active_request;
            active_request.source = domain_request.source;
            active_request.generated_domain_id = request.domain_id;
            active_request.isovalue = domain_request.isovalue;
            active_request.interface_marker = result.interface_marker;
            active_request.boundary_marker = boundary_marker;
            active_request.side = side;
            active_request.tolerance = domain_request.tolerance;
            active_request.quadrature_order =
                domain_request.resolvedInterfaceQuadratureOrder();
            active_request.frame = domain_request.frame;
            active_request.mesh_geometry_revision =
                domain_request.mesh_geometry_revision;
            active_request.mesh_topology_revision =
                domain_request.mesh_topology_revision;
            active_request.ownership_revision =
                domain_request.ownership_revision;
            active_request.quadrature_policy_key =
                domain_request.quadrature_policy_key;
            active_request.source_value_revision = result.value_revision;
            return active_request;
          };
      auto negative_boundary_request = make_active_boundary_request(
          svmp::FE::geometry::CutIntegrationSide::Negative);
      auto positive_boundary_request = make_active_boundary_request(
          svmp::FE::geometry::CutIntegrationSide::Positive);
      if (!sim.primary_mesh ||
            !sim.primary_mesh->has_field(
                svmp::EntityKind::Vertex,
                request.level_set_field_name)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Sharp active-boundary construction requires the synchronized scalar level-set mesh field '" +
              request.level_set_field_name + "'.");
        }
        const auto level_set_handle = sim.primary_mesh->field_handle(
            svmp::EntityKind::Vertex, request.level_set_field_name);
        if (sim.primary_mesh->field_type(level_set_handle) !=
                svmp::FieldScalarType::Float64 ||
            sim.primary_mesh->field_components(level_set_handle) < 1u) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Sharp active-boundary construction requires a scalar Float64 level-set mesh field.");
        }
        const auto* level_set_values = static_cast<const double*>(
            sim.primary_mesh->field_data(level_set_handle));
        const auto level_set_components =
            sim.primary_mesh->field_components(level_set_handle);
        const auto level_set_entities =
            sim.primary_mesh->field_entity_count(level_set_handle);
        if (level_set_values == nullptr) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Sharp active-boundary construction found an empty level-set mesh field.");
        }
        svmp::FE::interfaces::GeneratedActiveBoundaryScalarField scalar_field;
        scalar_field.value_at_node =
            [level_set_values,
             level_set_components,
             level_set_entities](svmp::FE::GlobalIndex node) {
              if (node < 0 ||
                  static_cast<std::size_t>(node) >= level_set_entities) {
                throw std::out_of_range(
                    "sharp active-boundary node is outside the synchronized level-set field");
              }
              return static_cast<svmp::FE::Real>(
                  level_set_values[static_cast<std::size_t>(node) *
                                   level_set_components]);
            };
        // Faces are built on the geometry threads; the self-check builds
        // the domains again on one thread and requires identical results.
        const int active_boundary_threads = svmp::FE::geometryThreadCount(
            sim.fe_system->assemblyThreadCount());
        const bool active_boundary_self_check =
            active_boundary_threads > 1 &&
            svmp::FE::geometryThreadsSelfCheckEnabled();
        std::optional<svmp::FE::interfaces::GeneratedActiveBoundaryRequest>
            negative_reference_request;
        std::optional<svmp::FE::interfaces::GeneratedActiveBoundaryRequest>
            positive_reference_request;
        if (active_boundary_self_check) {
          negative_reference_request = negative_boundary_request;
          positive_reference_request = positive_boundary_request;
        }
        auto negative_boundary =
            svmp::FE::interfaces::buildGeneratedActiveBoundaryDomain(
                std::move(negative_boundary_request),
                result.domain,
                intersection_domain,
                mesh_access,
                scalar_field,
                active_boundary_threads);
        auto positive_boundary =
            svmp::FE::interfaces::buildGeneratedActiveBoundaryDomain(
                std::move(positive_boundary_request),
                result.domain,
                intersection_domain,
                mesh_access,
                scalar_field,
                active_boundary_threads);
        if (active_boundary_self_check) {
          const auto negative_reference =
              svmp::FE::interfaces::buildGeneratedActiveBoundaryDomain(
                  std::move(*negative_reference_request), result.domain,
                  intersection_domain, mesh_access, scalar_field, 1);
          const auto positive_reference =
              svmp::FE::interfaces::buildGeneratedActiveBoundaryDomain(
                  std::move(*positive_reference_request), result.domain,
                  intersection_domain, mesh_access, scalar_field, 1);
          auto difference =
              svmp::FE::interfaces::compareGeneratedActiveBoundaryDomains(
                  negative_boundary, negative_reference);
          if (difference.empty()) {
            difference =
                svmp::FE::interfaces::compareGeneratedActiveBoundaryDomains(
                    positive_boundary, positive_reference);
          }
          const bool local_failure = !difference.empty();
          if (globalAnyBool(local_failure, comm)) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Geometry-thread self-check "
                "(SVMP_GEOMETRY_THREADS_SELF_CHECK): the active boundary built "
                "on " + std::to_string(active_boundary_threads) +
                " threads differs from the one-thread build (rank " +
                std::to_string(comm.rank()) + "): " +
                (difference.empty() ? "differs on another rank" : difference));
          }
          svmp::FE::logRecord(
              svmp::FE::LogRecord("LevelSetGeometryService", "geometry-thread self-check")
                  .diagnostic("geometry_threads_self_check")
                  .text("stage", "active_boundary")
                  .value("threads", active_boundary_threads)
                  .value("fragments", negative_boundary.fragments().size() + positive_boundary.fragments().size())
                  .rootOnly());
        }
        const auto partition =
            svmp::FE::interfaces::validateGeneratedActiveBoundaryPartition(
                negative_boundary,
                positive_boundary,
                result.domain,
                intersection_domain,
                mesh_access);
        if (svmp::FE::rank_log::oopDiagnosticsEnabled(svmp::FE::LogLevel::DEBUG)) {
            svmp::FE::logRecord(
                svmp::FE::LogRecord("LevelSetGeometryService", "generated sharp active boundary")
                    .diagnostic("generated_active_boundary_partition")
                    .value("interface_marker", result.interface_marker)
                    .value("boundary_marker", boundary_marker)
                    .value("negative_marker", negative_boundary.marker())
                    .value("positive_marker", positive_boundary.marker())
                    .value("boundary_faces", partition.boundary_face_count)
                    .value("cut_boundary_faces", partition.cut_boundary_face_count)
                    .value("contact_fragments", partition.source_contact_fragment_count)
                    .value("referenced_contact_fragments", partition.referenced_contact_fragment_count)
                    .value("orphan_source_references", partition.orphan_source_reference_count)
                    .value("stale_revision_count", partition.stale_revision_count)
                    .value("reference_parent_measure", partition.total_boundary_measure)
                    .value("reference_negative_measure", partition.negative_boundary_measure)
                    .value("reference_positive_measure", partition.positive_boundary_measure)
                    .value("max_reference_partition_error", partition.max_partition_error)
                    .rootOnly());
        }
        snapshot_active_boundary_domains.push_back(
            std::move(negative_boundary));
        snapshot_active_boundary_domains.push_back(
            std::move(positive_boundary));
        snapshot_contact_domains.push_back(std::move(intersection_domain));
    }
    if (domain_request.source.kind !=
            svmp::FE::interfaces::CutInterfaceSourceKind::Field ||
        domain_request.source.field_id == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Authoritative free-surface geometry snapshots currently require a registered scalar level-set field source.");
    }
    auto snapshot_cell_evaluator = std::make_shared<
        svmp::FE::level_set::LevelSetCellEvaluator>(
        svmp::FE::level_set::makeLevelSetCellEvaluator(
            *sim.fe_system,
            domain_request.source.field_id,
            fe_solution));
    // The snapshot's level-set evaluator; the cell evaluator caches the last
    // cell's coefficients, so each snapshot thread gets its own copy.
    const auto make_snapshot_scalar_evaluator =
        [](std::shared_ptr<svmp::FE::level_set::LevelSetCellEvaluator>
               snapshot_cell_evaluator) {
          svmp::FE::interfaces::FreeSurfaceGeometryScalarEvaluator
              snapshot_scalar_evaluator;
          snapshot_scalar_evaluator.value =
              [snapshot_cell_evaluator](
                  svmp::FE::GlobalIndex cell,
                  const std::array<svmp::FE::Real, 3>& parent_coordinate,
                  const svmp::FE::geometry::CutQuadratureProvenance& provenance) {
                const bool linear_corner =
                    provenance.selected_implicit_quadrature_backend ==
                        "LinearCorner";
                const bool high_order_implicit =
                    provenance.selected_implicit_quadrature_backend ==
                        "SayeHyperrectangle" ||
                    provenance.selected_implicit_quadrature_backend ==
                        "HighOrderSubcell";
                if (!linear_corner && !high_order_implicit) {
                  throw std::runtime_error(
                      "Authoritative free-surface geometry rule has an unsupported represented implicit backend '" +
                      provenance.selected_implicit_quadrature_backend + "'.");
                }
                // evaluateLinearCornerValue() is evaluateLinearCorner().value
                // bit for bit, without the gradient.
                return linear_corner
                    ? snapshot_cell_evaluator->evaluateLinearCornerValue(
                          cell, parent_coordinate)
                    : snapshot_cell_evaluator->evaluate(cell, parent_coordinate)
                          .value;
              };
          snapshot_scalar_evaluator.reference_gradient =
              [snapshot_cell_evaluator](
                  svmp::FE::GlobalIndex cell,
                  const std::array<svmp::FE::Real, 3>& parent_coordinate,
                  const svmp::FE::geometry::CutQuadratureProvenance& provenance) {
                const bool linear_corner =
                    provenance.selected_implicit_quadrature_backend ==
                        "LinearCorner";
                const bool high_order_implicit =
                    provenance.selected_implicit_quadrature_backend ==
                        "SayeHyperrectangle" ||
                    provenance.selected_implicit_quadrature_backend ==
                        "HighOrderSubcell";
                if (!linear_corner && !high_order_implicit) {
                  throw std::runtime_error(
                      "Authoritative free-surface geometry rule has an unsupported represented implicit backend '" +
                      provenance.selected_implicit_quadrature_backend + "'.");
                }
                const auto evaluation = linear_corner
                    ? snapshot_cell_evaluator->evaluateLinearCorner(
                          cell, parent_coordinate)
                    : snapshot_cell_evaluator->evaluate(cell, parent_coordinate);
                return evaluation.reference_gradient;
              };
          return snapshot_scalar_evaluator;
        };
    auto snapshot_scalar_evaluator =
        make_snapshot_scalar_evaluator(snapshot_cell_evaluator);
    snapshot_scalar_evaluator.make_concurrent_copy =
        [snapshot_cell_evaluator, make_snapshot_scalar_evaluator]() {
          return make_snapshot_scalar_evaluator(
              std::make_shared<svmp::FE::level_set::LevelSetCellEvaluator>(
                  *snapshot_cell_evaluator));
        };
    const int snapshot_threads = svmp::FE::geometryThreadCount(
        sim.fe_system->assemblyThreadCount());
    svmp::FE::interfaces::FreeSurfaceGeometrySnapshotPolicy snapshot_policy;
    snapshot_policy.tolerance = domain_request.tolerance;
    snapshot_policy.minimum_retained_volume_fraction =
        svmp::FE::assembly::CutIntegrationContext::
            minGeneratedCutVolumeFraction();
    snapshot_policy.minimum_achieved_quadrature_order = 0;
    snapshot_policy.require_complete_exterior_boundary_partition = true;
    if (snapshotMayStoreFullCellsClassificationOnly(sim)) {
      snapshot_policy.classification_only_full_cell_side =
          inactive_volume_side;
      snapshot_policy.classification_only_full_cells_on_both_sides =
          mesh_access.dimension() == 3;
    }
    svmp::FE::interfaces::FreeSurfaceGeometrySnapshotReuseCache*
        snapshot_reuse_cache = nullptr;
    if (incrementalRebuildEnabled()) {
      auto& slot =
          sim.free_surface_geometry_snapshot_reuse_caches[request.domain_id];
      if (!slot) {
        slot = std::make_unique<
            svmp::FE::interfaces::FreeSurfaceGeometrySnapshotReuseCache>();
      }
      snapshot_reuse_cache = slot.get();
    }
    std::shared_ptr<const svmp::FE::interfaces::FreeSurfaceGeometrySnapshot>
        reference_geometry_snapshot;
    if (incrementalRebuildSelfCheckEnabled()) {
      // Every rank takes this branch (the setting is read from the
      // environment), so the reference build's collectives stay matched.
      reference_geometry_snapshot =
          svmp::FE::interfaces::buildFreeSurfaceGeometrySnapshot(
              result.domain,
              snapshot_contact_domains,
              snapshot_active_boundary_domains,
              mesh_access,
              snapshot_policy,
              snapshot_scalar_evaluator,
              request.domain_id,
              snapshotOwnershipCollective(comm),
              nullptr,
              /*threads=*/1);
    }
    auto geometry_snapshot =
        svmp::FE::interfaces::buildFreeSurfaceGeometrySnapshot(
            result.domain,
            std::move(snapshot_contact_domains),
            std::move(snapshot_active_boundary_domains),
            mesh_access,
            snapshot_policy,
            std::move(snapshot_scalar_evaluator),
            request.domain_id,
            snapshotOwnershipCollective(comm),
            snapshot_reuse_cache,
            snapshot_threads);
    if (reference_geometry_snapshot) {
      const auto difference =
          svmp::FE::interfaces::compareFreeSurfaceGeometrySnapshots(
              *geometry_snapshot, *reference_geometry_snapshot);
      if (globalAnyBool(!difference.empty(), comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Incremental rebuild self-check: "
            "the reused free-surface geometry snapshot differs from a full "
            "rebuild (domain_id='" +
            request.domain_id + "' rank=" + std::to_string(comm.rank()) +
            "): " + (difference.empty() ? "differs on another rank"
                                        : difference));
      }
      svmp::FE::logRecord(
          svmp::FE::LogRecord("LevelSetGeometryService", "incremental rebuild self-check")
              .diagnostic("incremental_rebuild_self_check")
              .text("stage", "snapshot")
              .quoted("domain_id", request.domain_id)
              .text("identical", "1")
              .value("reused_full_cell_records", globalSumSize(snapshot_reuse_cache->lastBuild().full_cell_records_reused, comm))
              .value("rules", globalSumSize(geometry_snapshot->rules().size(), comm))
              .rootOnly());
      reference_geometry_snapshot.reset();
    }
    if (!sim.free_surface_geometry_snapshot_cache) {
      sim.free_surface_geometry_snapshot_cache = std::make_unique<
          svmp::FE::interfaces::FreeSurfaceGeometrySnapshotCache>();
    }
    const auto snapshot_revision_key =
        geometry_snapshot->revision().snapshot_revision_key;
    if (auto cached = sim.free_surface_geometry_snapshot_cache->find(
            snapshot_revision_key)) {
      geometry_snapshot = std::move(cached);
    } else {
      sim.free_surface_geometry_snapshot_cache->insert(geometry_snapshot);
    }
    // The cache statistics feed only the diagnostic line below.
    const auto geometry_cache_statistics =
        svmp::FE::rank_log::oopDiagnosticsEnabled(svmp::FE::LogLevel::DEBUG)
            ? sim.free_surface_geometry_snapshot_cache->statistics()
            : svmp::FE::interfaces::FreeSurfaceGeometrySnapshotCacheStatistics{};
    const auto& geometry_ledger = geometry_snapshot->ledger();
    if (svmp::FE::rank_log::oopDiagnosticsEnabled(svmp::FE::LogLevel::DEBUG)) {
        svmp::FE::logRecord(
            svmp::FE::LogRecord("LevelSetGeometryService", "authoritative free-surface geometry")
                .diagnostic("free_surface_geometry_snapshot")
                .quoted("domain_id", request.domain_id)
                .value("interface_marker", result.interface_marker)
                .value("snapshot_revision", geometry_snapshot->revision().snapshot_revision_key)
                .value("snapshot_resident_bytes", geometry_snapshot->residentBytes())
                .value("cache_live_snapshots", geometry_cache_statistics.live_snapshot_count)
                .value("cache_live_resident_bytes", geometry_cache_statistics.live_resident_bytes)
                .value("cache_peak_live_snapshots", geometry_cache_statistics.peak_live_snapshot_count)
                .value("cache_peak_live_resident_bytes", geometry_cache_statistics.peak_live_resident_bytes)
                .value("cache_hits", geometry_cache_statistics.hit_count)
                .value("cache_misses", geometry_cache_statistics.miss_count)
                .value("cache_expired_evictions", geometry_cache_statistics.expired_eviction_count)
                .value("source_value_revision", geometry_snapshot->revision().source_value_revision)
                .value("rules", geometry_ledger.rule_count)
                .value("retained_rules", geometry_ledger.retained_rule_count)
                .value("pruned_rules", geometry_ledger.pruned_rule_count)
                .value("qpoints", geometry_ledger.quadrature_point_count)
                .value("owned_rules", geometry_ledger.owned_rule_count)
                .value("global_owned_rules", geometry_ledger.global_owned_rule_count)
                .value("invalid_global_identities", geometry_ledger.invalid_global_identity_count)
                .value("duplicate_rule_identities", geometry_ledger.duplicate_rule_identity_count)
                .value("contact_fragments", geometry_ledger.contact_fragment_count)
                .value("orphan_contact_fragments", geometry_ledger.orphan_contact_fragment_count)
                .value("stale_revisions", geometry_ledger.stale_revision_count)
                .value("invalid_phase_points", geometry_ledger.invalid_phase_point_count)
                .value("represented_phase_points", geometry_ledger.represented_phase_point_count)
                .value("represented_phase_disagreements", geometry_ledger.represented_phase_disagreement_count)
                .value("max_root_residual", geometry_ledger.maximum_root_residual)
                .value("max_normal_angle_error", geometry_ledger.maximum_normal_angular_error)
                .value("max_represented_phase_disagreement", geometry_ledger.maximum_represented_phase_disagreement)
                .value("max_constant_moment_error", geometry_ledger.maximum_constant_moment_error)
                .value("certified_rules", geometry_ledger.certified_rule_count)
                .value("parent_cell_moment_certificates", geometry_ledger.parent_cell_moment_certificate_count)
                .value("centroid_moment_certificates", geometry_ledger.centroid_moment_certificate_count)
                .value("piecewise_affine_moment_certificates", geometry_ledger.piecewise_affine_moment_certificate_count)
                .value("backend_reference_moment_certificates", geometry_ledger.backend_reference_moment_certificate_count)
                .value("stored_generated_moment_certificates", geometry_ledger.stored_generated_moment_certificate_count)
                .value("validated_rule_polynomial_moments", geometry_ledger.validated_rule_polynomial_moment_count)
                .value("validated_polynomial_moments", geometry_ledger.validated_polynomial_moment_count)
                .value("max_polynomial_moment_error", geometry_ledger.maximum_polynomial_moment_error)
                .value("max_polynomial_moment_scaled_error", geometry_ledger.maximum_polynomial_moment_scaled_error)
                .value("max_volume_partition_error", geometry_ledger.maximum_volume_partition_error)
                .value("max_boundary_partition_error", geometry_ledger.maximum_boundary_partition_error)
                .value("negative_reference_volume", geometry_ledger.retained_negative_reference_volume)
                .value("positive_reference_volume", geometry_ledger.retained_positive_reference_volume)
                .value("negative_physical_volume", geometry_ledger.retained_negative_physical_volume)
                .value("positive_physical_volume", geometry_ledger.retained_positive_physical_volume)
                .value("resident_bytes", geometry_snapshot->residentBytes())
                .value("reuse_context_matched", (snapshot_reuse_cache != nullptr && snapshot_reuse_cache->lastBuild().context_matched ? 1 : 0))
                .value("reused_full_cell_records", (snapshot_reuse_cache != nullptr ? snapshot_reuse_cache->lastBuild().full_cell_records_reused : 0u))
                .value("built_full_cell_records", (snapshot_reuse_cache != nullptr ? snapshot_reuse_cache->lastBuild().full_cell_records_built : 0u))
                .rootOnly());
    }
    if (cutGeometryMemoryReportEnabled()) {
      logCutGeometryMemory(
          "after_snapshot_build", [&](svmp::FE::LogRecord& record) {
            svmp::FE::assembly::memory_report::addSnapshot(
                record,
                "snapshot",
                svmp::FE::assembly::memory_report::snapshotStorage(
                    *geometry_snapshot));
          });
    }
    topology_bindings.push_back(
        ActiveCutTopologySnapshotBinding{
            .request = &request,
            .snapshot = geometry_snapshot.get(),
        });
    // The context keeps only the classification of the dry full-cell rules
    // too when no cut-volume kernel integrates over the dry side.
    const auto classification_only_context_side =
        snapshot_policy.classification_only_full_cell_side.has_value() &&
                sim.fe_system->cutVolumeKernelCount(
                    result.interface_marker, inactive_volume_side) == 0u
            ? snapshot_policy.classification_only_full_cell_side
            : std::optional<svmp::FE::geometry::CutIntegrationSide>{};
    context->addFreeSurfaceGeometrySnapshot(
        geometry_snapshot,
        retained_volume_sides,
        classification_only_context_side);
    logCutGeometryMemory("after_context_import");
    const auto global_boundary_intersection_fragments =
        globalSumSize(local_boundary_intersection_fragments, comm);
    const auto global_active_boundary_intersection_fragments =
        globalSumSize(local_active_boundary_intersection_fragments, comm);
    const auto global_skipped_boundary_intersection_fragments =
        globalSumSize(local_skipped_boundary_intersection_fragments, comm);
    const auto global_boundary_intersection_qpoints =
        globalSumSize(local_boundary_intersection_qpoints, comm);
    const auto global_boundary_intersection_measure =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(local_boundary_intersection_measure), comm));
    const auto active_measure_summary =
        collectCutVolumeMeasures(
            mesh_access,
            context->generatedVolumeRulesForMarkerAndSide(
                result.interface_marker,
                active_volume_side),
            context.get());
    const auto inactive_measure_summary =
        collectCutVolumeMeasures(
            mesh_access,
            context->generatedVolumeRulesForMarkerAndSide(
                result.interface_marker,
                inactive_volume_side),
            context.get());
    const auto global_active_physical_volume =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(active_measure_summary.physical_measure),
            comm));
    const auto global_active_physical_volume_rules =
        globalSumSize(active_measure_summary.physical_rule_count, comm);
    const auto global_active_skipped_physical_volume_rules =
        globalSumSize(active_measure_summary.skipped_physical_rule_count, comm);
    const auto global_inactive_physical_volume =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(inactive_measure_summary.physical_measure),
            comm));
    const auto global_inactive_physical_volume_rules =
        globalSumSize(inactive_measure_summary.physical_rule_count, comm);
    const auto global_inactive_skipped_physical_volume_rules =
        globalSumSize(inactive_measure_summary.skipped_physical_rule_count, comm);
    const auto global_negative_physical_volume =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(
                geometry_ledger.owned_unpruned_negative_physical_volume),
            comm));
    const auto global_positive_physical_volume =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(
                geometry_ledger.owned_unpruned_positive_physical_volume),
            comm));
    constexpr std::size_t global_negative_skipped_physical_volume_rules = 0u;
    constexpr std::size_t global_positive_skipped_physical_volume_rules = 0u;
    const auto global_snapshot_negative_physical_volume =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(
                geometry_ledger.owned_retained_negative_physical_volume),
            comm));
    const auto global_snapshot_positive_physical_volume =
        static_cast<svmp::FE::Real>(globalSumDouble(
            static_cast<double>(
                geometry_ledger.owned_retained_positive_physical_volume),
            comm));
    const auto global_snapshot_active_physical_volume =
        active_volume_side ==
                svmp::FE::geometry::CutIntegrationSide::Negative
            ? global_snapshot_negative_physical_volume
            : global_snapshot_positive_physical_volume;
    const auto global_snapshot_inactive_physical_volume =
        inactive_volume_side ==
                svmp::FE::geometry::CutIntegrationSide::Negative
            ? global_snapshot_negative_physical_volume
            : global_snapshot_positive_physical_volume;
    const auto active_constant_one_error = std::abs(
        global_snapshot_active_physical_volume -
        global_active_physical_volume);
    const auto inactive_constant_one_error = std::abs(
        global_snapshot_inactive_physical_volume -
        global_inactive_physical_volume);
    const auto constant_one_tolerance =
        svmp::FE::Real{1024.0} *
            std::numeric_limits<svmp::FE::Real>::epsilon() *
            std::max({svmp::FE::Real{1.0},
                      std::abs(global_snapshot_negative_physical_volume),
                      std::abs(global_snapshot_positive_physical_volume)}) +
        snapshot_policy.tolerance;
    const bool retains_inactive_volume = !retained_volume_sides.has_value();
    if (global_active_skipped_physical_volume_rules != 0u ||
        active_constant_one_error > constant_one_tolerance ||
        (retains_inactive_volume &&
         (global_inactive_skipped_physical_volume_rules != 0u ||
          inactive_constant_one_error > constant_one_tolerance))) {
      throw std::runtime_error(
          "Authoritative free-surface snapshot and constant-one cut-volume assembly measures disagree.");
    }
    report.negative_physical_volume += global_negative_physical_volume;
    report.positive_physical_volume += global_positive_physical_volume;
    const auto local_active_available_cut_volume_rules =
        active_volume_side == svmp::FE::geometry::CutIntegrationSide::Negative
            ? local_negative_available_cut_volume_rules
            : local_positive_available_cut_volume_rules;
    const auto local_inactive_available_cut_volume_rules =
        inactive_volume_side == svmp::FE::geometry::CutIntegrationSide::Negative
            ? local_negative_available_cut_volume_rules
            : local_positive_available_cut_volume_rules;
    const auto local_active_retained_cut_volume_rules =
        active_measure_summary.rule_count;
    const auto local_inactive_retained_cut_volume_rules =
        inactive_measure_summary.rule_count;
    const auto global_active_available_cut_volume_rules =
        globalSumSize(local_active_available_cut_volume_rules, comm);
    const auto global_inactive_available_cut_volume_rules =
        globalSumSize(local_inactive_available_cut_volume_rules, comm);
    const auto global_active_retained_cut_volume_rules =
        globalSumSize(local_active_retained_cut_volume_rules, comm);
    const auto global_inactive_retained_cut_volume_rules =
        globalSumSize(local_inactive_retained_cut_volume_rules, comm);
    const auto active_cut_volume_consumer_count =
        sim.fe_system->cutVolumeKernelCount(
            result.interface_marker, active_volume_side);
    const auto inactive_cut_volume_consumer_count =
        sim.fe_system->cutVolumeKernelCount(
            result.interface_marker, inactive_volume_side);
    const auto requireRetainedCutVolumeRules =
        [&](const char* logical_side,
            svmp::FE::geometry::CutIntegrationSide side,
            std::size_t consumer_count,
            std::size_t available_rule_count,
            std::size_t retained_rule_count) {
          if (consumer_count == 0u || available_rule_count == 0u ||
              retained_rule_count > 0u) {
            return;
          }
          throw std::runtime_error(
              std::string("[svMultiPhysics::Application] Generated cut-volume "
                          "consumer has no retained quadrature rules") +
              " marker=" + std::to_string(result.interface_marker) +
              " field='" + request.level_set_field_name + "'" +
              " domain_id='" + request.domain_id + "'" +
              " active_side=" + activeSideName(request.active_side) +
              " logical_side=" + logical_side +
              " cut_volume_side=" + cutIntegrationSideName(side) +
              " retained_volume_sides=" +
              retainedVolumeSidesName(request.volume_retention) +
              " cut_volume_consumer_count=" +
              std::to_string(consumer_count) +
              " available_cut_volume_rule_count=" +
              std::to_string(available_rule_count) +
              " retained_cut_volume_rule_count=" +
              std::to_string(retained_rule_count));
        };
    requireRetainedCutVolumeRules(
        "active",
        active_volume_side,
        active_cut_volume_consumer_count,
        global_active_available_cut_volume_rules,
        global_active_retained_cut_volume_rules);
    requireRetainedCutVolumeRules(
        "inactive",
        inactive_volume_side,
        inactive_cut_volume_consumer_count,
        global_inactive_available_cut_volume_rules,
        global_inactive_retained_cut_volume_rules);
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
    if (request.volume_retention ==
        ActiveCutVolumeRetention::ActiveAndInactive) {
      const auto inactive_side =
          request.active_side == LevelSetActiveSide::Negative
              ? LevelSetActiveSide::Positive
              : LevelSetActiveSide::Negative;
      (void)addGeneratedCutAdjacentFacetSet(
          *context, result.domain, mesh_access, inactive_side);
    }
    const auto facet_set_handle = addGeneratedCutAdjacentFacetSet(
        *context, result.domain, mesh_access, request.active_side);
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
    svmp::FE::logRecord(
        svmp::FE::LogRecord("LevelSetGeometryService", "active-domain cut context")
            .diagnostic("cut_context_rebuild")
            .value("provenance", (provenance != nullptr ? provenance : "unknown"))
            .value("solution_source", (solution_source != nullptr ? solution_source : "unknown"))
            .value("marker", result.interface_marker)
            .quoted("field", request.level_set_field_name)
            .quoted("domain_id", request.domain_id)
            .value("active_side", activeSideName(request.active_side))
            .value("retained_volume_sides", retainedVolumeSidesName(request.volume_retention))
            .text("interface_contract", "one_sided_embedded")
            .value("generated_interface_boundary_intersection_fragments", global_boundary_intersection_fragments)
            .value("generated_interface_boundary_intersection_active_fragments", global_active_boundary_intersection_fragments)
            .value("generated_interface_boundary_intersection_skipped_fragments", global_skipped_boundary_intersection_fragments)
            .value("generated_interface_boundary_intersection_qpoints", global_boundary_intersection_qpoints)
            .value("generated_interface_boundary_intersection_measure", global_boundary_intersection_measure)
            .value("generated_interface_geometry", svmp::FE::level_set::generatedInterfaceGeometryModeName(options.geometry_mode))
            .value("implicit_cut_quadrature_backend", svmp::FE::level_set::implicitCutQuadratureBackendName(options.implicit_cut_quadrature_backend))
            .value("selected_implicit_cut_quadrature_backend_counts", formatImplicitCutBackendCounts(global_selected_backend_counts))
            .value("implicit_cut_backend_qualification_counts", formatImplicitCutBackendQualificationCounts(global_backend_qualification_counts))
            .value("required_implicit_cut_backend_qualification", (options.require_production_qualified_implicit_cut_backend ? "ProductionQualified" : "none"))
            .value("implicit_cut_backend_seconds", backend_timing.local)
            .value("implicit_cut_backend_seconds_min", backend_timing.min)
            .value("implicit_cut_backend_seconds_mean", backend_timing.mean)
            .value("implicit_cut_backend_seconds_max", backend_timing.max)
            .value("implicit_cut_backend_internal_seconds", result.backend_elapsed_seconds)
            .value("implicit_cut_backend_internal_seconds_total", global_backend_elapsed_seconds)
            .text("generated_cell_work_scope", "summed_rank_local_including_ghost_cells")
            .value("generated_cell_cache_hits", global_generated_cell_cache_hits)
            .value("generated_cell_cache_misses", global_generated_cell_cache_misses)
            .value("generated_cell_cache_unchanged_dof_hits", global_generated_cell_cache_unchanged_dof_hits)
            .value("generated_cell_refresh_candidates", global_generated_cell_refresh_candidates)
            .value("generated_cell_directly_affected", global_generated_cell_directly_affected)
            .value("generated_cell_affected_neighborhood", global_generated_cell_affected_neighborhood)
            .value("generated_cell_affected_neighborhood_layers", result.affected_cell_neighborhood_layers)
            .value("generated_domain_cache_hits", global_generated_domain_cache_hits)
            .value("linear_full_cell_fast_path_cells", global_linear_full_cell_fast_path_cells)
            .value("implicit_cut_fallback_policy", svmp::FE::level_set::implicitCutFallbackPolicyName(options.implicit_cut_fallback_policy))
            .value("geometry_tangent_policy", svmp::FE::level_set::geometryTangentPolicyName(options.geometry_tangent_policy))
            .value("geometry_tangent_warning", (high_order_refreshed_frozen_tangent ? "quadrature_sensitivities_omitted" : "none"))
            .value("implicit_cut_root_tolerance", options.implicit_cut_root_tolerance)
            .value("implicit_cut_root_coordinate_tolerance", options.implicit_cut_root_coordinate_tolerance)
            .value("implicit_cut_root_max_iterations", options.implicit_cut_root_max_iterations)
            .value("implicit_cut_max_subdivision_depth", options.implicit_cut_max_subdivision_depth)
            .value("affected_cell_neighborhood_layers", options.affected_cell_neighborhood_layers)
            .value("generated_cut_ghost_layers", kept_ghost_cut_layers)
            .value("withheld_ghost_cut_fragments", global_withheld_ghost_cut_fragments)
            .value("withheld_ghost_cut_regions", global_withheld_ghost_cut_regions)
            .value("aligned_zero_interface_parent_side", cutIntegrationSideName(options.aligned_zero_interface_parent_side))
            .value("quadrature_order", options.quadrature_order)
            .value("interface_quadrature_order", options.interface_quadrature_order)
            .value("volume_quadrature_order", options.volume_quadrature_order)
            .value("achieved_interface_quadrature_order", result.achieved_interface_quadrature_order)
            .value("achieved_volume_quadrature_order", result.achieved_volume_quadrature_order)
            .value("implicit_cut_fallback_cells", global_implicit_cut_fallback_cells)
            .value("backend_volume_quadrature_point_count", global_backend_volume_quadrature_points)
            .value("backend_interface_quadrature_point_count", global_backend_interface_quadrature_points)
            .value("backend_total_quadrature_point_count", (global_backend_volume_quadrature_points + global_backend_interface_quadrature_points))
            .value("allow_corner_linearized_geometry", (request.allow_corner_linearized_geometry ? "true" : "false"))
            .value("isovalue", request.isovalue)
            .value("cut_context_revision", result.value_revision)
            .value("cut_context_topology_key", topology_key)
            .value("active_cut_request_policy_key", report.request_policy_key)
            .value("quadrature_policy_key", domain_request.quadrature_policy_key)
            .value("source_layout_revision", domain_request.source.layout_revision)
            .value("source_value_revision", domain_request.source.value_revision)
            .value("mesh_geometry_revision", domain_request.mesh_geometry_revision)
            .value("mesh_topology_revision", domain_request.mesh_topology_revision)
            .value("ownership_revision", domain_request.ownership_revision)
            .value("cell_count", global_cell_count)
            .value("corner_linearized_cells", global_corner_linearized_cells)
            .value("active_side_volume", global_active_volume)
            .text("active_side_volume_frame", "reference")
            .value("active_side_volume_local", active_volume)
            .value("active_side_physical_volume", global_active_physical_volume)
            .value("snapshot_active_side_physical_volume", global_snapshot_active_physical_volume)
            .value("active_constant_one_measure_error", active_constant_one_error)
            .value("constant_one_measure_tolerance", constant_one_tolerance)
            .value("active_side_physical_rule_count", global_active_physical_volume_rules)
            .value("active_side_skipped_physical_rule_count", global_active_skipped_physical_volume_rules)
            .value("active_side_available_cut_volume_rule_count", global_active_available_cut_volume_rules)
            .value("active_side_retained_cut_volume_rule_count", global_active_retained_cut_volume_rules)
            .value("active_side_cut_volume_consumer_count", active_cut_volume_consumer_count)
            .value("inactive_side_physical_volume", global_inactive_physical_volume)
            .value("snapshot_inactive_side_physical_volume", global_snapshot_inactive_physical_volume)
            .value("inactive_constant_one_measure_error", inactive_constant_one_error)
            .value("inactive_side_physical_rule_count", global_inactive_physical_volume_rules)
            .value("inactive_side_skipped_physical_rule_count", global_inactive_skipped_physical_volume_rules)
            .value("inactive_side_available_cut_volume_rule_count", global_inactive_available_cut_volume_rules)
            .value("inactive_side_retained_cut_volume_rule_count", global_inactive_retained_cut_volume_rules)
            .value("inactive_side_cut_volume_consumer_count", inactive_cut_volume_consumer_count)
            .value("active_side_raw_volume", global_raw_active_volume)
            .value("active_side_raw_volume_local", raw_active_volume)
            .value("interface_fragments", global_interface_fragments)
            .value("active_interface_fragments", global_active_interface_fragments)
            .value("interface_rule_count", global_active_interface_fragments)
            .value("interface_quadrature_point_count", global_interface_quadrature_points)
            .value("domain_interface_quadrature_point_count", global_interface_quadrature_points)
            .value("domain_volume_quadrature_point_count", global_domain_volume_quadrature_points)
            .value("domain_total_quadrature_point_count", global_domain_total_quadrature_points)
            .value("active_volume_regions", global_active_volume_regions)
            .value("active_volume_rule_count", global_active_volume_regions)
            .value("active_raw_volume_regions", global_raw_active_volume_regions)
            .value("active_pruned_volume_regions", global_pruned_volume_regions)
            .value("active_pruned_volume", global_pruned_volume)
            .value("generated_volume_prune_min_fraction", svmp::FE::assembly::CutIntegrationContext:: minGeneratedCutVolumeFraction())
            .text("generated_pruned_volume_scope", "summed_rank_local_including_ghost_cells")
            .value("generated_pruned_volume_rules", global_generated_pruned_volume_rules)
            .value("generated_pruned_volume", global_generated_pruned_volume)
            .value("active_wet_cells", global_active_wet_cells)
            .value("active_cut_cells", global_cut_cells)
            .value("active_full_wet_cells", global_full_wet_cells)
            .value("active_full_dry_cells", global_full_dry_cells)
            .value("active_quadrature_points", global_active_quadrature_points)
            .value("active_volume_quadrature_point_count", global_active_quadrature_points)
            .value("active_empty_quadrature_regions", global_empty_quadrature_regions)
            .value("active_nonfinite_measure_regions", global_nonfinite_measure_regions)
            .value("active_negative_measure_regions", global_negative_measure_regions)
            .value("active_min_volume_fraction", global_min_volume_fraction)
            .value("active_max_volume_fraction", global_max_volume_fraction)
            .text("cut_adjacent_scope", "summed_rank_local_including_ghost_cells")
            .value("cut_adjacent_facets", global_cut_adjacent_facets)
            .value("cut_adjacent_metadata", global_cut_adjacent_metadata)
            .value("cut_adjacent_zero_scale", global_zero_scale_count)
            .value("cut_adjacent_nonfinite_scale", global_nonfinite_scale_count)
            .value("cut_adjacent_capped_scale", global_capped_scale_count)
            .value("cut_adjacent_min_scale", global_min_scale)
            .value("cut_adjacent_max_scale", global_max_scale)
            .value("cut_adjacent_mean_scale", global_mean_scale)
            .value("process_vm_kb", memory.vm_kb)
            .value("process_rss_kb", memory.rss_kb)
            .value("basis_cache_entries", svmp::FE::basis::BasisCache::instance().size())
            .value("negative_volume", global_negative_volume)
            .value("negative_reference_volume", global_negative_volume)
            .value("negative_physical_volume", global_negative_physical_volume)
            .value("negative_skipped_physical_rule_count", global_negative_skipped_physical_volume_rules)
            .value("negative_volume_local", summary.negative_volume_measure)
            .value("positive_volume", global_positive_volume)
            .value("positive_reference_volume", global_positive_volume)
            .value("positive_physical_volume", global_positive_physical_volume)
            .value("positive_skipped_physical_rule_count", global_positive_skipped_physical_volume_rules)
            .value("positive_volume_local", summary.positive_volume_measure)
            .rootOnly());
  }

  report.topology_key =
      activeCutContextTopologyFingerprint(topology_bindings, comm);
  if (cutGeometryMemoryReportEnabled()) {
    logCutGeometryMemory(
        "before_install", [&](svmp::FE::LogRecord& record) {
          record.integer(
              "fe_context_transaction_active",
              sim.fe_system->cutIntegrationContextTransactionActive() ? 1 : 0);
          addCutGeometryLifecycleMemory(record, lifecycle);
          addCutGeometryContextMemory(record, "new", context.get());
          addCutGeometryContextMemory(
              record, "installed", sim.fe_system->cutIntegrationContext());
        });
  }
  report.installed_context = context;
  sim.fe_system->setCutIntegrationContext(std::move(context));
  logCutGeometryMemory("after_install");
  sim.fe_system->rebuildConstraintState();
  logCutGeometryMemory("after_constraint_rebuild");
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "active pressure support constraint refresh")
          .diagnostic("active_pressure_constraint_refresh")
          .value("provenance", (provenance != nullptr ? provenance : "unknown"))
          .value("solution_source", (solution_source != nullptr ? solution_source : "unknown"))
          .value("synchronized_level_set_fields", synchronized_level_set_fields)
          .text("support_source", "retained_cut_context")
          .value("constraints", sim.fe_system->constraints().numConstraints())
          .rootOnly());
  return report;
}

namespace {

bool activeCutContextMatchesRefreshCache(
    const svmp::FE::systems::FESystem& system,
    const ActiveCutContextRefreshCache& cache,
    const svmp::FE::systems::CollectiveCommunicator& comm,
    std::string_view phase)
{
  bool local_matches = false;
  std::exception_ptr local_validation_failure;
  try {
    const auto* context = system.cutIntegrationContext();
    if (context != nullptr &&
        context->freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(
            system.meshAccess()) &&
        cache.topology_key.has_value() &&
        *cache.topology_key != 0u &&
        !cache.evaluated_state_source_revisions.empty()) {
      local_matches = true;
      for (const auto& [marker, source_revision] :
           cache.evaluated_state_source_revisions) {
        if (source_revision == 0u ||
            !context->hasFreeSurfaceGeometrySnapshotForMarker(marker) ||
            !context->hasExpectedGeneratedSourceValueRevision(marker) ||
            context->expectedGeneratedSourceValueRevision(marker) !=
                source_revision) {
          local_matches = false;
          break;
        }
      }
    }
  } catch (...) {
    local_validation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_validation_failure, comm, phase);
  return !globalAnyBool(!local_matches, comm);
}

} // namespace

void observeActiveCutContextRefresh(
    const ActiveCutContextRefreshCache& cache,
    const ActiveCutContextRefreshReport& report,
    const char* provenance)
{
  if (cache.observer) {
    cache.observer(
        report,
        provenance != nullptr ? std::string_view{provenance}
                              : std::string_view{"unknown"});
  }
}

namespace {

const char* activeCutRefreshSignatureKindName(
    ActiveCutContextRefreshSignature::SolutionSignatureKind kind) noexcept
{
  switch (kind) {
    case ActiveCutContextRefreshSignature::SolutionSignatureKind::
        LevelSetFieldByteHash:
      return "level_set_field_byte_hash";
    case ActiveCutContextRefreshSignature::SolutionSignatureKind::
        VectorValueRevision:
      return "vector_value_revision";
  }
  return "unknown";
}

// Content key of the level-set field DOF layouts read by the generated cut
// geometry: field identity, monolithic offset, DOF count and the cell and
// vertex DOF maps. Unlike FESystem::dofLayoutRevision(), it does not change
// when setup() redistributes an identical layout.
std::uint64_t activeCutLevelSetDofLayoutKey(
    const svmp::FE::systems::FESystem& system,
    const std::vector<ActiveCutVolumeRequest>& requests)
{
  const auto mix = [](std::uint64_t& h, std::uint64_t value) noexcept {
    mixCutContextHash(h, avalancheCutTopologyFingerprint(value));
  };
  std::set<std::string> field_names;
  for (const auto& request : requests) {
    field_names.insert(request.level_set_field_name);
  }
  std::uint64_t h = kCutContextHashOffset;
  mix(h, static_cast<std::uint64_t>(field_names.size()));
  for (const auto& field_name : field_names) {
    mixCutContextHash(h, field_name);
    const auto field = system.findFieldByName(field_name);
    mix(h, static_cast<std::uint64_t>(field));
    if (field == svmp::FE::INVALID_FIELD_ID) {
      continue;
    }
    const auto& dofs = system.fieldDofHandler(field);
    mix(h, static_cast<std::uint64_t>(system.fieldDofOffset(field)));
    mix(h, static_cast<std::uint64_t>(dofs.getNumDofs()));
    const auto n_cells = dofs.getDofMap().getNumCells();
    mix(h, static_cast<std::uint64_t>(n_cells));
    for (svmp::FE::GlobalIndex cell = 0; cell < n_cells; ++cell) {
      const auto cell_dofs = dofs.getCellDofs(cell);
      mix(h, static_cast<std::uint64_t>(cell_dofs.size()));
      for (const auto dof : cell_dofs) {
        mix(h, static_cast<std::uint64_t>(dof));
      }
    }
    const auto* entity_map = dofs.getEntityDofMap();
    mix(h, entity_map != nullptr ? 1u : 0u);
    if (entity_map == nullptr) {
      continue;
    }
    const auto n_vertices = entity_map->numVertices();
    mix(h, static_cast<std::uint64_t>(n_vertices));
    for (svmp::FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
      const auto vertex_dofs = entity_map->getVertexDofs(vertex);
      mix(h, static_cast<std::uint64_t>(vertex_dofs.size()));
      for (const auto dof : vertex_dofs) {
        mix(h, static_cast<std::uint64_t>(dof));
      }
    }
  }
  return h;
}

void populateActiveCutContextRefreshDependencies(
    ActiveCutContextRefreshSignature& signature,
    const svmp::FE::systems::FESystem& system,
    const std::vector<ActiveCutVolumeRequest>& requests)
{
  const auto& mesh = system.meshAccess();
  signature.mesh_geometry_revision = mesh.geometryRevision();
  signature.mesh_topology_revision = mesh.topologyRevision();
  signature.mesh_ownership_revision = mesh.ownershipRevision();
  signature.mesh_numbering_revision = mesh.numberingRevision();
  signature.mesh_field_layout_revision = mesh.fieldLayoutRevision();
  signature.mesh_label_revision = mesh.labelRevision();
  signature.mesh_active_configuration_epoch =
      mesh.activeConfigurationEpoch();
  signature.mesh_coordinate_configuration_key =
      mesh.coordinateConfigurationKey();
  signature.system_space_revision = system.spaceRevision();
  signature.system_dof_layout_revision = system.dofLayoutRevision();
  signature.system_block_layout_revision = system.blockLayoutRevision();
  signature.level_set_dof_layout_key =
      activeCutLevelSetDofLayoutKey(system, requests);
}

void logActiveCutContextRefreshSkipped(
    const ActiveCutContextRefreshSignature& signature,
    const char* provenance,
    const char* solution_source,
    const char* skip_reason)
{
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "cut-context refresh skipped")
          .diagnostic("cut_context_refresh_skip")
          .value("provenance", (provenance != nullptr ? provenance : "unknown"))
          .value("solution_source", (solution_source != nullptr ? solution_source : "unknown"))
          .value("skip_reason", (skip_reason != nullptr ? skip_reason : "unknown"))
          .value("active_cut_request_policy_key", signature.request_policy_key)
          .value("signature_kind", activeCutRefreshSignatureKindName(signature.solution_signature_kind))
          .value("solution_hash", signature.solution_hash)
          .value("solution_size", signature.solution_size)
          .value("mesh_geometry_revision", signature.mesh_geometry_revision)
          .value("mesh_topology_revision", signature.mesh_topology_revision)
          .value("ownership_revision", signature.mesh_ownership_revision)
          .value("mesh_numbering_revision", signature.mesh_numbering_revision)
          .value("mesh_field_layout_revision", signature.mesh_field_layout_revision)
          .value("mesh_label_revision", signature.mesh_label_revision)
          .value("mesh_active_configuration_epoch", signature.mesh_active_configuration_epoch)
          .value("mesh_coordinate_configuration_key", signature.mesh_coordinate_configuration_key)
          .value("system_space_revision", signature.system_space_revision)
          .value("system_dof_layout_revision", signature.system_dof_layout_revision)
          .value("system_block_layout_revision", signature.system_block_layout_revision)
          .value("level_set_dof_layout_key", signature.level_set_dof_layout_key)
          .rootOnly());
}

// Diagnostic only: why a cached refresh could not keep the installed context
// (the first differing signature component, or a cache/context mismatch).
const char* activeCutRefreshMissReason(
    const std::optional<ActiveCutContextRefreshSignature>& signature,
    const ActiveCutContextRefreshCache& cache,
    bool context_matches_cache) noexcept
{
  if (!signature.has_value()) {
    return "signature_unavailable";
  }
  if (!cache.last_signature.has_value()) {
    return "cache_empty";
  }
  const auto& a = *signature;
  const auto& b = *cache.last_signature;
  if (a.request_policy_key != b.request_policy_key) {
    return "request_policy_key";
  }
  if (a.solution_signature_kind != b.solution_signature_kind) {
    return "signature_kind";
  }
  if (a.solution_size != b.solution_size) {
    return "solution_size";
  }
  if (a.solution_hash != b.solution_hash) {
    return "level_set_field_hash";
  }
  if (a.level_set_dof_layout_key != b.level_set_dof_layout_key) {
    return "level_set_dof_layout_key";
  }
  if (a.mesh_geometry_revision != b.mesh_geometry_revision ||
      a.mesh_topology_revision != b.mesh_topology_revision ||
      a.mesh_ownership_revision != b.mesh_ownership_revision ||
      a.mesh_numbering_revision != b.mesh_numbering_revision) {
    return "mesh_geometry_topology_revision";
  }
  if (a.mesh_field_layout_revision != b.mesh_field_layout_revision) {
    return "mesh_field_layout_revision";
  }
  if (a.mesh_label_revision != b.mesh_label_revision ||
      a.mesh_active_configuration_epoch !=
          b.mesh_active_configuration_epoch ||
      a.mesh_coordinate_configuration_key !=
          b.mesh_coordinate_configuration_key) {
    return "mesh_label_or_configuration";
  }
  if (a.system_space_revision != b.system_space_revision) {
    return "system_space_revision";
  }
  if (!(a == b)) {
    return "system_layout_revision";
  }
  if (!context_matches_cache) {
    return "installed_context_not_cached";
  }
  return "cache_disabled";
}

void logActiveCutContextRefreshMiss(
    const char* provenance,
    const char* solution_source,
    const char* reason)
{
  if (svmp::FE::rank_log::oopDiagnosticsEnabled(svmp::FE::LogLevel::DEBUG)) {
      svmp::FE::logRecord(
          svmp::FE::LogRecord("LevelSetGeometryService", "cut-context refresh miss")
              .diagnostic("cut_context_refresh_miss")
              .value("provenance", (provenance != nullptr ? provenance : "unknown"))
              .value("solution_source", (solution_source != nullptr ? solution_source : "unknown"))
              .value("reason", (reason != nullptr ? reason : "unknown"))
              .rootOnly());
  }
}

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

} // namespace

std::optional<ActiveCutContextRefreshSignature> activeCutContextRefreshSignature(
    const LevelSetGeometryHost& sim,
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
  ActiveCutContextRefreshSignature signature;
  signature.request_policy_key = activeCutVolumeRequestPolicyKey(requests);
  populateActiveCutContextRefreshDependencies(
      signature, *sim.fe_system, requests);
  signature.solution_signature_kind =
      ActiveCutContextRefreshSignature::SolutionSignatureKind::
          LevelSetFieldByteHash;
  signature.solution_size = fe_solution.size();
  signature.solution_hash = *solution_hash;
  return signature;
}

std::optional<ActiveCutContextRefreshSignature> activeCutContextRefreshSignature(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::backends::GenericVector& fe_solution)
{
  if (!sim.fe_system || requests.empty()) {
    return std::nullopt;
  }
  const auto raw_solution_size = fe_solution.size();
  if (raw_solution_size < 0) {
    return std::nullopt;
  }
  const auto solution_size = static_cast<std::size_t>(raw_solution_size);
  try {
    const auto values = fe_solution.localSpan();
    if (values.size() == solution_size) {
      const auto solution_hash =
          hashActiveCutLevelSetFieldBytes(*sim.fe_system, requests, values);
      if (solution_hash.has_value()) {
        ActiveCutContextRefreshSignature signature;
        signature.request_policy_key =
            activeCutVolumeRequestPolicyKey(requests);
        populateActiveCutContextRefreshDependencies(
            signature, *sim.fe_system, requests);
        signature.solution_signature_kind =
            ActiveCutContextRefreshSignature::SolutionSignatureKind::
                LevelSetFieldByteHash;
        signature.solution_size = solution_size;
        signature.solution_hash = *solution_hash;
        return signature;
      }
    }
  } catch (...) {
  }

  if (!activeCutLevelSetFieldLayoutIsAvailable(
          *sim.fe_system,
          requests,
          solution_size)) {
    return std::nullopt;
  }

  ActiveCutContextRefreshSignature signature;
  signature.request_policy_key = activeCutVolumeRequestPolicyKey(requests);
  populateActiveCutContextRefreshDependencies(
      signature, *sim.fe_system, requests);
  signature.solution_signature_kind =
      ActiveCutContextRefreshSignature::SolutionSignatureKind::
          VectorValueRevision;
  signature.solution_size = solution_size;
  signature.solution_hash = fe_solution.valueRevision();
  return signature;
}

namespace {

// Moves the installed context into the reinstall slot of an open reinstall
// window (see ActiveCutContextRefreshCache::reinstall_window_open), replacing
// any previously retained context. Called just before a rebuild replaces the
// installed context. The context is retained only when the cache describes
// it on every rank; the decision is collective.
void retainInstalledActiveCutContextForReinstall(
    const LevelSetGeometryHost& sim,
    ActiveCutContextRefreshCache& cache)
{
  cache.reinstall_candidate.reset();
  if (!cache.reinstall_window_open || !sim.fe_system) {
    return;
  }
  auto context = cache.last_context.lock();
  const bool local_retainable =
      context != nullptr && cache.last_signature.has_value() &&
      context.get() == sim.fe_system->cutIntegrationContext() &&
      !cache.evaluated_state_source_revisions.empty() &&
      cache.topology_key.has_value() && *cache.topology_key != 0u;
  if (globalAnyBool(!local_retainable,
                    activeFESystemCommunicator(*sim.fe_system))) {
    return;
  }
  ActiveCutContextRefreshCache::RetainedContext retained;
  retained.signature = *cache.last_signature;
  retained.context = std::move(context);
  retained.evaluated_state_source_revisions =
      cache.evaluated_state_source_revisions;
  retained.topology_key = cache.topology_key;
  cache.reinstall_candidate = std::move(retained);
}

// Reinstalls the context retained in an open reinstall window when the
// requested state provably has the same generated-geometry input: an equal
// content signature (level-set bytes, level-set DOF layout content, mesh
// revisions other than the field-layout revision, request policy) and a
// retained context whose snapshots and source revisions are still current.
// As in reinstallRetainedActiveCutContextForRestoredState, the mesh vertex
// field is rewritten from the requested state and the constraints are
// rebuilt exactly as after a rebuild of the same geometry. The installed
// context, when the cache describes it, becomes the new reinstall candidate.
// Every decision is collective. Returns nullopt when the caller must rebuild.
std::optional<ActiveCutContextRefreshReport>
reinstallActiveCutContextCandidate(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::uint64_t request_policy_key,
    std::span<const svmp::FE::Real> fe_solution,
    const std::optional<ActiveCutContextRefreshSignature>& signature,
    ActiveCutContextRefreshCache& cache,
    const char* provenance,
    const char* solution_source)
{
  if (!sim.fe_system || requests.empty()) {
    return std::nullopt;
  }
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const bool local_available =
      cache.reinstall_window_open &&
      cache.reinstall_candidate.has_value() &&
      cache.reinstall_candidate->context != nullptr &&
      signature.has_value() &&
      !parseBoolEnv("SVMP_DISABLE_ACTIVE_CUT_REFRESH_CACHE", false);
  if (globalAnyBool(!local_available, comm)) {
    return std::nullopt;
  }

  bool local_matches = false;
  std::exception_ptr local_match_failure;
  try {
    const auto& candidate = *cache.reinstall_candidate;
    // Every rebuild that changes the level set rewrites its mesh vertex
    // field and so advances the mesh field-layout revision. That revision is
    // not a geometry input, and the reinstall below rewrites the field from
    // the requested state exactly as a rebuild would, so it is excluded.
    auto comparable = candidate.signature;
    comparable.mesh_field_layout_revision =
        signature->mesh_field_layout_revision;
    if (*signature == comparable && candidate.topology_key.has_value() &&
        *candidate.topology_key != 0u &&
        !candidate.evaluated_state_source_revisions.empty() &&
        candidate.context->freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(
            sim.fe_system->meshAccess())) {
      local_matches = true;
      for (const auto& [marker, source_revision] :
           candidate.evaluated_state_source_revisions) {
        if (source_revision == 0u ||
            !candidate.context->hasFreeSurfaceGeometrySnapshotForMarker(
                marker) ||
            !candidate.context->hasExpectedGeneratedSourceValueRevision(
                marker) ||
            candidate.context->expectedGeneratedSourceValueRevision(
                marker) != source_revision) {
          local_matches = false;
          break;
        }
      }
    }
  } catch (...) {
    local_match_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_match_failure,
      comm,
      "active_cut_reinstall_candidate_before_reuse_consensus");
  if (globalAnyBool(!local_matches, comm)) {
    return std::nullopt;
  }

  auto reinstalled = std::move(*cache.reinstall_candidate);
  cache.reinstall_candidate.reset();
  // The context being replaced becomes the next candidate, so that the
  // geometry can be swapped back without a rebuild.
  retainInstalledActiveCutContextForReinstall(sim, cache);

  std::size_t synchronized_level_set_fields{0u};
  std::exception_ptr local_sync_failure;
  try {
    synchronized_level_set_fields =
        syncActiveLevelSetVertexFieldsFromSolution(sim, requests, fe_solution);
  } catch (...) {
    local_sync_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_sync_failure,
      comm,
      "active_cut_reinstall_source_sync");
  sim.fe_system->setCutIntegrationContext(reinstalled.context);
  sim.fe_system->rebuildConstraintState();
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "active pressure support constraint refresh")
          .diagnostic("active_pressure_constraint_refresh")
          .value("provenance", (provenance != nullptr ? provenance : "unknown"))
          .value("solution_source", (solution_source != nullptr ? solution_source : "unknown"))
          .value("synchronized_level_set_fields", synchronized_level_set_fields)
          .text("support_source", "retained_cut_context")
          .value("constraints", sim.fe_system->constraints().numConstraints())
          .rootOnly());
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "cut context reinstalled")
          .diagnostic("cut_context_reinstall")
          .value("provenance", (provenance != nullptr ? provenance : "unknown"))
          .value("solution_source", (solution_source != nullptr ? solution_source : "unknown"))
          .value("solution_hash", reinstalled.signature.solution_hash)
          .value("level_set_dof_layout_key", reinstalled.signature.level_set_dof_layout_key)
          .value("cut_context_topology_key", *reinstalled.topology_key)
          .value("active_cut_request_policy_key", request_policy_key)
          .rootOnly());

  // As after a rebuild, the cache records the signature of the state after
  // the mesh field was rewritten.
  std::optional<ActiveCutContextRefreshSignature> refreshed_signature;
  try {
    refreshed_signature =
        activeCutContextRefreshSignature(sim, requests, fe_solution);
  } catch (...) {
    refreshed_signature.reset();
  }
  cache.last_signature =
      refreshed_signature.has_value() ? *refreshed_signature : *signature;
  cache.last_vector_signature.reset();
  cache.evaluated_state_source_revisions =
      reinstalled.evaluated_state_source_revisions;
  cache.topology_key = reinstalled.topology_key;
  cache.last_context = reinstalled.context;

  ActiveCutContextRefreshReport report{};
  report.refreshed = true;
  report.request_policy_key = request_policy_key;
  report.topology_key = *reinstalled.topology_key;
  report.evaluated_state_source_revisions =
      reinstalled.evaluated_state_source_revisions;
  report.value_revision =
      reinstalled.evaluated_state_source_revisions.rbegin()->second;
  report.installed_context = reinstalled.context;
  return report;
}

// Keeps the installed context when the requested state differs from the one
// that generated it only in the mesh field-layout revision. That revision
// advances whenever any mesh field is rewritten (in parallel runs it advances
// between the accepted step and the next step without a level-set change),
// but it is not a geometry input: a refresh first rewrites the level-set
// vertex field from the requested state and generates the geometry from that
// state. The level-set bytes, the level-set DOF layout, the other mesh
// revisions and the request policy must be equal, and the installed context
// must match the cache's snapshot source revisions. As after a rebuild of the
// same geometry, the vertex field is rewritten, the context is installed
// again (running the context-update callbacks) and the constraint state is
// rebuilt. Every decision is collective. Returns nullopt when the caller
// must rebuild.
std::optional<ActiveCutContextRefreshReport>
resyncInstalledActiveCutContext(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::uint64_t request_policy_key,
    std::span<const svmp::FE::Real> fe_solution,
    const std::optional<ActiveCutContextRefreshSignature>& signature,
    ActiveCutContextRefreshCache& cache,
    bool context_matches_cache,
    const char* provenance,
    const char* solution_source)
{
  if (!sim.fe_system || requests.empty()) {
    return std::nullopt;
  }
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  auto context = cache.last_context.lock();
  bool local_resyncable = false;
  if (context_matches_cache && signature.has_value() &&
      cache.last_signature.has_value() && context != nullptr &&
      context.get() == sim.fe_system->cutIntegrationContext() &&
      cache.topology_key.has_value() && *cache.topology_key != 0u &&
      !cache.evaluated_state_source_revisions.empty() &&
      !parseBoolEnv("SVMP_DISABLE_ACTIVE_CUT_REFRESH_CACHE", false) &&
      !parseBoolEnv("SVMP_DISABLE_CUT_CONTEXT_RESYNC", false)) {
    auto comparable = *cache.last_signature;
    comparable.mesh_field_layout_revision =
        signature->mesh_field_layout_revision;
    local_resyncable = *signature == comparable;
  }
  if (globalAnyBool(!local_resyncable, comm)) {
    return std::nullopt;
  }

  std::size_t synchronized_level_set_fields{0u};
  std::exception_ptr local_sync_failure;
  try {
    synchronized_level_set_fields =
        syncActiveLevelSetVertexFieldsFromSolution(sim, requests, fe_solution);
  } catch (...) {
    local_sync_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_sync_failure,
      comm,
      "active_cut_resync_source_sync");
  sim.fe_system->setCutIntegrationContext(context);
  sim.fe_system->rebuildConstraintState();
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "active pressure support constraint refresh")
          .diagnostic("active_pressure_constraint_refresh")
          .value("provenance", (provenance != nullptr ? provenance : "unknown"))
          .value("solution_source", (solution_source != nullptr ? solution_source : "unknown"))
          .value("synchronized_level_set_fields", synchronized_level_set_fields)
          .text("support_source", "retained_cut_context")
          .value("constraints", sim.fe_system->constraints().numConstraints())
          .rootOnly());
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "cut context kept after a mesh field change")
          .diagnostic("cut_context_resync")
          .value("provenance", (provenance != nullptr ? provenance : "unknown"))
          .value("solution_source", (solution_source != nullptr ? solution_source : "unknown"))
          .value("solution_hash", signature->solution_hash)
          .value("cut_context_topology_key", *cache.topology_key)
          .value("active_cut_request_policy_key", request_policy_key)
          .rootOnly());

  // As after a rebuild, the cache records the signature of the state after
  // the mesh field was rewritten.
  std::optional<ActiveCutContextRefreshSignature> refreshed_signature;
  try {
    refreshed_signature =
        activeCutContextRefreshSignature(sim, requests, fe_solution);
  } catch (...) {
    refreshed_signature.reset();
  }
  cache.last_signature =
      refreshed_signature.has_value() ? *refreshed_signature : *signature;
  cache.last_vector_signature.reset();

  ActiveCutContextRefreshReport report{};
  report.refreshed = true;
  report.request_policy_key = request_policy_key;
  report.topology_key = *cache.topology_key;
  report.evaluated_state_source_revisions =
      cache.evaluated_state_source_revisions;
  report.value_revision =
      cache.evaluated_state_source_revisions.rbegin()->second;
  report.installed_context = context;
  return report;
}

} // namespace

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextFromSolutionCached(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    std::span<const svmp::FE::Real> fe_solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance,
    const char* solution_source)
{
  const auto& requests = active_cut_requests;
  ActiveCutContextRefreshReport skipped_report{};
  if (!sim.fe_system) {
    if (!requests.empty()) {
      throw std::runtime_error(
          "Cached active-cut refresh has configured requests but no FE system; communicator consensus is unavailable.");
    }
    return skipped_report;
  }
  skipped_report.request_policy_key =
      requireActiveCutRequestConsensus(*sim.fe_system, requests);
  if (requests.empty()) {
    return skipped_report;
  }
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const auto signature =
      activeCutContextRefreshSignature(sim, requests, fe_solution);
  const bool disable_refresh_cache =
      parseBoolEnv("SVMP_DISABLE_ACTIVE_CUT_REFRESH_CACHE", false);
  const bool context_matches_cache = activeCutContextMatchesRefreshCache(
      *sim.fe_system,
      cache,
      comm,
      "active_cut_span_context_before_cache_consensus");
  const bool local_can_skip =
      !disable_refresh_cache &&
      signature.has_value() &&
      cache.last_signature.has_value() &&
      *signature == *cache.last_signature &&
      context_matches_cache;
  if (!globalAnyBool(!local_can_skip, comm)) {
    skipped_report.evaluated_state_source_revisions =
        cache.evaluated_state_source_revisions;
    skipped_report.topology_key =
        cache.topology_key.value_or(0u);
    skipped_report.reused_installed_context = true;
    logActiveCutContextRefreshSkipped(
        *signature,
        provenance,
        solution_source,
        "level_set_field_hash_unchanged");
    observeActiveCutContextRefresh(cache, skipped_report, provenance);
    return skipped_report;
  }

  logActiveCutContextRefreshMiss(
      provenance,
      solution_source,
      activeCutRefreshMissReason(signature, cache, context_matches_cache));
  if (auto resynced = resyncInstalledActiveCutContext(
          sim,
          requests,
          skipped_report.request_policy_key,
          fe_solution,
          signature,
          cache,
          context_matches_cache,
          provenance,
          solution_source)) {
    observeActiveCutContextRefresh(cache, *resynced, provenance);
    return *resynced;
  }
  if (auto reinstalled = reinstallActiveCutContextCandidate(
          sim,
          requests,
          skipped_report.request_policy_key,
          fe_solution,
          signature,
          cache,
          provenance,
          solution_source)) {
    observeActiveCutContextRefresh(cache, *reinstalled, provenance);
    return *reinstalled;
  }
  retainInstalledActiveCutContextForReinstall(sim, cache);
  auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      active_cut_requests,
      fe_solution,
      lifecycle,
      provenance,
      solution_source,
      true);
  if (report.refreshed) {
    const auto refreshed_signature =
        activeCutContextRefreshSignature(sim, requests, fe_solution);
    if (refreshed_signature.has_value()) {
      cache.last_signature = *refreshed_signature;
    } else if (signature.has_value()) {
      cache.last_signature = *signature;
    }
    cache.last_vector_signature.reset();
    cache.evaluated_state_source_revisions =
        report.evaluated_state_source_revisions;
    cache.topology_key = report.topology_key;
    cache.last_context = report.installed_context;
  }
  observeActiveCutContextRefresh(cache, report, provenance);
  return report;
}

namespace {

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCachedFromVector(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    svmp::FE::backends::GenericVector& solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance,
    const char* solution_source)
{
  const auto& requests = active_cut_requests;
  ActiveCutContextRefreshReport skipped_report{};
  if (!sim.fe_system) {
    if (!requests.empty()) {
      throw std::runtime_error(
          "Cached active-cut vector refresh has configured requests but no FE system; communicator consensus is unavailable.");
    }
    return skipped_report;
  }
  skipped_report.request_policy_key =
      requireActiveCutRequestConsensus(*sim.fe_system, requests);
  if (requests.empty()) {
    return skipped_report;
  }
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  std::optional<ActiveCutContextRefreshSignature> vector_signature;
  std::exception_ptr local_vector_signature_failure;
  try {
    vector_signature =
        activeCutContextRefreshSignature(sim, requests, solution);
  } catch (...) {
    local_vector_signature_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_vector_signature_failure,
      comm,
      "active_cut_vector_signature_before_cache_consensus");
  const bool disable_refresh_cache =
      parseBoolEnv("SVMP_DISABLE_ACTIVE_CUT_REFRESH_CACHE", false);
  const bool context_matches_cache = activeCutContextMatchesRefreshCache(
      *sim.fe_system,
      cache,
      comm,
      "active_cut_vector_context_before_cache_consensus");
  const bool local_vector_can_skip =
      !disable_refresh_cache &&
      vector_signature.has_value() &&
      cache.last_vector_signature.has_value() &&
      *vector_signature == *cache.last_vector_signature &&
      context_matches_cache;
  if (!globalAnyBool(!local_vector_can_skip, comm)) {
    skipped_report.evaluated_state_source_revisions =
        cache.evaluated_state_source_revisions;
    skipped_report.topology_key =
        cache.topology_key.value_or(0u);
    skipped_report.reused_installed_context = true;
    const char* skip_reason =
        vector_signature->solution_signature_kind ==
                ActiveCutContextRefreshSignature::SolutionSignatureKind::
                    LevelSetFieldByteHash
            ? "level_set_field_hash_unchanged"
            : "vector_value_revision_unchanged";
    logActiveCutContextRefreshSkipped(
        *vector_signature,
        provenance,
        solution_source,
        skip_reason);
    observeActiveCutContextRefresh(cache, skipped_report, provenance);
    return skipped_report;
  }

  std::vector<svmp::FE::Real> fe_solution;
  std::optional<ActiveCutContextRefreshSignature> signature;
  std::exception_ptr local_gather_failure;
  try {
    fe_solution = gatherFeOrderedSolution(solution, comm);
    signature = activeCutContextRefreshSignature(
        sim,
        requests,
        std::span<const svmp::FE::Real>(
            fe_solution.data(), fe_solution.size()));
  } catch (...) {
    local_gather_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_gather_failure,
      comm,
      "active_cut_vector_gather_before_field_hash_consensus");
  const bool local_fe_can_skip =
      !disable_refresh_cache &&
      signature.has_value() &&
      cache.last_signature.has_value() &&
      *signature == *cache.last_signature &&
      context_matches_cache;
  if (!globalAnyBool(!local_fe_can_skip, comm)) {
    skipped_report.evaluated_state_source_revisions =
        cache.evaluated_state_source_revisions;
    skipped_report.topology_key =
        cache.topology_key.value_or(0u);
    skipped_report.reused_installed_context = true;
    if (vector_signature.has_value()) {
      cache.last_vector_signature = *vector_signature;
    }
    logActiveCutContextRefreshSkipped(
        *signature,
        provenance,
        solution_source,
        "level_set_field_hash_unchanged");
    observeActiveCutContextRefresh(cache, skipped_report, provenance);
    return skipped_report;
  }

  logActiveCutContextRefreshMiss(
      provenance,
      solution_source,
      activeCutRefreshMissReason(signature, cache, context_matches_cache));
  if (auto resynced = resyncInstalledActiveCutContext(
          sim,
          requests,
          skipped_report.request_policy_key,
          std::span<const svmp::FE::Real>(
              fe_solution.data(), fe_solution.size()),
          signature,
          cache,
          context_matches_cache,
          provenance,
          solution_source)) {
    const auto refreshed_vector_signature =
        activeCutContextRefreshSignature(sim, requests, solution);
    if (refreshed_vector_signature.has_value()) {
      cache.last_vector_signature = *refreshed_vector_signature;
    } else if (vector_signature.has_value()) {
      cache.last_vector_signature = *vector_signature;
    }
    observeActiveCutContextRefresh(cache, *resynced, provenance);
    return *resynced;
  }
  if (auto reinstalled = reinstallActiveCutContextCandidate(
          sim,
          requests,
          skipped_report.request_policy_key,
          std::span<const svmp::FE::Real>(
              fe_solution.data(), fe_solution.size()),
          signature,
          cache,
          provenance,
          solution_source)) {
    const auto refreshed_vector_signature =
        activeCutContextRefreshSignature(sim, requests, solution);
    if (refreshed_vector_signature.has_value()) {
      cache.last_vector_signature = *refreshed_vector_signature;
    } else if (vector_signature.has_value()) {
      cache.last_vector_signature = *vector_signature;
    }
    observeActiveCutContextRefresh(cache, *reinstalled, provenance);
    return *reinstalled;
  }
  retainInstalledActiveCutContextForReinstall(sim, cache);
  auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      active_cut_requests,
      std::span<const svmp::FE::Real>(fe_solution.data(), fe_solution.size()),
      lifecycle,
      provenance,
      solution_source,
      true);
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
    cache.evaluated_state_source_revisions =
        report.evaluated_state_source_revisions;
    cache.topology_key = report.topology_key;
    cache.last_context = report.installed_context;
  }
  observeActiveCutContextRefresh(cache, report, provenance);
  return report;
}

} // namespace

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCached(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    svmp::FE::backends::GenericVector& solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance)
{
  return refreshActiveCutIntegrationContextCachedFromVector(
      sim,
      active_cut_requests,
      solution,
      lifecycle,
      cache,
      provenance,
      "fe_vector");
}

ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCached(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
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
        active_cut_requests,
        *solution,
        lifecycle,
        cache,
        provenance,
        solution_source);
  }
  const auto fe_solution = gatherFeOrderedSolution(
      state, activeFESystemCommunicator(*sim.fe_system));
  return refreshActiveCutIntegrationContextFromSolutionCached(
      sim,
      active_cut_requests,
      std::span<const svmp::FE::Real>(fe_solution.data(), fe_solution.size()),
      lifecycle,
      cache,
      provenance,
      solution_source);
}

void armActiveCutRestoreRetention(
    ActiveCutContextRefreshCache& cache) noexcept
{
  cache.retained_restore_context.reset();
  cache.restore_pin_state =
      ActiveCutContextRefreshCache::RestorePinState::Armed;
  cache.closeReinstallWindow();
}

void releaseActiveCutRestoreRetention(
    ActiveCutContextRefreshCache& cache) noexcept
{
  cache.retained_restore_context.reset();
  cache.restore_pin_state =
      ActiveCutContextRefreshCache::RestorePinState::Idle;
  cache.closeReinstallWindow();
}

// Called after every outer fixed-point refresh. Between
// armActiveCutRestoreRetention() (before_physics_solve) and the second outer
// pass of the attempt, it retains the installed context, so that at the end
// of the entry synchronization the cache holds the geometry of the canonical
// entry state, which RestoredOuterFixedPointState rolls back to. Only with
// SVMP_RETAIN_RESTORED_CUT_CONTEXT=1 (see ActiveCutContextRefreshCache).
void retainActiveCutContextAtAttemptEntry(
    const LevelSetGeometryHost& sim,
    ActiveCutContextRefreshCache& cache,
    bool outer_fixed_point_state,
    bool projected_outer_fixed_point_state)
{
  using PinState = ActiveCutContextRefreshCache::RestorePinState;
  if (!parseBoolEnv("SVMP_RETAIN_RESTORED_CUT_CONTEXT", false)) {
    cache.retained_restore_context.reset();
    cache.restore_pin_state = PinState::Idle;
    return;
  }
  if (outer_fixed_point_state) {
    if (cache.restore_pin_state == PinState::Armed) {
      cache.restore_pin_state = PinState::Pinning;
    } else if (cache.restore_pin_state == PinState::Pinning) {
      // A later outer pass: the entry geometry is already retained.
      cache.restore_pin_state = PinState::Idle;
      return;
    }
  }
  if (cache.restore_pin_state != PinState::Pinning ||
      (!outer_fixed_point_state && !projected_outer_fixed_point_state)) {
    return;
  }
  auto context = cache.last_context.lock();
  if (!sim.fe_system || !context || !cache.last_signature.has_value() ||
      context.get() != sim.fe_system->cutIntegrationContext() ||
      cache.evaluated_state_source_revisions.empty() ||
      !cache.topology_key.has_value() || *cache.topology_key == 0u) {
    cache.retained_restore_context.reset();
    return;
  }
  ActiveCutContextRefreshCache::RetainedContext retained;
  retained.signature = *cache.last_signature;
  retained.context = std::move(context);
  retained.evaluated_state_source_revisions =
      cache.evaluated_state_source_revisions;
  retained.topology_key = cache.topology_key;
  cache.retained_restore_context = std::move(retained);
}

// Reinstalls the retained entry geometry for a restored state when the
// restored state provably has the same generated-geometry input: an equal
// content signature (level-set bytes, level-set DOF layout content, mesh
// revisions and request policy) and a retained context whose snapshots and
// source revisions are still current. The installed context is replaced and
// the constraints are rebuilt exactly as after a refresh that rebuilds the
// same geometry. Every decision is collective. Returns nullopt when the
// caller must refresh normally (which also covers an installed context that
// already has this content).
std::optional<ActiveCutContextRefreshReport>
reinstallRetainedActiveCutContextForRestoredState(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    const svmp::FE::systems::SystemStateView& state,
    ActiveCutContextRefreshCache& cache,
    const char* provenance)
{
  if (!sim.fe_system) {
    return std::nullopt;
  }
  const auto& requests = active_cut_requests;
  const auto request_policy_key =
      requireActiveCutRequestConsensus(*sim.fe_system, requests);
  if (requests.empty()) {
    return std::nullopt;
  }
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const bool local_available =
      cache.retained_restore_context.has_value() &&
      cache.retained_restore_context->context != nullptr &&
      !parseBoolEnv("SVMP_DISABLE_ACTIVE_CUT_REFRESH_CACHE", false) &&
      parseBoolEnv("SVMP_RETAIN_RESTORED_CUT_CONTEXT", false);
  if (globalAnyBool(!local_available, comm)) {
    return std::nullopt;
  }

  std::vector<svmp::FE::Real> fe_solution;
  std::optional<ActiveCutContextRefreshSignature> signature;
  bool local_retained_matches = false;
  std::exception_ptr local_signature_failure;
  try {
    fe_solution = gatherFeOrderedSolution(state, comm);
    signature = activeCutContextRefreshSignature(
        sim,
        requests,
        std::span<const svmp::FE::Real>(
            fe_solution.data(), fe_solution.size()));
    const auto& retained = *cache.retained_restore_context;
    // Every rebuild that changes the level set rewrites its mesh vertex
    // field and so advances the mesh field-layout revision. That revision is
    // not a geometry input, and the reinstall below rewrites the field from
    // the restored state exactly as a rebuild would, so it is excluded here.
    std::optional<ActiveCutContextRefreshSignature> comparable;
    if (signature.has_value()) {
      comparable = retained.signature;
      comparable->mesh_field_layout_revision =
          signature->mesh_field_layout_revision;
    }
    if (comparable.has_value() && *signature == *comparable &&
        retained.topology_key.has_value() &&
        *retained.topology_key != 0u &&
        !retained.evaluated_state_source_revisions.empty() &&
        retained.context->freeSurfaceGeometrySnapshotsMatchCurrentMeshRevision(
            sim.fe_system->meshAccess())) {
      local_retained_matches = true;
      for (const auto& [marker, source_revision] :
           retained.evaluated_state_source_revisions) {
        if (source_revision == 0u ||
            !retained.context->hasFreeSurfaceGeometrySnapshotForMarker(
                marker) ||
            !retained.context->hasExpectedGeneratedSourceValueRevision(
                marker) ||
            retained.context->expectedGeneratedSourceValueRevision(
                marker) != source_revision) {
          local_retained_matches = false;
          break;
        }
      }
    }
  } catch (...) {
    local_signature_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_signature_failure,
      comm,
      "restored_active_cut_signature_before_reuse_consensus");
  const bool installed_context_matches =
      activeCutContextMatchesRefreshCache(
          *sim.fe_system,
          cache,
          comm,
          "restored_active_cut_installed_context_before_reuse_consensus") &&
      signature.has_value() && cache.last_signature.has_value() &&
      *signature == *cache.last_signature;
  if (!globalAnyBool(!installed_context_matches, comm) ||
      globalAnyBool(!local_retained_matches, comm)) {
    return std::nullopt;
  }

  const auto retained = *cache.retained_restore_context;
  std::size_t synchronized_level_set_fields{0u};
  std::exception_ptr local_sync_failure;
  try {
    synchronized_level_set_fields =
        syncActiveLevelSetVertexFieldsFromSolution(
            sim,
            requests,
            std::span<const svmp::FE::Real>(
                fe_solution.data(), fe_solution.size()));
  } catch (...) {
    local_sync_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_sync_failure,
      comm,
      "restored_active_cut_source_sync_before_reinstall");
  sim.fe_system->setCutIntegrationContext(retained.context);
  sim.fe_system->rebuildConstraintState();
  const char* solution_source =
      state.u_vector != nullptr ? "state_vector_fe_ordered"
                                : "state_span_assumed_fe_ordered";
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "active pressure support constraint refresh")
          .diagnostic("active_pressure_constraint_refresh")
          .value("provenance", (provenance != nullptr ? provenance : "unknown"))
          .value("solution_source", solution_source)
          .value("synchronized_level_set_fields", synchronized_level_set_fields)
          .text("support_source", "retained_cut_context")
          .value("constraints", sim.fe_system->constraints().numConstraints())
          .rootOnly());
  svmp::FE::LogRecord reuse_line("LevelSetGeometryService", "restored-state cut context reused");
  reuse_line.diagnostic("cut_context_restored_reuse")
      .value("provenance", (provenance != nullptr ? provenance : "unknown"))
      .value("solution_source", solution_source)
      .value("solution_hash", retained.signature.solution_hash)
      .value("level_set_dof_layout_key", retained.signature.level_set_dof_layout_key)
      .value("cut_context_topology_key", *retained.topology_key)
      .value("active_cut_request_policy_key", request_policy_key)
      .rootOnly();
  for (const auto& [marker, source_revision] :
       retained.evaluated_state_source_revisions) {
    reuse_line.child("cut_context_restored_reuse_marker", {})
        .value("marker", marker)
        .value("source_value_revision", source_revision);
  }
  svmp::FE::logRecord(reuse_line);

  // As after a rebuild, the cache records the signature of the state after
  // the mesh field was rewritten.
  std::optional<ActiveCutContextRefreshSignature> refreshed_signature;
  try {
    refreshed_signature = activeCutContextRefreshSignature(
        sim,
        requests,
        std::span<const svmp::FE::Real>(
            fe_solution.data(), fe_solution.size()));
  } catch (...) {
    refreshed_signature.reset();
  }
  cache.last_signature =
      refreshed_signature.has_value() ? *refreshed_signature : *signature;
  cache.last_vector_signature.reset();
  cache.evaluated_state_source_revisions =
      retained.evaluated_state_source_revisions;
  cache.topology_key = retained.topology_key;
  cache.last_context = retained.context;

  ActiveCutContextRefreshReport report{};
  report.refreshed = true;
  report.request_policy_key = request_policy_key;
  report.topology_key = *retained.topology_key;
  report.evaluated_state_source_revisions =
      retained.evaluated_state_source_revisions;
  report.value_revision =
      retained.evaluated_state_source_revisions.rbegin()->second;
  report.installed_context = retained.context;
  observeActiveCutContextRefresh(cache, report, provenance);
  return report;
}

namespace {

PrescribedFieldCoefficientCheckpoint
capturePrescribedFieldCoefficients(
    const svmp::FE::systems::FESystem& system,
    std::span<const svmp::FE::FieldId> fields)
{
  PrescribedFieldCoefficientCheckpoint checkpoint;
  std::set<svmp::FE::FieldId> visited;
  for (const auto field : fields) {
    if (field == svmp::FE::INVALID_FIELD_ID ||
        !visited.insert(field).second) {
      if (field == svmp::FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Cannot checkpoint an invalid prescribed field.");
      }
      continue;
    }
    const auto& record = system.fieldRecord(field);
    if (record.source_kind !=
        svmp::FE::systems::FieldSourceKind::PrescribedData) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Geometry transaction field '" +
          record.name + "' is not prescribed data.");
    }
    const auto coefficients =
        system.prescribedFieldCoefficients(field);
    PrescribedFieldCoefficientCheckpoint::Field saved;
    saved.id = field;
    saved.had_coefficients = !coefficients.empty();
    saved.coefficients.assign(
        coefficients.begin(), coefficients.end());
    checkpoint.fields.push_back(std::move(saved));
  }
  return checkpoint;
}

void restorePrescribedFieldCoefficients(
    svmp::FE::systems::FESystem& system,
    const PrescribedFieldCoefficientCheckpoint& checkpoint)
{
  for (const auto& field : checkpoint.fields) {
    if (field.had_coefficients) {
      system.setPrescribedFieldCoefficients(
          field.id,
          std::span<const svmp::FE::Real>(
              field.coefficients.data(), field.coefficients.size()));
    } else {
      system.clearPrescribedFieldCoefficients(field.id);
    }
  }
}

} // namespace

// LevelSetGeometryTransaction (LevelSetMaintenanceGeometryTransaction in the
// driver), with the member functions defined out of the class.

LevelSetGeometryTransaction::LevelSetGeometryTransaction(
    LevelSetGeometryHost sim,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& refresh_cache,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::span<const svmp::FE::FieldId> prescribed_fields)
    : sim_(sim),
      lifecycle_(lifecycle),
      refresh_cache_(refresh_cache),
      requests_(requests)
{
  if (!sim_.fe_system) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance geometry transaction requires an FE system.");
  }
  const auto comm = activeFESystemCommunicator(*sim_.fe_system);
  std::exception_ptr local_checkpoint_failure;
  try {
    refresh_cache_backup_ = refresh_cache;
    mesh_field_checkpoint_ =
        captureActiveLevelSetMeshFields(sim, requests);
    prescribed_field_checkpoint_ =
        capturePrescribedFieldCoefficients(
            *sim_.fe_system, prescribed_fields);
  } catch (...) {
    local_checkpoint_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_checkpoint_failure,
      comm,
      "level_set_geometry_transaction_checkpoint");

  std::exception_ptr local_system_preflight_failure;
  try {
    if (!sim_.fe_system->isSetup()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set maintenance geometry transaction requires a setup FE system.");
    }
    if (sim_.fe_system->cutIntegrationContextTransactionActive()) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Cut-integration-context transaction is already active.");
    }
  } catch (...) {
    local_system_preflight_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_system_preflight_failure,
      comm,
      "level_set_geometry_transaction_system_preflight");

  std::exception_ptr local_lifecycle_failure;
  try {
    lifecycle_.beginTransaction();
    lifecycle_transaction_active_ = true;
  } catch (...) {
    local_lifecycle_failure = std::current_exception();
  }
  if (globalAnyBool(local_lifecycle_failure != nullptr, comm)) {
    std::exception_ptr local_cleanup_failure;
    if (lifecycle_transaction_active_) {
      try {
        lifecycle_.rollbackTransaction();
        lifecycle_transaction_active_ = false;
      } catch (...) {
        local_cleanup_failure = std::current_exception();
      }
    }
    const bool cleanup_failed =
        globalAnyBool(local_cleanup_failure != nullptr, comm);
    if (local_lifecycle_failure) {
      std::rethrow_exception(local_lifecycle_failure);
    }
    if (local_cleanup_failure) {
      std::rethrow_exception(local_cleanup_failure);
    }
    throw std::runtime_error(
        cleanup_failed
            ? "[svMultiPhysics::Application] Another active FE rank failed "
              "level-set geometry lifecycle begin and cleanup did not "
              "complete on every rank."
            : "[svMultiPhysics::Application] Another active FE rank failed "
              "level-set geometry lifecycle begin.");
  }

  std::exception_ptr local_system_failure;
  try {
    // FESystem coordinates its own rank-local backup preparation before it
    // publishes the transaction backup.
    sim_.fe_system->beginCutIntegrationContextTransaction();
    system_transaction_active_ = true;
  } catch (...) {
    local_system_failure = std::current_exception();
  }
  if (globalAnyBool(local_system_failure != nullptr, comm)) {
    std::exception_ptr local_cleanup_failure;
    try {
      lifecycle_.rollbackTransaction();
      lifecycle_transaction_active_ = false;
    } catch (...) {
      local_cleanup_failure = std::current_exception();
    }
    const bool cleanup_failed =
        globalAnyBool(local_cleanup_failure != nullptr, comm);
    if (local_system_failure) {
      std::rethrow_exception(local_system_failure);
    }
    if (local_cleanup_failure) {
      std::rethrow_exception(local_cleanup_failure);
    }
    throw std::runtime_error(
        cleanup_failed
            ? "[svMultiPhysics::Application] Another active FE rank failed "
              "cut-context transaction begin and lifecycle cleanup did not "
              "complete on every rank."
            : "[svMultiPhysics::Application] Another active FE rank failed "
              "cut-context transaction begin.");
  }
  active_ = true;
}

LevelSetGeometryTransaction::~LevelSetGeometryTransaction()
{
  if (!active_) {
    return;
  }
  try {
    rollback();
  } catch (...) {
  }
}

ActiveCutContextRefreshReport LevelSetGeometryTransaction::refresh(
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    std::span<const svmp::FE::Real> candidate,
    bool force_rebuild)
{
  if (!active_) {
    throw std::logic_error(
        "level-set maintenance geometry transaction is not active");
  }
  if (force_rebuild) {
    refresh_cache_.last_signature.reset();
    refresh_cache_.last_vector_signature.reset();
  }
  return refreshActiveCutIntegrationContextFromSolutionCached(
      sim_,
      active_cut_requests,
      candidate,
      lifecycle_,
      refresh_cache_,
      "accepted_step_maintenance_candidate",
      "staged_fe_solution");
}

ActiveCutContextRefreshReport LevelSetGeometryTransaction::refreshCandidate(
    std::span<const svmp::FE::Real> candidate,
    bool force_rebuild)
{
  return refresh(requests_, candidate, force_rebuild);
}

ConservativePhaseGeometryRefresh LevelSetGeometryTransaction::refresh(
    std::span<const svmp::FE::Real> candidate)
{
  return ConservativePhaseGeometryRefresh{
      .refreshed = refreshCandidate(candidate).refreshed,
  };
}

const std::vector<ActiveCutVolumeRequest>&
LevelSetGeometryTransaction::requests() const noexcept
{
  return requests_;
}

bool LevelSetGeometryTransaction::active() const noexcept
{
  return active_;
}

bool LevelSetGeometryTransaction::publicationStarted() const noexcept
{
  return publication_started_;
}

CurrentCandidateProducerLinkageResult
LevelSetGeometryTransaction::currentCandidateProducerLinkage(
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    std::span<const svmp::FE::Real> candidate,
    const ActiveCutContextRefreshReport& refresh_report) const
{
  const auto comm = activeFESystemCommunicator(*sim_.fe_system);
  const auto unavailable = [](const char* diagnostic) {
    return CurrentCandidateProducerLinkageResult{
        .success = false,
        .diagnostic = diagnostic,
    };
  };
  if (globalAnyBool(
          !active_ || !system_transaction_active_ ||
              !lifecycle_transaction_active_ || publication_started_ ||
              !sim_.fe_system->isSetup() ||
              !sim_.fe_system->cutIntegrationContextTransactionActive() ||
              !lifecycle_.transactionActive(),
          comm)) {
    return unavailable(
        "current_candidate_producer_linkage_transaction_inactive");
  }

  std::vector<ActiveCutVolumeRequest> requests;
  std::optional<ActiveCutContextRefreshSignature> signature;
  std::exception_ptr local_preparation_failure;
  try {
    requests = active_cut_requests;
    signature = activeCutContextRefreshSignature(
        sim_, requests, candidate);
  } catch (...) {
    local_preparation_failure = std::current_exception();
  }
  if (globalAnyBool(local_preparation_failure != nullptr, comm)) {
    return unavailable(
        "current_candidate_producer_linkage_preparation_failed");
  }
  if (globalAnyBool(requests.empty(), comm)) {
    return unavailable(
        "current_candidate_producer_linkage_active_requests_unavailable");
  }
  if (globalAnyBool(!signature.has_value(), comm)) {
    return unavailable(
        "current_candidate_producer_linkage_span_signature_unavailable");
  }
  if (globalAnyBool(!refresh_cache_.last_signature.has_value(), comm)) {
    return unavailable(
        "current_candidate_producer_linkage_cache_signature_unavailable");
  }
  if (globalAnyBool(
          *signature != *refresh_cache_.last_signature, comm)) {
    return unavailable(
        "current_candidate_producer_linkage_span_signature_mismatch");
  }
  bool context_matches_cache = false;
  std::exception_ptr local_context_failure;
  try {
    context_matches_cache = activeCutContextMatchesRefreshCache(
        *sim_.fe_system,
        refresh_cache_,
        comm,
        "current_candidate_producer_linkage_context_preparation");
  } catch (...) {
    local_context_failure = std::current_exception();
  }
  if (globalAnyBool(local_context_failure != nullptr, comm)) {
    return unavailable(
        "current_candidate_producer_linkage_context_preparation_failed");
  }
  if (!context_matches_cache) {
    return unavailable(
        "current_candidate_producer_linkage_context_cache_mismatch");
  }
  if (globalAnyBool(
          refresh_report.request_policy_key == 0u ||
              refresh_report.request_policy_key !=
                  signature->request_policy_key,
          comm)) {
    return unavailable(
        "current_candidate_producer_linkage_report_policy_mismatch");
  }
  if (globalAnyBool(
          !refresh_cache_.topology_key.has_value() ||
              refresh_report.topology_key == 0u ||
              refresh_report.topology_key !=
                  *refresh_cache_.topology_key,
          comm)) {
    return unavailable(
        "current_candidate_producer_linkage_report_topology_mismatch");
  }
  if (globalAnyBool(
          refresh_report.evaluated_state_source_revisions.empty() ||
              refresh_report.evaluated_state_source_revisions !=
                  refresh_cache_.evaluated_state_source_revisions,
          comm)) {
    return unavailable(
        "current_candidate_producer_linkage_report_source_mapping_mismatch");
  }

  bool local_marker_mapping_matches = true;
  std::exception_ptr local_marker_failure;
  try {
    const auto* context = sim_.fe_system->cutIntegrationContext();
    local_marker_mapping_matches = context != nullptr;
    for (const auto& request : requests) {
      const auto marker =
          resolvedActiveCutVolumeInterfaceMarker(
              *sim_.fe_system, request);
      if (!marker.has_value()) {
        local_marker_mapping_matches = false;
        continue;
      }
      const auto source =
          refresh_report.evaluated_state_source_revisions.find(*marker);
      if (source ==
              refresh_report.evaluated_state_source_revisions.end() ||
          source->second == 0u || context == nullptr ||
          !context->hasFreeSurfaceGeometrySnapshotForMarker(*marker) ||
          !context->hasExpectedGeneratedSourceValueRevision(*marker) ||
          context->expectedGeneratedSourceValueRevision(*marker) !=
              source->second) {
        local_marker_mapping_matches = false;
      }
    }
  } catch (...) {
    local_marker_failure = std::current_exception();
  }
  if (globalAnyBool(local_marker_failure != nullptr, comm)) {
    return unavailable(
        "current_candidate_producer_linkage_marker_mapping_preparation_failed");
  }
  if (globalAnyBool(!local_marker_mapping_matches, comm)) {
    return unavailable(
        "current_candidate_producer_linkage_marker_source_mapping_mismatch");
  }
  return CurrentCandidateProducerLinkageResult{
      .success = true,
      .diagnostic = "current_candidate_producer_linkage_available",
  };
}

void LevelSetGeometryTransaction::commit()
{
  if (!active_ || !system_transaction_active_ ||
      !lifecycle_transaction_active_) {
    throw std::logic_error(
        "level-set maintenance geometry transaction is not active");
  }
  publication_started_ = true;
  sim_.fe_system->commitCutIntegrationContextTransaction();
  system_transaction_active_ = false;
  lifecycle_.commitTransaction();
  lifecycle_transaction_active_ = false;
  active_ = false;
}

void LevelSetGeometryTransaction::rollback()
{
  if (publication_started_) {
    throw std::logic_error(
        "level-set maintenance geometry transaction cannot claim "
        "rollback after geometry publication has started");
  }
  if (!active_) {
    return;
  }
  std::exception_ptr first_failure;
  bool mesh_fields_restored = false;
  bool prescribed_fields_restored = false;
  bool refresh_cache_restored = false;
  const auto attempt = [&](auto&& action) {
    try {
      action();
    } catch (...) {
      if (!first_failure) {
        first_failure = std::current_exception();
      }
    }
  };
  if (system_transaction_active_) {
    attempt([&] {
      sim_.fe_system->rollbackCutIntegrationContextTransaction();
      system_transaction_active_ = false;
    });
  }
  if (lifecycle_transaction_active_) {
    attempt([&] {
      lifecycle_.rollbackTransaction();
      lifecycle_transaction_active_ = false;
    });
  }
  // Transactional context callbacks can update mesh and prescribed fields
  // while restoring their own state. Restore those dependent checkpoints
  // after the context and lifecycle on every attempt, including a retry
  // after one of the upstream rollback operations failed.
  attempt([&] {
    restoreActiveLevelSetMeshFields(sim_, mesh_field_checkpoint_);
    mesh_fields_restored = true;
  });
  attempt([&] {
    restorePrescribedFieldCoefficients(
        *sim_.fe_system, prescribed_field_checkpoint_);
    prescribed_fields_restored = true;
  });
  attempt([&] {
    refresh_cache_ = refresh_cache_backup_;
    refresh_cache_restored = true;
  });
  active_ =
      system_transaction_active_ ||
      lifecycle_transaction_active_ ||
      !mesh_fields_restored ||
      !prescribed_fields_restored ||
      !refresh_cache_restored;
  if (first_failure) {
    std::rethrow_exception(first_failure);
  }
}

// ---------------------------------------------------------------------------
// LevelSetGeometryService
// ---------------------------------------------------------------------------

LevelSetGeometryService::LevelSetGeometryService(
    LevelSetGeometryHost host,
    std::vector<ActiveCutVolumeRequest> requests)
    : host_(host),
      requests_(std::move(requests))
{
}

void LevelSetGeometryService::trackTopology()
{
  refresh_cache_.observer =
      [this](const ActiveCutContextRefreshReport& report,
             std::string_view provenance) {
        topology_tracker_.observe(report, provenance);
      };
}

ActiveCutContextRefreshReport LevelSetGeometryService::refreshCached(
    svmp::FE::backends::GenericVector& solution,
    const char* provenance)
{
  return refreshActiveCutIntegrationContextCached(
      host_, requests_, solution, lifecycle_, refresh_cache_, provenance);
}

ActiveCutContextRefreshReport LevelSetGeometryService::refreshCached(
    const svmp::FE::systems::SystemStateView& state,
    const char* provenance)
{
  return refreshActiveCutIntegrationContextCached(
      host_, requests_, state, lifecycle_, refresh_cache_, provenance);
}

std::optional<ActiveCutContextRefreshReport>
LevelSetGeometryService::reinstallForRestoredState(
    const svmp::FE::systems::SystemStateView& state,
    const char* provenance)
{
  return reinstallRetainedActiveCutContextForRestoredState(
      host_, requests_, state, refresh_cache_, provenance);
}

void LevelSetGeometryService::retainAtAttemptEntry(
    bool outer_fixed_point_state,
    bool projected_outer_fixed_point_state)
{
  retainActiveCutContextAtAttemptEntry(
      host_,
      refresh_cache_,
      outer_fixed_point_state,
      projected_outer_fixed_point_state);
}

std::unique_ptr<LevelSetGeometryTransaction>
LevelSetGeometryService::beginTransaction(
    std::span<const svmp::FE::FieldId> prescribed_fields)
{
  return std::make_unique<LevelSetGeometryTransaction>(
      host_, lifecycle_, refresh_cache_, requests_, prescribed_fields);
}

// ---------------------------------------------------------------------------
// LevelSetGeometryParticipant
// ---------------------------------------------------------------------------
//
// The bodies are the cut-context part of the generated-state synchronization
// and the cut-topology epoch hooks of ApplicationDriver's runSteadyState and
// runTransient, moved with the statements and log text unchanged.

LevelSetGeometryParticipant::LevelSetGeometryParticipant(
    std::shared_ptr<LevelSetGeometryService> service,
    Options options)
    : service_(std::move(service)), options_(options)
{
  if (!service_) {
    throw std::invalid_argument(
        "LevelSetGeometryParticipant: null geometry service");
  }
}

void LevelSetGeometryParticipant::prepare(
    const svmp::FE::timestepping::GeneratedStateSync& sync)
{
  using StateSyncPoint =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  auto& cache = service_->refreshCache();
  if (options_.solve == svmp::FE::timestepping::GeneratedStateSolve::Steady) {
    if (sync.point == StateSyncPoint::RestoredOuterFixedPointState) {
      cache.last_signature.reset();
      cache.last_vector_signature.reset();
    }
    return;
  }
  if (sync.restoring_generated_state) {
    // A restore changes the source vector. Value-revision counters are
    // per vector, so only a content signature may prove that the
    // installed (or the retained entry) geometry is still current.
    cache.last_vector_signature.reset();
    cache.closeReinstallWindow();
  }
  if (sync.point == StateSyncPoint::EndpointCandidateState &&
      options_.reinstall_after_contact_stage_endpoint) {
    // The final candidate gate reconstructs the dynamic-contact stage:
    // it refreshes the stage geometry, which the last outer pass
    // generated from the same level set, and then the endpoint again.
    // Keep the context replaced by the endpoint rebuild so that both
    // refreshes can reinstall geometry instead of rebuilding it.
    cache.reinstall_candidate.reset();
    cache.reinstall_window_open = true;
  }
}

void LevelSetGeometryParticipant::synchronize(
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::timestepping::GeneratedStateSync& sync)
{
  using StateSyncPoint =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  const auto point = sync.point;
  const char* const point_name =
      svmp::FE::timestepping::generatedStateSyncPointName(point);
  if (options_.solve == svmp::FE::timestepping::GeneratedStateSolve::Steady) {
    const auto report = service_->refreshCached(state, point_name);
    logCutTopologyChange(
        report, point, service_->lastLoggedTopologyKey(), "steady");
    sync.evaluated_state_source_revisions =
        report.evaluated_state_source_revisions;
    return;
  }
  std::optional<ActiveCutContextRefreshReport> restored_report;
  if (sync.restoring_generated_state) {
    restored_report = service_->reinstallForRestoredState(state, point_name);
  }
  const auto report =
      restored_report.has_value()
          ? *restored_report
          : service_->refreshCached(state, point_name);
  service_->retainAtAttemptEntry(
      point == StateSyncPoint::OuterFixedPointState,
      point == StateSyncPoint::ProjectedOuterFixedPointState);
  if (report.refreshed &&
      report.request_policy_key != options_.expected_request_policy_key) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Active cut request policy changed during transient Newton synchronization.");
  }
  logCutTopologyChange(
      report, point, service_->lastLoggedTopologyKey(), "transient");
  sync.evaluated_state_source_revisions =
      report.evaluated_state_source_revisions;
}

void LevelSetGeometryParticipant::invalidate()
{
  service_->refreshCache().invalidateGeneratedState();
}

bool LevelSetGeometryParticipant::providesEpochs() const noexcept
{
  return options_.solve ==
             svmp::FE::timestepping::GeneratedStateSolve::Transient &&
         options_.track_topology;
}

bool LevelSetGeometryParticipant::epochDiscontinuity(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
    const
{
  using TransientStateSyncPoint =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  const auto* cut_topology_attempt_tracker = &service_->topologyTracker();
  return point ==
             TransientStateSyncPoint::OuterFixedPointState &&
         cut_topology_attempt_tracker->attemptActive() &&
         cut_topology_attempt_tracker->attemptTainted();
}

void LevelSetGeometryParticipant::acknowledgeEpochDiscontinuity(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
{
  using TransientStateSyncPoint =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  auto* cut_topology_attempt_tracker = &service_->topologyTracker();
  if (point !=
      TransientStateSyncPoint::OuterFixedPointState) {
    throw std::logic_error(
        "[svMultiPhysics::Application] A transient cut-topology restart was acknowledged outside an outer fixed-point refresh.");
  }
  const auto previous_attempt_key =
      cut_topology_attempt_tracker->attemptTopologyKey()
          .value_or(0u);
  cut_topology_attempt_tracker->acknowledgeNonlinearRestart();
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "transient cut-topology nonlinear restart")
          .diagnostic("cut_topology_nonlinear_restart")
          .value("committed_topology_key", cut_topology_attempt_tracker->acceptedTopologyKey().value_or(0u))
          .value("previous_attempt_topology_key", previous_attempt_key)
          .value("restarted_attempt_topology_key", cut_topology_attempt_tracker->attemptTopologyKey().value_or(0u))
          .value("first_mismatched_topology_key", cut_topology_attempt_tracker->firstMismatchedTopologyKey().value_or(0u))
          .text("action", "continue_on_refreshed_epoch")
          .rootOnly());
}

bool LevelSetGeometryParticipant::epochRevisited(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
    const
{
  using TransientStateSyncPoint =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  const auto* cut_topology_attempt_tracker = &service_->topologyTracker();
  return point ==
             TransientStateSyncPoint::OuterFixedPointState &&
         cut_topology_attempt_tracker->revisitPending();
}

void LevelSetGeometryParticipant::acknowledgeFrozenEpoch()
{
  auto* cut_topology_attempt_tracker = &service_->topologyTracker();
  cut_topology_attempt_tracker->acknowledgeFrozenEpoch();
  svmp::FE::logRecord(
      svmp::FE::LogRecord("LevelSetGeometryService", "transient cut-topology cycle")
          .diagnostic("cut_topology_cycle")
          .value("committed_topology_key", cut_topology_attempt_tracker->acceptedTopologyKey().value_or(0u))
          .value("frozen_attempt_topology_key", cut_topology_attempt_tracker->attemptTopologyKey().value_or(0u))
          .text("action", "accept_on_frozen_epoch")
          .rootOnly());
}

// ---------------------------------------------------------------------------
// LevelSetGeometryStepParticipant
// ---------------------------------------------------------------------------
//
// The statements are the cut-geometry statements of the TimeLoop callbacks
// of ApplicationDriver's runTransient, unchanged and in their order.  The
// retention, attempt-entry refresh and baseline check were the first
// statements of the before-physics-solve callback and the release the first
// of the accepted callback; the attempt completion was the last statement of
// the commit callback.  The attempt begin (step start) and discard
// (candidate discarded) move ahead of caller statements that neither refresh
// the cut geometry nor read the tracker, and the discard stays ahead of the
// caller's rollback refresh.

LevelSetGeometryStepParticipant::LevelSetGeometryStepParticipant(
    std::shared_ptr<LevelSetGeometryService> service,
    Part part,
    Options options)
    : service_(std::move(service)), part_(part), options_(options)
{
  if (!service_) {
    throw std::invalid_argument(
        "LevelSetGeometryStepParticipant: null geometry service");
  }
}

std::shared_ptr<LevelSetGeometryStepParticipant>
LevelSetGeometryStepParticipant::addTo(
    svmp::FE::timestepping::StepParticipantList& list,
    std::shared_ptr<LevelSetGeometryService> service,
    Options options)
{
  auto entry = std::make_shared<LevelSetGeometryStepParticipant>(
      service, Part::Entry, options);
  auto exit = std::make_shared<LevelSetGeometryStepParticipant>(
      std::move(service), Part::Exit, options);
  list.add(entry, svmp::FE::timestepping::StepParticipantOrder::BeforeBase);
  list.add(std::move(exit),
           svmp::FE::timestepping::StepParticipantOrder::AfterBase);
  return entry;
}

svmp::FE::timestepping::StepHook LevelSetGeometryStepParticipant::hooks()
    const noexcept
{
  using svmp::FE::timestepping::StepHook;
  if (part_ == Part::Entry) {
    return (options_.track_topology ? StepHook::StepStart : StepHook::None) |
           StepHook::BeforePhysicsSolve | StepHook::Accepted;
  }
  return StepHook::CandidateDiscarded |
         (options_.track_topology ? StepHook::CommitReady : StepHook::None);
}

void LevelSetGeometryStepParticipant::onStepStart(
    const svmp::FE::timestepping::TimeHistory& /*history*/)
{
  if (options_.track_topology) {
    service_->topologyTracker().beginAttempt();
  }
}

bool LevelSetGeometryStepParticipant::onBeforePhysicsSolve(
    svmp::FE::timestepping::TimeHistory& history,
    double /*solve_time*/,
    double /*dt*/)
{
  armActiveCutRestoreRetention(service_->refreshCache());
  attempt_entry_report_ =
      service_->refreshCached(history.u(), "before_physics_solve");
  if (options_.track_topology) {
    service_->topologyTracker().requireAcceptedBaseline();
  }
  return true;
}

void LevelSetGeometryStepParticipant::onStepCandidateDiscarded(
    svmp::FE::timestepping::TimeHistory& /*history*/)
{
  service_->refreshCache().closeReinstallWindow();
  if (options_.track_topology) {
    service_->topologyTracker().discardAttempt();
  }
}

void LevelSetGeometryStepParticipant::onStepCommitReady(
    svmp::FE::timestepping::TimeHistory& /*history*/)
{
  if (options_.track_topology) {
    service_->topologyTracker().completeAttempt(
        service_->refreshCache().topology_key);
  }
}

void LevelSetGeometryStepParticipant::onStepAccepted(
    svmp::FE::timestepping::TimeHistory& /*history*/)
{
  releaseActiveCutRestoreRetention(service_->refreshCache());
}

} // namespace level_set
} // namespace FE
} // namespace svmp

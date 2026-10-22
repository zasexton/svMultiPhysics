/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetConservativePhaseWorkflow.h"

#include "Assembly/Assembler.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Core/RankLog.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Interfaces/MaterialInterfaceTransportVelocity.h"
#include "LevelSet/LevelSetConservativePhaseArtifact.h"
#include "LevelSet/LevelSetConservativePhaseState.h"
#include "Systems/SystemState.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace level_set {

namespace {

// The moved bodies keep the names of the driver's communicator helpers.
// Each name forwards to the FE/Systems collective that the driver's helper
// of the same name calls, so every rank issues the same MPI calls in the
// same order as before the move.
using svmp::FE::systems::requireCollectivePhasePreparation;
using svmp::FE::systems::scatterFeOrderedSolution;

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

bool globalAnyBool(
    bool local, const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveAnyTrue(local, comm);
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

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::captureFeOrderedVectorCollectively(
      solution, comm);
}

} // namespace

// Moved from Application/Core/ApplicationDriver.cpp (lines 23041-23768 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.

namespace {

svmp::FE::geometry::CutIntegrationSide conservativePhaseSide(
    const svmp::FE::level_set::LevelSetConservativePhaseOptions& options)
{
  return options.liquid_side ==
                 svmp::FE::level_set::LevelSetPhaseSide::Negative
             ? svmp::FE::geometry::CutIntegrationSide::Negative
             : svmp::FE::geometry::CutIntegrationSide::Positive;
}

void requireConservativePhaseGeometryBinding(
    const svmp::FE::systems::FESystem& system,
    const ConservativePhaseRequest& request)
{
  if (!request.configuration->volume_cut_request.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport requires one matching authoritative active cut-volume request for level-set field '" +
        request.configuration->transport.level_set.field_name + "'.");
  }
  const auto requested_side = conservativePhaseSide(
      request.configuration->transport.conservative_phase);
  const auto active_side = cutIntegrationSide(
      request.configuration->volume_cut_request->active_side);
  if (requested_side != active_side) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase liquid side disagrees with the active cut-volume side for level-set field '" +
        request.configuration->transport.level_set.field_name + "'.");
  }
  const auto marker = generatedCutContextMarkerForMaintenance(
      system, request.configuration->volume_cut_request);
  if (!marker.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport could not resolve an authoritative generated interface marker for level-set field '" +
        request.configuration->transport.level_set.field_name + "'.");
  }
}

} // namespace

svmp::FE::level_set::LevelSetP1PhaseTransportGraph&
requireCurrentConservativePhaseGraph(
    svmp::FE::systems::FESystem& system,
    ConservativePhaseRequest& request,
    bool allow_graph_rebuild)
{
  const auto phase_field = system.findFieldByName(
      request.configuration->transport.conservative_phase.liquid_indicator.field_name);
  if (phase_field == svmp::FE::INVALID_FIELD_ID) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport could not find field '" +
        request.configuration->transport.conservative_phase.liquid_indicator.field_name + "'.");
  }
  const auto& mesh = system.meshAccess();
  const auto& dofs = system.fieldDofHandler(phase_field);
  const auto graph_is_current = [&] {
    if (!request.runtime.conservative_phase_graph.has_value()) {
      return false;
    }
    const auto& graph = *request.runtime.conservative_phase_graph;
    return graph.success &&
           graph.geometry_revision == mesh.geometryRevision() &&
           graph.topology_revision == mesh.topologyRevision() &&
           graph.ownership_revision == mesh.ownershipRevision() &&
           graph.numbering_revision == mesh.numberingRevision() &&
           graph.dof_layout_revision == dofs.dofLayoutRevision() &&
           graph.nodes == static_cast<std::size_t>(dofs.getNumDofs());
  };
  const bool local_graph_is_current = graph_is_current();
  const auto comm = activeFESystemCommunicator(system);
  const bool graph_rebuild_required =
      globalAnyBool(!local_graph_is_current, comm);
  if (graph_rebuild_required) {
    if (!allow_graph_rebuild) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase split transport requires its accepted q^n graph and fixed background mesh to remain unchanged; rebuilding a stale graph during candidate maintenance is forbidden.");
    }
    svmp::FE::level_set::LevelSetP1PhaseGraphOptions graph_options;
    graph_options.invariant_tolerance =
        request.configuration->transport.conservative_phase.invariant_tolerance;
    auto graph =
        svmp::FE::level_set::buildLevelSetP1PhaseTransportGraph(
            system, phase_field, graph_options);
    if (!graph.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase graph assembly failed for field '" +
          request.configuration->transport.conservative_phase.liquid_indicator.field_name +
          "': " + graph.diagnostic);
    }
    request.runtime.conservative_phase_graph = std::move(graph);
  }
  return *request.runtime.conservative_phase_graph;
}

svmp::FE::level_set::LevelSetP1PhaseProjectionOptions
conservativePhaseProjectionOptions(
    const svmp::FE::systems::FESystem& system,
    const ConservativePhaseRequest& request)
{
  const auto marker = generatedCutContextMarkerForMaintenance(
      system, request.configuration->volume_cut_request);
  if (!marker.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase projection could not resolve its generated interface marker.");
  }
  svmp::FE::level_set::LevelSetP1PhaseProjectionOptions options;
  options.interface_marker = *marker;
  options.liquid_side = conservativePhaseSide(
      request.configuration->transport.conservative_phase);
  options.invariant_tolerance =
      request.configuration->transport.conservative_phase.invariant_tolerance;
  return options;
}

svmp::FE::level_set::LevelSetP1PhaseProjectionResult
projectCurrentConservativePhaseGeometry(
    svmp::FE::systems::FESystem& system,
    ConservativePhaseRequest& request,
    bool allow_graph_rebuild)
{
  requireConservativePhaseGeometryBinding(system, request);
  auto& graph = requireCurrentConservativePhaseGraph(
      system, request, allow_graph_rebuild);
  const auto phase_field = system.findFieldByName(
      request.configuration->transport.conservative_phase.liquid_indicator.field_name);
  const auto projection_options = conservativePhaseProjectionOptions(
      system, request);
  return svmp::FE::level_set::
      projectLevelSetP1PhaseIndicatorFromCutContext(
          system, phase_field, graph, projection_options);
}

namespace {

void assignConservativePhaseSlice(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId phase_field,
    std::span<const svmp::FE::Real> phase_values,
    std::vector<svmp::FE::Real>& solution)
{
  const auto offset = static_cast<std::size_t>(
      system.fieldDofOffset(phase_field));
  const auto count = static_cast<std::size_t>(
      system.fieldDofHandler(phase_field).getNumDofs());
  if (phase_values.size() != count || offset + count > solution.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase field slice does not match the FE solution layout.");
  }
  std::copy(phase_values.begin(), phase_values.end(),
            solution.begin() + static_cast<std::ptrdiff_t>(offset));
}

void zeroConservativePhaseRateSlice(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId phase_field,
    svmp::FE::backends::GenericVector& rate)
{
  auto values = gatherFeOrderedSolution(
      rate, activeFESystemCommunicator(system));
  const auto offset = static_cast<std::size_t>(
      system.fieldDofOffset(phase_field));
  const auto count = static_cast<std::size_t>(
      system.fieldDofHandler(phase_field).getNumDofs());
  if (offset + count > values.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase rate slice exceeds the FE vector layout.");
  }
  std::fill(values.begin() + static_cast<std::ptrdiff_t>(offset),
            values.begin() + static_cast<std::ptrdiff_t>(offset + count),
            svmp::FE::Real{0.0});
  scatterFeOrderedSolution(rate, values);
}

} // namespace

void initializeConservativePhaseStates(
    const ConservativePhaseWorkflowHost& host,
    std::span<ConservativePhaseRequest> requests)
{
  if (!host.fe_system || !host.time_history) {
    return;
  }
  auto& system = *host.fe_system;
  auto& history = *host.time_history;
  const auto comm = activeFESystemCommunicator(system);
  for (auto& request : requests) {
    if (!request.configuration->transport.conservative_phase.enabled ||
        request.runtime.conservative_phase_initialized) {
      continue;
    }
    const auto projection = projectCurrentConservativePhaseGeometry(
        system, request);
    if (!projection.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Initial conservative phase projection failed for field '" +
          request.configuration->transport.conservative_phase.liquid_indicator.field_name +
          "': " + projection.diagnostic);
    }
    const auto phase_field = system.findFieldByName(
        request.configuration->transport.conservative_phase.liquid_indicator.field_name);
    auto current = gatherFeOrderedSolution(history.u(), comm);
    assignConservativePhaseSlice(
        system, phase_field, projection.liquid_indicator, current);
    std::vector<std::vector<svmp::FE::Real>> initialized_history;
    initialized_history.reserve(
        static_cast<std::size_t>(history.historyDepth()));
    for (int k = 1; k <= history.historyDepth(); ++k) {
      auto previous =
          gatherFeOrderedSolution(history.uPrevK(k), comm);
      assignConservativePhaseSlice(
          system, phase_field, projection.liquid_indicator, previous);
      initialized_history.push_back(std::move(previous));
    }
    scatterFeOrderedSolution(history.u(), current);
    for (int k = 1; k <= history.historyDepth(); ++k) {
      scatterFeOrderedSolution(
          history.uPrevK(k),
          initialized_history[static_cast<std::size_t>(k - 1)]);
    }
    if (history.hasUDotState()) {
      zeroConservativePhaseRateSlice(system, phase_field, history.uDot());
    }
    if (history.hasUDDotState()) {
      zeroConservativePhaseRateSlice(system, phase_field, history.uDDot());
    }
    history.updateGhosts();
    request.runtime.conservative_phase_initialized = true;
    svmp::FE::rank_log::oopCout()
        << "[svMultiPhysics::Application] Conservative phase initialized"
        << " field='"
        << request.configuration->transport.conservative_phase.liquid_indicator.field_name << "'"
        << " level_set_field='" << request.configuration->transport.level_set.field_name << "'"
        << " liquid_side="
        << (request.configuration->transport.conservative_phase.liquid_side ==
                    svmp::FE::level_set::LevelSetPhaseSide::Negative
                ? "negative"
                : "positive")
        << " retained_measure=" << projection.retained_liquid_measure
        << " projected_measure=" << projection.projected_liquid_measure
        << " cut_context_revision=" << projection.cut_context_revision
        << " source_value_revision=" << projection.source_value_revision
        << " graph_nodes=" << projection.nodes
        << std::endl;
  }
}

namespace {

void allReduceConservativePhaseBuffer(
    std::vector<svmp::FE::Real>& values,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
#if FE_HAS_MPI
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0) {
    MPI_Finalized(&finalized);
  }
  if (comm.size() > 1) {
    if (initialized == 0 || finalized != 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase reduction requires an active MPI runtime.");
    }
    MPI_Datatype datatype = MPI_LONG_DOUBLE;
    if constexpr (std::is_same_v<svmp::FE::Real, float>) {
      datatype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<svmp::FE::Real, double>) {
      datatype = MPI_DOUBLE;
    }
    const auto [minimum_value_count, maximum_value_count] =
        globalMinMaxUint64(
            static_cast<std::uint64_t>(values.size()), comm);
    if (minimum_value_count != maximum_value_count) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase reduction buffer sizes differ across ranks.");
    }
    std::size_t offset = 0u;
    while (offset < values.size()) {
      const auto remaining = values.size() - offset;
      const int count = static_cast<int>(std::min<std::size_t>(
          remaining,
          static_cast<std::size_t>(std::numeric_limits<int>::max())));
      MPI_Allreduce(
          MPI_IN_PLACE,
          values.data() + offset,
          count,
          datatype,
          MPI_SUM,
          comm.native());
      offset += static_cast<std::size_t>(count);
    }
  }
#else
  (void)comm;
#endif
}

} // namespace

std::vector<std::array<svmp::FE::Real, 3>>
sampleConservativePhaseVelocity(
    const ConservativePhaseWorkflowHost& host,
    const ConservativePhaseRequest& request,
    const svmp::FE::level_set::LevelSetP1PhaseTransportGraph& graph,
    const svmp::FE::systems::SystemStateView& state)
{
  std::vector<std::array<svmp::FE::Real, 3>> velocity;
  if (request.configuration->transport.velocity.source ==
      svmp::FE::level_set::LevelSetVelocitySource::ConstantVector) {
    std::exception_ptr local_preparation_failure;
    try {
      velocity.resize(graph.nodes);
      for (const auto value : request.configuration->transport.velocity.constant_value) {
        if (!std::isfinite(value)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Conservative phase constant velocity contains a non-finite component.");
        }
      }
      std::fill(velocity.begin(), velocity.end(),
                request.configuration->transport.velocity.constant_value);
    } catch (...) {
      local_preparation_failure = std::current_exception();
    }
    if (host.fe_system) {
      requireCollectivePhasePreparation(
          local_preparation_failure,
          activeFESystemCommunicator(*host.fe_system),
          "conservative_phase_constant_velocity");
    } else if (local_preparation_failure) {
      std::rethrow_exception(local_preparation_failure);
    }
    return velocity;
  }
  if (!host.fe_system || !host.evaluate_vertex_field) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase velocity sampling requires an FE system and primary mesh.");
  }
  auto& system = *host.fe_system;
  const auto comm = activeFESystemCommunicator(system);
  std::vector<svmp::FE::Real> accumulated;
  std::exception_ptr local_preparation_failure;
  try {
    if (graph.nodes >
        std::numeric_limits<std::size_t>::max() / 4u) {
      throw std::overflow_error(
          "[svMultiPhysics::Application] Conservative phase velocity sampling graph size overflows its accumulation layout.");
    }
    velocity.resize(graph.nodes);
    accumulated.assign(
        graph.nodes * 4u, svmp::FE::Real{0.0});
    const auto phase_field = system.findFieldByName(
        request.configuration->transport.conservative_phase.liquid_indicator.field_name);
    if (phase_field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase velocity sampling could not resolve its phase field.");
    }
    const auto* entity_map =
        system.fieldDofHandler(phase_field).getEntityDofMap();
    if (entity_map == nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase velocity sampling requires a vertex-nodal phase field.");
    }

    const bool material_interface_velocity =
        request.configuration->transport.velocity.source ==
        svmp::FE::level_set::LevelSetVelocitySource::
            MaterialInterfacePhasePair;
    std::vector<double> vertex_velocity;
    std::vector<double> negative_vertex_velocity;
    std::vector<double> positive_vertex_velocity;
    const svmp::FE::interfaces::
        MaterialInterfaceTransportVelocityDeclaration*
        material_declaration = nullptr;
    if (material_interface_velocity) {
      const auto level_set_field = system.findFieldByName(
          request.configuration->transport.level_set.field_name);
      if (level_set_field == svmp::FE::INVALID_FIELD_ID ||
          request.configuration->transport.velocity.material_interface_marker < 0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase material-interface velocity sampling requires a level-set field and interface marker.");
      }
      for (const auto& declaration :
           system.materialInterfaceTransportVelocityDeclarations()) {
        if (declaration.level_set_field != level_set_field ||
            declaration.interface_marker !=
                request.configuration->transport.velocity.material_interface_marker) {
          continue;
        }
        if (material_declaration != nullptr) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Conservative phase material-interface velocity has multiple matching declarations.");
        }
        material_declaration = &declaration;
      }
      if (material_declaration == nullptr ||
          material_declaration->dimension != graph.dimension) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase material-interface velocity declaration is missing or dimensionally incompatible.");
      }
      negative_vertex_velocity = host.evaluate_vertex_field(
          material_declaration->negative_velocity_field,
          state,
          static_cast<std::size_t>(graph.dimension),
          "sampling conservative negative-phase velocity");
      positive_vertex_velocity = host.evaluate_vertex_field(
          material_declaration->positive_velocity_field,
          state,
          static_cast<std::size_t>(graph.dimension),
          "sampling conservative positive-phase velocity");
    } else {
      const auto velocity_field = system.findFieldByName(
          request.configuration->transport.velocity.field_name);
      if (velocity_field == svmp::FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase velocity sampling could not resolve its velocity field.");
      }
      const auto& velocity_record = system.fieldRecord(velocity_field);
      if (velocity_record.components != graph.dimension) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase velocity dimension does not match the phase graph.");
      }
      vertex_velocity = host.evaluate_vertex_field(
          velocity_field,
          state,
          static_cast<std::size_t>(graph.dimension),
          "sampling conservative phase velocity");
    }
    for (std::size_t vertex = 0;
         vertex < host.primary_mesh_vertex_count(); ++vertex) {
      const auto dofs = entity_map->getVertexDofs(
          static_cast<svmp::FE::GlobalIndex>(vertex));
      if (dofs.size() != 1u || dofs.front() < 0 ||
          static_cast<std::size_t>(dofs.front()) >= graph.nodes) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase field does not have exactly one in-range DOF per mesh vertex.");
      }
      const auto node = static_cast<std::size_t>(dofs.front());
      std::array<svmp::FE::Real, 3> sampled_velocity{};
      if (material_interface_velocity) {
        std::array<svmp::FE::Real, 3> negative{};
        std::array<svmp::FE::Real, 3> positive{};
        for (int d = 0; d < graph.dimension; ++d) {
          const auto component = static_cast<std::size_t>(d);
          negative[component] = static_cast<svmp::FE::Real>(
              negative_vertex_velocity[
                  vertex * static_cast<std::size_t>(graph.dimension) +
                  component]);
          positive[component] = static_cast<svmp::FE::Real>(
              positive_vertex_velocity[
                  vertex * static_cast<std::size_t>(graph.dimension) +
                  component]);
        }
        sampled_velocity = svmp::FE::interfaces::
            materialInterfaceCommonTraceVelocity(
                *material_declaration,
                negative,
                positive);
      } else {
        for (int d = 0; d < graph.dimension; ++d) {
          sampled_velocity[static_cast<std::size_t>(d)] =
              static_cast<svmp::FE::Real>(
                  vertex_velocity[
                      vertex * static_cast<std::size_t>(graph.dimension) +
                      static_cast<std::size_t>(d)]);
        }
      }
      for (int d = 0; d < graph.dimension; ++d) {
        accumulated[4u * node + static_cast<std::size_t>(d)] +=
            sampled_velocity[static_cast<std::size_t>(d)];
      }
      accumulated[4u * node + 3u] += svmp::FE::Real{1.0};
    }
  } catch (...) {
    local_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_preparation_failure,
      comm,
      "conservative_phase_velocity_local_sampling");
  allReduceConservativePhaseBuffer(accumulated, comm);

  std::exception_ptr local_validation_failure;
  try {
    for (std::size_t node = 0; node < graph.nodes; ++node) {
      const auto samples = accumulated[4u * node + 3u];
      if (!std::isfinite(samples) || !(samples > svmp::FE::Real{0.0})) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase velocity sampling left a graph node without a value.");
      }
      for (int d = 0; d < graph.dimension; ++d) {
        const auto value =
            accumulated[4u * node + static_cast<std::size_t>(d)] / samples;
        if (!std::isfinite(value)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Conservative phase velocity sampling produced a non-finite value.");
        }
        velocity[node][static_cast<std::size_t>(d)] = value;
      }
    }
  } catch (...) {
    local_validation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_validation_failure,
      comm,
      "conservative_phase_velocity_post_reduction_validation");
  return velocity;
}

namespace {

std::vector<std::array<svmp::FE::Real, 3>>
conservativePhaseNodeCoordinates(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId phase_field,
    const svmp::FE::level_set::LevelSetP1PhaseTransportGraph& graph)
{
  const svmp::FE::dofs::EntityDofMap* entity_map = nullptr;
  const auto& mesh = system.meshAccess();
  const auto comm = activeFESystemCommunicator(system);
  std::vector<svmp::FE::Real> accumulated;
  std::exception_ptr local_coordinate_preparation_failure;
  try {
    entity_map =
        system.fieldDofHandler(phase_field).getEntityDofMap();
    if (entity_map == nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase fixed regions require a vertex-nodal phase field.");
    }
    if (graph.nodes >
        std::numeric_limits<std::size_t>::max() / 4u) {
      throw std::overflow_error(
          "[svMultiPhysics::Application] Conservative phase fixed-region coordinate buffer size overflows.");
    }
    accumulated.assign(
        graph.nodes * 4u, svmp::FE::Real{0.0});
    for (svmp::FE::GlobalIndex vertex = 0;
         vertex < mesh.numVertices(); ++vertex) {
      const auto dofs = entity_map->getVertexDofs(vertex);
      if (dofs.size() != 1u || dofs.front() < 0 ||
          static_cast<std::size_t>(dofs.front()) >= graph.nodes) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase fixed regions require exactly one in-range phase DOF per vertex.");
      }
      const auto coordinate = mesh.getNodeCoordinates(vertex);
      const auto node = static_cast<std::size_t>(dofs.front());
      for (std::size_t dimension = 0u; dimension < 3u; ++dimension) {
        if (!std::isfinite(coordinate[dimension])) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Conservative phase fixed region found a non-finite node coordinate.");
        }
        accumulated[4u * node + dimension] += coordinate[dimension];
      }
      accumulated[4u * node + 3u] += svmp::FE::Real{1.0};
    }
  } catch (...) {
    local_coordinate_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_coordinate_preparation_failure,
      comm,
      "conservative_phase_fixed_region_coordinate_preparation");
  allReduceConservativePhaseBuffer(accumulated, comm);

  std::vector<std::array<svmp::FE::Real, 3>> coordinates;
  std::exception_ptr local_coordinate_validation_failure;
  try {
    coordinates.resize(graph.nodes);
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
      const auto samples = accumulated[4u * node + 3u];
      if (!std::isfinite(samples) || !(samples > svmp::FE::Real{0.0})) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase fixed region left a graph node without coordinates.");
      }
      for (std::size_t dimension = 0u; dimension < 3u; ++dimension) {
        coordinates[node][dimension] =
            accumulated[4u * node + dimension] / samples;
      }
    }
  } catch (...) {
    local_coordinate_validation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_coordinate_validation_failure,
      comm,
      "conservative_phase_fixed_region_coordinate_validation");
  return coordinates;
}

std::pair<std::vector<svmp::FE::Real>, std::vector<svmp::FE::Real>>
conservativePhaseOneRingBounds(
    const svmp::FE::level_set::LevelSetP1PhaseTransportGraph& graph,
    std::span<const svmp::FE::Real> previous)
{
  if (previous.size() != graph.nodes) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase state does not match its graph.");
  }
  std::vector<svmp::FE::Real> lower(previous.begin(), previous.end());
  std::vector<svmp::FE::Real> upper(previous.begin(), previous.end());
  for (const auto& edge : graph.edges) {
    const auto first = static_cast<std::size_t>(edge.first_node);
    const auto second = static_cast<std::size_t>(edge.second_node);
    const auto first_value = previous[first];
    const auto second_value = previous[second];
    lower[first] = std::min(lower[first], second_value);
    lower[second] = std::min(lower[second], first_value);
    upper[first] = std::max(upper[first], second_value);
    upper[second] = std::max(upper[second], first_value);
  }
  for (std::size_t node = 0; node < graph.nodes; ++node) {
    lower[node] = std::clamp(lower[node], svmp::FE::Real{0.0},
                             svmp::FE::Real{1.0});
    upper[node] = std::clamp(upper[node], svmp::FE::Real{0.0},
                             svmp::FE::Real{1.0});
  }
  return {std::move(lower), std::move(upper)};
}

} // namespace

std::vector<std::uint8_t> conservativePhaseContactProtectedNodes(
    const svmp::FE::systems::FESystem& system,
    const ConservativePhaseRequest& request,
    const svmp::FE::level_set::LevelSetP1PhaseTransportGraph& graph,
    std::span<const svmp::FE::level_set::LevelSetWallContactConstraint>
        constraints)
{
  const auto comm = activeFESystemCommunicator(system);
  std::vector<std::uint8_t> protected_nodes;
  std::exception_ptr local_mask_storage_failure;
  try {
    protected_nodes.assign(graph.nodes, 0u);
  } catch (...) {
    local_mask_storage_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_mask_storage_failure,
      comm,
      "conservative_phase_contact_mask_storage");
  if (globalSumSize(constraints.size(), comm) == 0u) {
    return protected_nodes;
  }
  svmp::FE::FieldId phase_field = svmp::FE::INVALID_FIELD_ID;
  const svmp::FE::dofs::DofHandler* phase_dofs = nullptr;
  std::exception_ptr local_phase_layout_failure;
  try {
    phase_field = system.findFieldByName(
        request.configuration->transport.conservative_phase.liquid_indicator.field_name);
    if (phase_field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase contact protection could not resolve the phase field.");
    }
    phase_dofs = &system.fieldDofHandler(phase_field);
  } catch (...) {
    local_phase_layout_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_phase_layout_failure,
      comm,
      "conservative_phase_contact_phase_layout");
  const auto& mesh = system.meshAccess();
  const bool local_identity_available =
      mesh.parallelSize() == 1 || mesh.globalEntityIdsAvailable();
  if (globalMinDouble(local_identity_available ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase contact protection requires globally unique parent-cell identities in a distributed run.");
  }

  std::set<svmp::FE::GlobalIndex> local_parent_ids;
  bool local_constraints_valid = true;
  std::vector<svmp::FE::Real> global_mask;
  std::set<svmp::FE::GlobalIndex> matched_parent_ids;
  try {
    global_mask.assign(graph.nodes, svmp::FE::Real{0.0});
    for (const auto& constraint : constraints) {
      local_constraints_valid =
          constraint.parent_cell_global_id !=
              svmp::FE::INVALID_GLOBAL_INDEX &&
          local_constraints_valid;
      if (constraint.parent_cell_global_id !=
          svmp::FE::INVALID_GLOBAL_INDEX) {
        local_parent_ids.insert(constraint.parent_cell_global_id);
      }
    }
    mesh.forEachOwnedCell([&](svmp::FE::GlobalIndex cell) {
      const auto global_cell = mesh.globalEntityIdsAvailable()
          ? mesh.getCellGlobalId(cell)
          : cell;
      if (!local_parent_ids.contains(global_cell)) {
        return;
      }
      const auto cell_dofs = phase_dofs->getCellDofs(cell);
      if (cell_dofs.empty()) {
        throw std::runtime_error(
            "a protected contact parent has no phase DOFs");
      }
      for (const auto dof : cell_dofs) {
        if (dof < 0 || static_cast<std::size_t>(dof) >= graph.nodes) {
          throw std::runtime_error(
              "a protected contact parent has an out-of-range phase DOF");
        }
        global_mask[static_cast<std::size_t>(dof)] =
            svmp::FE::Real{1.0};
      }
      matched_parent_ids.insert(global_cell);
    });
    local_constraints_valid =
        matched_parent_ids == local_parent_ids && local_constraints_valid;
  } catch (...) {
    local_constraints_valid = false;
  }
  if (globalMinDouble(local_constraints_valid ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase contact protection could not map every accepted contact parent to its phase DOFs.");
  }
  allReduceConservativePhaseBuffer(global_mask, comm);
  std::exception_ptr local_mask_validation_failure;
  try {
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
      if (!std::isfinite(global_mask[node]) ||
          global_mask[node] < svmp::FE::Real{0.0}) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase contact protection produced an invalid distributed node mask.");
      }
      protected_nodes[node] =
          global_mask[node] > svmp::FE::Real{0.0} ? 1u : 0u;
    }
  } catch (...) {
    local_mask_validation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_mask_validation_failure,
      comm,
      "conservative_phase_contact_mask_validation");
  return protected_nodes;
}

// Moved from Application/Core/ApplicationDriver.cpp (lines 24194-25603 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.

namespace {

void appendConservativePhaseReplicatedGraphIdentityWords(
    std::vector<std::uint64_t>& words,
    const svmp::FE::level_set::LevelSetP1PhaseGraphIdentity& identity)
{
  appendMaintenanceScheduleSigned(words, identity.dimension);
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(identity.nodes));
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(identity.edges));
  // Mesh event counters are rank-local cache stamps.  The fixed-background
  // guard compares them locally against the accepted q^n graph; only the
  // replicated layout and graph-content identities enter rank consensus.
  appendMaintenanceScheduleWord(words, identity.dof_layout_revision);
  appendMaintenanceScheduleWord(words, identity.content_revision);
}

void appendConservativePhaseCandidateStageWords(
    std::vector<std::uint64_t>& words,
    const ConservativePhaseCandidateStageSnapshot& stage)
{
  appendMaintenanceScheduleWord(words, 0x4350535441474531ull);
  appendMaintenanceScheduleEnum(words, stage.scheme);
  appendMaintenanceScheduleSigned(words, stage.temporal_order);
  appendMaintenanceScheduleWord(words, stage.prospective_step);
  appendMaintenanceScheduleWord(words, stage.attempt);
  appendMaintenanceScheduleReal(words, stage.step_start_time);
  appendMaintenanceScheduleReal(words, stage.step_end_time);
  appendMaintenanceScheduleReal(words, stage.q_input_time);
  appendMaintenanceScheduleReal(words, stage.velocity_state_time);
  appendMaintenanceScheduleReal(words, stage.time_step);
  appendMaintenanceScheduleWord(words, stage.operator_state_revision);
  appendMaintenanceScheduleWord(
      words,
      static_cast<std::uint64_t>(stage.request_schedule_words.size()));
  for (const auto word : stage.request_schedule_words) {
    appendMaintenanceScheduleWord(words, word);
  }
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(stage.requests.size()));
  for (const auto& request : stage.requests) {
    appendMaintenanceScheduleBool(words, request.enabled);
    appendMaintenanceScheduleString(words, request.phase_field_name);
    appendMaintenanceScheduleString(words, request.velocity_field_name);
    appendMaintenanceScheduleEnum(words, request.velocity_source);
    appendMaintenanceScheduleSigned(
        words, request.material_interface_marker);
    if (!request.enabled) {
      continue;
    }
    appendConservativePhaseReplicatedGraphIdentityWords(
        words, request.graph_identity);
    appendMaintenanceScheduleWord(
        words,
        static_cast<std::uint64_t>(
            request.sampled_nodal_velocity.size()));
    appendMaintenanceScheduleWord(
        words,
        svmp::FE::level_set::levelSetP1PhaseVelocityContentRevision(
            request.sampled_nodal_velocity));
  }
}

} // namespace

bool collectiveConservativePhaseVelocityBitsAgree(
    std::span<const std::array<svmp::FE::Real, 3>> velocity,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
#if FE_HAS_MPI
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0) {
    MPI_Finalized(&finalized);
  }
  if (comm.size() > 1) {
    if (initialized == 0 || finalized != 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase velocity comparison requires an active MPI runtime.");
    }
    static_assert(
        sizeof(svmp::FE::Real) == sizeof(std::uint64_t),
        "Conservative phase exact velocity comparison expects binary64 FE scalars");
    constexpr std::size_t chunk_nodes = 128u;
    constexpr std::size_t chunk_words = 3u * chunk_nodes;
    std::array<std::uint64_t, chunk_words> local_words{};
    std::array<std::uint64_t, chunk_words> minimum_words{};
    std::array<std::uint64_t, chunk_words> maximum_words{};
    const auto [minimum_nodes, maximum_nodes] = globalMinMaxUint64(
        static_cast<std::uint64_t>(velocity.size()), comm);
    const bool sizes_agree = minimum_nodes == maximum_nodes;
    const auto collective_nodes =
        static_cast<std::size_t>(maximum_nodes);
    bool values_agree = true;
#ifdef MPI_UINT64_T
    const MPI_Datatype datatype = MPI_UINT64_T;
#else
    const MPI_Datatype datatype = MPI_UNSIGNED_LONG_LONG;
#endif
    for (std::size_t offset = 0u; offset < collective_nodes;) {
      const auto nodes =
          std::min(chunk_nodes, collective_nodes - offset);
      const auto words = nodes * 3u;
      std::fill_n(local_words.begin(), words, 0u);
      const auto local_nodes = offset < velocity.size()
          ? std::min(nodes, velocity.size() - offset)
          : 0u;
      for (std::size_t node = 0u; node < local_nodes; ++node) {
        for (std::size_t component = 0u; component < 3u; ++component) {
          local_words[3u * node + component] =
              std::bit_cast<std::uint64_t>(
                  velocity[offset + node][component]);
        }
      }
      const auto count = static_cast<int>(words);
      MPI_Allreduce(
          local_words.data(), minimum_words.data(), count, datatype,
          MPI_MIN, comm.native());
      MPI_Allreduce(
          local_words.data(), maximum_words.data(), count, datatype,
          MPI_MAX, comm.native());
      values_agree =
          std::equal(
              minimum_words.begin(),
              minimum_words.begin() +
                  static_cast<std::ptrdiff_t>(words),
              maximum_words.begin()) &&
          values_agree;
      offset += nodes;
    }
    return sizes_agree && values_agree;
  }
#else
  (void)velocity;
  (void)comm;
#endif
  return true;
}

bool collectiveConservativePhaseCandidateStageSnapshotAgrees(
    const ConservativePhaseCandidateStageSnapshot& stage,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  std::vector<std::uint64_t> words;
  std::exception_ptr local_word_failure;
  try {
    appendConservativePhaseCandidateStageWords(words, stage);
  } catch (...) {
    local_word_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_word_failure,
      comm,
      "conservative_phase_candidate_stage_consensus_words");
  if (!collectiveLevelSetMaintenanceCanonicalWordsAgree(
          words, comm)) {
    return false;
  }
  for (const auto& request : stage.requests) {
    if (request.enabled &&
        !collectiveConservativePhaseVelocityBitsAgree(
            request.sampled_nodal_velocity, comm)) {
      return false;
    }
  }
  return true;
}

void appendConservativePhaseSplitStageProvenanceWords(
    std::vector<std::uint64_t>& words,
    const std::optional<
        svmp::FE::level_set::LevelSetP1PhaseSplitStageProvenance>&
        provenance)
{
  appendMaintenanceScheduleBool(words, provenance.has_value());
  if (!provenance.has_value()) {
    return;
  }
  appendMaintenanceScheduleEnum(words, provenance->scheme);
  appendMaintenanceScheduleEnum(words, provenance->transport_mesh_policy);
  appendMaintenanceScheduleSigned(words, provenance->temporal_order);
  appendMaintenanceScheduleWord(words, provenance->prospective_step);
  appendMaintenanceScheduleWord(words, provenance->attempt);
  appendMaintenanceScheduleReal(words, provenance->step_start_time);
  appendMaintenanceScheduleReal(words, provenance->step_end_time);
  appendMaintenanceScheduleReal(words, provenance->q_input_time);
  appendMaintenanceScheduleReal(words, provenance->velocity_state_time);
  appendMaintenanceScheduleReal(words, provenance->time_step);
  appendMaintenanceScheduleWord(words, provenance->operator_state_revision);
  appendMaintenanceScheduleWord(words, provenance->previous_q_revision);
  appendMaintenanceScheduleWord(words, provenance->nodal_velocity_revision);
  appendConservativePhaseReplicatedGraphIdentityWords(
      words, provenance->previous_graph_identity);
  appendConservativePhaseReplicatedGraphIdentityWords(
      words, provenance->operator_graph_identity);
  appendMaintenanceScheduleWord(
      words, provenance->final_flux_ledger_digest);
  appendMaintenanceScheduleReal(
      words, provenance->stage_options.invariant_tolerance);
  appendMaintenanceScheduleReal(
      words, provenance->stage_options.component_activity_tolerance);
  appendMaintenanceScheduleReal(
      words, provenance->stage_options.maximum_courant);
  appendMaintenanceScheduleBool(
      words, provenance->stage_options.enforce_courant_limit);
  appendMaintenanceScheduleBool(
      words, provenance->stage_options.require_constant_preservation);
}

std::vector<ConservativePhaseFixedGraphBinding>
captureConservativePhaseFixedGraphBindings(
    const svmp::FE::systems::FESystem& system,
    std::span<const ConservativePhaseRequest> requests,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  std::vector<ConservativePhaseFixedGraphBinding> bindings;
  std::exception_ptr local_failure;
  try {
    bindings.resize(requests.size());
    for (std::size_t index = 0u; index < requests.size(); ++index) {
      const auto& request = requests[index];
      auto& binding = bindings[index];
      binding.enabled = request.configuration->transport.conservative_phase.enabled;
      binding.phase_field_name =
          request.configuration->transport.conservative_phase.liquid_indicator.field_name;
      if (!binding.enabled) {
        continue;
      }
      if (!request.runtime.conservative_phase_graph.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase fixed-background transport requires an initialized accepted q graph.");
      }
      const auto& graph = *request.runtime.conservative_phase_graph;
      binding.identity =
          svmp::FE::level_set::levelSetP1PhaseGraphIdentity(graph);
      binding.geometry_revision = graph.geometry_revision;
      binding.topology_revision = graph.topology_revision;
      binding.ownership_revision = graph.ownership_revision;
      binding.numbering_revision = graph.numbering_revision;
      if (!graph.success || binding.identity.content_revision == 0u) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase accepted q graph has no valid content identity.");
      }
    }
  } catch (...) {
    local_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_failure, comm, "conservative_phase_fixed_graph_capture");

  std::vector<std::uint64_t> words;
  std::exception_ptr local_word_failure;
  try {
    appendMaintenanceScheduleWord(
        words, static_cast<std::uint64_t>(bindings.size()));
    for (const auto& binding : bindings) {
      appendMaintenanceScheduleBool(words, binding.enabled);
      appendMaintenanceScheduleString(words, binding.phase_field_name);
      if (binding.enabled) {
        appendConservativePhaseReplicatedGraphIdentityWords(
            words, binding.identity);
      }
    }
  } catch (...) {
    local_word_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_word_failure, comm,
      "conservative_phase_fixed_graph_consensus_words");
  if (!collectiveLevelSetMaintenanceCanonicalWordsAgree(
          words, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase accepted q graph identity differs across the active FE communicator.");
  }
  (void)system;
  return bindings;
}

void requireConservativePhaseFixedGraphBindingsCurrent(
    const svmp::FE::systems::FESystem& system,
    std::span<const ConservativePhaseRequest> requests,
    std::span<const ConservativePhaseFixedGraphBinding> bindings,
    const svmp::FE::systems::CollectiveCommunicator& comm,
    std::string_view phase)
{
  bool local_current = bindings.size() == requests.size();
  std::exception_ptr local_failure;
  try {
    const auto& mesh = system.meshAccess();
    if (local_current) {
      for (std::size_t index = 0u; index < requests.size(); ++index) {
        const auto& request = requests[index];
        const auto& binding = bindings[index];
        if (binding.enabled != request.configuration->transport.conservative_phase.enabled ||
            binding.phase_field_name !=
                request.configuration->transport.conservative_phase.liquid_indicator.field_name) {
          local_current = false;
          break;
        }
        if (!binding.enabled) {
          continue;
        }
        const auto phase_field = system.findFieldByName(
            binding.phase_field_name);
        if (phase_field == svmp::FE::INVALID_FIELD_ID ||
            !request.runtime.conservative_phase_graph.has_value()) {
          local_current = false;
          break;
        }
        const auto& graph = *request.runtime.conservative_phase_graph;
        const auto current_identity =
            svmp::FE::level_set::levelSetP1PhaseGraphIdentity(graph);
        const auto& dofs = system.fieldDofHandler(phase_field);
        if (!sameConservativePhaseGraphIdentity(
                current_identity, binding.identity) ||
            graph.geometry_revision != binding.geometry_revision ||
            graph.topology_revision != binding.topology_revision ||
            graph.ownership_revision != binding.ownership_revision ||
            graph.numbering_revision != binding.numbering_revision ||
            mesh.geometryRevision() != binding.geometry_revision ||
            mesh.topologyRevision() != binding.topology_revision ||
            mesh.ownershipRevision() != binding.ownership_revision ||
            mesh.numberingRevision() != binding.numbering_revision ||
            dofs.dofLayoutRevision() !=
                binding.identity.dof_layout_revision) {
          local_current = false;
          break;
        }
      }
    }
  } catch (...) {
    local_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(local_failure, comm, phase);
  if (globalAnyBool(!local_current, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport requires the accepted q^n graph and fixed background mesh to remain unchanged through the physical solve (phase='" +
        std::string(phase) + "').");
  }
}

ConservativePhaseCandidateStageSnapshot
buildConservativePhaseCandidateStageSnapshot(
    const ConservativePhaseWorkflowHost& host,
    svmp::FE::timestepping::TimeHistory& history,
    std::span<ConservativePhaseRequest> requests,
    std::span<const ConservativePhaseFixedGraphBinding> fixed_graph_bindings,
    const svmp::FE::timestepping::CandidateStageObservation& observation,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  requireConservativePhaseFixedGraphBindingsCurrent(
      *host.fe_system,
      requests,
      fixed_graph_bindings,
      comm,
      "candidate_stage_fixed_background_preflight");

  bool local_observation_ready = false;
  std::exception_ptr local_observation_failure;
  try {
    const auto& mesh = host.fe_system->meshAccess();
    const auto expected_end_time = history.time() + history.dt();
    local_observation_ready =
        observation.scheme ==
            svmp::FE::timestepping::SchemeKind::BackwardEuler &&
        observation.temporal_order == 1 &&
        observation.step_index == history.stepIndex() + 1 &&
        observation.attempt_index >= 0 &&
        observation.attempt_index < std::numeric_limits<int>::max() &&
        observation.state_vector != nullptr &&
        observation.state_vector->size() ==
            host.fe_system->dofHandler().getNumDofs() &&
        !observation.generalized_alpha.has_value() &&
        observation.mesh_revision.has_value() &&
        std::isfinite(observation.step_start_time) &&
        std::isfinite(observation.step_end_time) &&
        std::isfinite(observation.state_time) &&
        std::isfinite(observation.dt) && observation.dt > 0.0 &&
        sameConservativePhaseStageRealBits(
            observation.step_start_time, history.time()) &&
        sameConservativePhaseStageRealBits(
            observation.step_end_time, expected_end_time) &&
        sameConservativePhaseStageRealBits(
            observation.state_time, observation.step_end_time) &&
        sameConservativePhaseStageRealBits(
            observation.rate_time, observation.step_end_time) &&
        sameConservativePhaseStageRealBits(
            observation.dt, history.dt());
    if (local_observation_ready) {
      const auto& revision = *observation.mesh_revision;
      local_observation_ready =
          revision.geometry_revision == mesh.geometryRevision() &&
          revision.topology_revision == mesh.topologyRevision() &&
          revision.ownership_revision == mesh.ownershipRevision() &&
          revision.numbering_revision == mesh.numberingRevision() &&
          revision.field_layout_revision == mesh.fieldLayoutRevision() &&
          revision.label_revision == mesh.labelRevision() &&
          revision.active_configuration_epoch ==
              mesh.activeConfigurationEpoch() &&
          revision.coordinate_configuration_key ==
              mesh.coordinateConfigurationKey();
    }
  } catch (...) {
    local_observation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_observation_failure,
      comm,
      "conservative_phase_candidate_stage_metadata");
  if (globalAnyBool(!local_observation_ready, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport requires one exact temporal-order-one Backward Euler endpoint-stage observation on every active FE rank.");
  }

  ConservativePhaseCandidateStageSnapshot snapshot;
  std::exception_ptr local_state_failure;
  try {
    snapshot.temporal_order = observation.temporal_order;
    snapshot.prospective_step =
        static_cast<std::uint64_t>(observation.step_index);
    snapshot.attempt =
        static_cast<std::uint64_t>(observation.attempt_index) + 1u;
    snapshot.step_start_time = observation.step_start_time;
    snapshot.step_end_time = observation.step_end_time;
    snapshot.q_input_time = observation.step_start_time;
    snapshot.velocity_state_time = observation.state_time;
    snapshot.time_step = observation.dt;
    auto request_schedule =
        host.prospective_request_schedule(
            observation.step_index);
    if (!request_schedule.supported) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase operator-stage provenance cannot bind an unsupported maintenance request schedule.");
    }
    snapshot.request_schedule_words =
        std::move(request_schedule.words);
    snapshot.requests.resize(requests.size());
  } catch (...) {
    local_state_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_state_failure,
      comm,
      "conservative_phase_operator_endpoint_state");
  if (!collectiveLevelSetMaintenanceCanonicalWordsAgree(
          snapshot.request_schedule_words, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase request metadata differs across the active FE communicator before source-dependent endpoint velocity sampling.");
  }
  snapshot.operator_state_revision =
      host.operator_state_fingerprint(
          *observation.state_vector);
  if (snapshot.operator_state_revision == 0u) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase operator endpoint has no nonzero content revision.");
  }

  svmp::FE::systems::SystemStateView operator_state;
  operator_state.time = observation.state_time;
  operator_state.dt = observation.dt;
  operator_state.effective_dt = observation.dt;
  operator_state.dt_prev = history.dtPrev();
  operator_state.u_vector = observation.state_vector;
  operator_state.u_prev = history.uPrevSpan();
  operator_state.u_prev_vector = &history.uPrev();

  for (std::size_t index = 0u; index < requests.size(); ++index) {
    auto& request = requests[index];
    auto& staged_request = snapshot.requests[index];
    std::exception_ptr local_request_failure;
    try {
      staged_request.enabled = request.configuration->transport.conservative_phase.enabled;
      staged_request.phase_field_name =
          request.configuration->transport.conservative_phase.liquid_indicator.field_name;
      staged_request.velocity_field_name = request.configuration->transport.velocity.field_name;
      staged_request.velocity_source = request.configuration->transport.velocity.source;
      staged_request.material_interface_marker =
          request.configuration->transport.velocity.material_interface_marker;
      if (staged_request.enabled) {
        if (!request.runtime.conservative_phase_graph.has_value()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Conservative phase stage lost its accepted fixed-background graph.");
        }
        const auto& graph = *request.runtime.conservative_phase_graph;
        staged_request.graph_identity =
            svmp::FE::level_set::levelSetP1PhaseGraphIdentity(graph);
        staged_request.graph_geometry_revision = graph.geometry_revision;
        staged_request.graph_topology_revision = graph.topology_revision;
        staged_request.graph_ownership_revision = graph.ownership_revision;
        staged_request.graph_numbering_revision = graph.numbering_revision;
      }
    } catch (...) {
      local_request_failure = std::current_exception();
    }
    requireCollectivePhasePreparation(
        local_request_failure,
        comm,
        "conservative_phase_operator_stage_request_preparation");
    if (!staged_request.enabled) {
      continue;
    }
    staged_request.sampled_nodal_velocity =
        sampleConservativePhaseVelocity(
            host,
            request,
            *request.runtime.conservative_phase_graph,
            operator_state);
    if (!collectiveConservativePhaseVelocityBitsAgree(
            staged_request.sampled_nodal_velocity, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase sampled endpoint velocity differs bitwise across the active FE communicator.");
    }
  }

  std::vector<std::uint64_t> stage_words;
  std::exception_ptr local_consensus_failure;
  try {
    appendConservativePhaseCandidateStageWords(stage_words, snapshot);
  } catch (...) {
    local_consensus_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_consensus_failure,
      comm,
      "conservative_phase_candidate_stage_consensus_words");
  if (!collectiveLevelSetMaintenanceCanonicalWordsAgree(
          stage_words, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase operator-stage metadata or sampled nodal velocity differs across the active FE communicator.");
  }
  return snapshot;
}

namespace {

std::uint64_t conservativePhaseArtifactFingerprint(
    const ConservativePhaseRequest& request,
    const ConservativePhaseMaintenanceStageLedger& ledger,
    std::uint64_t accepted_step,
    svmp::FE::Real accepted_time,
    svmp::FE::Real time_step,
    std::uint64_t accepted_state_revision)
{
  std::uint64_t fingerprint = 14695981039346656037ull;
  mixConservativePhaseArtifactFingerprint(fingerprint, accepted_step);
  mixConservativePhaseArtifactFingerprint(fingerprint, accepted_time);
  mixConservativePhaseArtifactFingerprint(fingerprint, time_step);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, accepted_state_revision);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, request.configuration->transport.level_set.field_name);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      request.configuration->transport.conservative_phase.liquid_indicator.field_name);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      request.configuration->volume_cut_request.has_value()
          ? std::string_view(request.configuration->volume_cut_request->domain_id)
          : std::string_view{});
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      request.configuration->transport.conservative_phase
          .classify_nonprimary_components_as_satellites);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, request.runtime.conservative_phase_graph.has_value());
  if (request.runtime.conservative_phase_graph.has_value()) {
    const auto live_graph_identity =
        svmp::FE::level_set::levelSetP1PhaseGraphIdentity(
            *request.runtime.conservative_phase_graph);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, live_graph_identity.dimension);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, live_graph_identity.nodes);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, live_graph_identity.edges);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, live_graph_identity.dof_layout_revision);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, live_graph_identity.content_revision);
  }
  mixConservativePhaseArtifactFingerprint(
      fingerprint, request.configuration->transport.conservative_phase.fixed_flux_regions.size());
  for (const auto& box : request.configuration->transport.conservative_phase.fixed_flux_regions) {
    mixConservativePhaseArtifactFingerprint(fingerprint, box.name);
    mixConservativePhaseArtifactFingerprint(fingerprint, box.kind);
    for (std::size_t dimension = 0u; dimension < 3u; ++dimension) {
      mixConservativePhaseArtifactFingerprint(
          fingerprint, box.minimum[dimension]);
      mixConservativePhaseArtifactFingerprint(
          fingerprint, box.maximum[dimension]);
    }
  }
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.split_stage_provenance.has_value());
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      svmp::FE::level_set::levelSetP1PhaseFluxLedgerDigest(
          ledger.transport_stage));
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      svmp::FE::level_set::levelSetP1PhaseVelocityContentRevision(
          ledger.transport_stage.sampled_nodal_velocity));
  if (ledger.split_stage_provenance.has_value()) {
    const auto& provenance = *ledger.split_stage_provenance;
    const auto mix_replicated_graph = [&](const auto& identity) {
      mixConservativePhaseArtifactFingerprint(
          fingerprint, identity.dimension);
      mixConservativePhaseArtifactFingerprint(
          fingerprint, identity.nodes);
      mixConservativePhaseArtifactFingerprint(
          fingerprint, identity.edges);
      mixConservativePhaseArtifactFingerprint(
          fingerprint, identity.dof_layout_revision);
      mixConservativePhaseArtifactFingerprint(
          fingerprint, identity.content_revision);
    };
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.scheme);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.transport_mesh_policy);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.temporal_order);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.prospective_step);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.attempt);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.step_start_time);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.step_end_time);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.q_input_time);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.velocity_state_time);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.time_step);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.operator_state_revision);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.previous_q_revision);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.nodal_velocity_revision);
    mix_replicated_graph(provenance.previous_graph_identity);
    mix_replicated_graph(provenance.operator_graph_identity);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.final_flux_ledger_digest);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.stage_options.invariant_tolerance);
    mixConservativePhaseArtifactFingerprint(
        fingerprint,
        provenance.stage_options.component_activity_tolerance);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.stage_options.maximum_courant);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, provenance.stage_options.enforce_courant_limit);
    mixConservativePhaseArtifactFingerprint(
        fingerprint,
        provenance.stage_options.require_constant_preservation);
  }
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.reinitialization_due);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.reinitialization_applied);
  const auto& repair = ledger.reinitialization;
  mixConservativePhaseArtifactFingerprint(fingerprint, repair.success);
  mixConservativePhaseArtifactFingerprint(fingerprint, repair.method);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.repaired_dofs);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.preserved_dofs);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.interface_fragments);
  mixConservativePhaseArtifactFingerprint(fingerprint, repair.cut_cells);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.interface_displacement_samples);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_abs_update);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_distance);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_interface_displacement);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.l2_interface_displacement);
  mixConservativePhaseArtifactFingerprint(fingerprint, repair.iterations);
  mixConservativePhaseArtifactFingerprint(fingerprint, repair.converged);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.zero_set_bound_satisfied);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_iteration_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_signed_distance_error);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_unconstrained_signed_distance_error);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_wall_constrained_signed_distance_error);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.wall_contact_constraints);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.wall_contact_cells);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.wall_contact_dofs);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.wall_contact_constraints_satisfied);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_wall_contact_scale_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_prescribed_contact_value_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      repair.max_prescribed_contact_angle_error_radians);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_contact_line_displacement);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.max_contact_angle_change_radians);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.preserve_band_width);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, repair.diagnostic);

  const auto& reconciliation = ledger.reconciliation;
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.success);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.target_reached);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.limited_by_displacement);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.limited_by_topology);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.iterations);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.line_search_evaluations);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.geometry_refresh_requests);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.geometry_rebuilds);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.rejected_geometry_trials);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.contact_protected_nodes);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.allowed_interface_displacement);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      reconciliation.accumulated_interface_displacement_bound);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.initial_residual_norm);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.final_residual_norm);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.maximum_final_nodal_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.final_total_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.maximum_removed_contact_increment);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.last_rejected_trial_diagnostic);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, reconciliation.diagnostic);

  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.raw_post_transport_phase_measure);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.post_limit_phase_measure);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.post_correction_phase_measure);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.raw_post_transport_geometry_measure);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.post_reinitialization_geometry_measure);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      ledger.post_reinitialization_mismatch.maximum_nodal_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.post_reinitialization_mismatch.residual_norm);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.post_reinitialization_mismatch.total_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.post_correction_geometry_measure);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      ledger.post_correction_mismatch.maximum_nodal_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.post_correction_mismatch.residual_norm);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.post_correction_mismatch.total_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.boundary_flux_policy);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.maximum_nodal_boundary_mass_transfer);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.boundary_mass_tolerance);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      std::string_view{
          svmp::FE::level_set::
              levelSetConservativePhaseBoundaryFluxScope(
                  ledger.boundary_flux_policy)});

  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.region_ledger.success);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.region_ledger.all_balances_satisfied);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.region_ledger.maximum_balance_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      ledger.region_ledger.maximum_flux_reconstruction_residual);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.region_ledger.diagnostic);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, ledger.region_ledger.regions.size());
  for (const auto& region : ledger.region_ledger.regions) {
    mixConservativePhaseArtifactFingerprint(fingerprint, region.name);
    mixConservativePhaseArtifactFingerprint(fingerprint, region.kind);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.balance_satisfied);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.member_nodes.size());
    for (const auto node : region.member_nodes) {
      mixConservativePhaseArtifactFingerprint(fingerprint, node);
    }
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.crossing_edges.size());
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.internal_edges);
    for (const auto& edge : region.crossing_edges) {
      mixConservativePhaseArtifactFingerprint(
          fingerprint, edge.first_node);
      mixConservativePhaseArtifactFingerprint(
          fingerprint, edge.second_node);
      mixConservativePhaseArtifactFingerprint(
          fingerprint, edge.low_order_mass_transfer_into_region);
      mixConservativePhaseArtifactFingerprint(
          fingerprint,
          edge.raw_antidiffusive_mass_transfer_into_region);
      mixConservativePhaseArtifactFingerprint(
          fingerprint,
          edge.limited_antidiffusive_mass_transfer_into_region);
    }
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.previous_liquid_measure);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.low_order_liquid_measure);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.raw_target_liquid_measure);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.limited_liquid_measure);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.physical_boundary_mass_transfer);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.discrete_divergence_mass_source);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.low_order_nodal_interior_mass_transfer);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.raw_nodal_antidiffusive_mass_transfer);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.limited_nodal_antidiffusive_mass_transfer);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.low_order_crossing_mass_transfer);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.raw_crossing_antidiffusive_mass_transfer);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.limited_crossing_antidiffusive_mass_transfer);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.low_order_flux_reconstruction_residual);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.raw_flux_reconstruction_residual);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.limited_flux_reconstruction_residual);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.low_order_balance_residual);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.raw_target_balance_residual);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.limited_balance_residual);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.maximum_internal_pair_cancellation_residual);
  }
  return fingerprint;
}

} // namespace

void writeAcceptedConservativePhaseArtifacts(
    std::string_view results_folder,
    std::span<const ConservativePhaseRequest> requests,
    const ConservativePhaseCandidateResult& candidate,
    std::uint64_t accepted_step,
    svmp::FE::Real accepted_time,
    svmp::FE::Real time_step,
    std::uint64_t state_revision,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  const bool locally_any_artifact_configured =
      std::any_of(
          requests.begin(), requests.end(), [](const auto& request) {
            return request.configuration->transport.conservative_phase.enabled &&
                   request.configuration->transport.conservative_phase.write_flux_artifacts;
          });
  if (!globalAnyBool(locally_any_artifact_configured, comm)) {
    return;
  }

  const double local_request_count =
      static_cast<double>(requests.size());
  if (globalMinDouble(local_request_count, comm) !=
      globalMaxDouble(local_request_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase artifact request count differs across ranks.");
  }

  std::filesystem::path output_directory;
  std::vector<std::uint64_t> output_policy_words;
  std::vector<bool> artifact_due;
  std::exception_ptr local_artifact_preparation_failure;
  try {
    output_directory = ".";
    if (!results_folder.empty()) {
      output_directory = results_folder;
    }
    output_directory /= "conservative_phase_flux";
    output_directory =
        std::filesystem::absolute(output_directory).lexically_normal();
    const auto output_directory_key = output_directory.generic_string();
    output_policy_words.reserve(2u + output_directory_key.size());
    output_policy_words.push_back(1u);
    appendMaintenanceScheduleString(
        output_policy_words, output_directory_key);
    artifact_due.assign(requests.size(), false);
  } catch (...) {
    local_artifact_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_artifact_preparation_failure,
      comm,
      "conservative phase artifact output-policy preparation");
  if (!collectiveLevelSetMaintenanceCanonicalWordsAgree(
          output_policy_words, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase artifact output directory differs across ranks.");
  }

  bool any_artifact_due = false;
  bool local_preflight_failure = false;
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const auto& request = requests[index];
    const int cadence =
        request.configuration->transport.conservative_phase.flux_artifact_cadence_steps;
    const bool locally_configured =
        request.configuration->transport.conservative_phase.enabled &&
        request.configuration->transport.conservative_phase.write_flux_artifacts;
    const bool configured_on_any_rank =
        globalAnyBool(locally_configured, comm);
    const bool not_configured_on_any_rank =
        globalAnyBool(!locally_configured, comm);
    if (configured_on_any_rank && not_configured_on_any_rank) {
      local_preflight_failure = true;
    }
    if (configured_on_any_rank &&
        globalMinDouble(static_cast<double>(cadence), comm) !=
            globalMaxDouble(static_cast<double>(cadence), comm)) {
      local_preflight_failure = true;
    }
    if (locally_configured && cadence <= 0) {
      local_preflight_failure = true;
    }
    const bool locally_due =
        locally_configured && cadence > 0 &&
        accepted_step % static_cast<std::uint64_t>(cadence) == 0u;
    const bool due_on_any_rank = globalAnyBool(locally_due, comm);
    const bool not_due_on_any_rank = globalAnyBool(!locally_due, comm);
    if (due_on_any_rank && not_due_on_any_rank) {
      local_preflight_failure = true;
    }
    artifact_due[index] = due_on_any_rank && !not_due_on_any_rank;
    any_artifact_due = any_artifact_due || artifact_due[index];
    if (artifact_due[index]) {
      bool local_ledger_ready = false;
      std::uint64_t fingerprint = 0u;
      std::exception_ptr local_fingerprint_preparation_failure;
      try {
        const auto live_graph_identity =
            request.runtime.conservative_phase_graph.has_value()
                ? svmp::FE::level_set::levelSetP1PhaseGraphIdentity(
                      *request.runtime.conservative_phase_graph)
                : svmp::FE::level_set::LevelSetP1PhaseGraphIdentity{};
        const auto live_flux_ledger_digest =
            candidate.maintenance_ledgers.size() > index
                ? svmp::FE::level_set::levelSetP1PhaseFluxLedgerDigest(
                      candidate.maintenance_ledgers[index]
                          .transport_stage)
                : 0u;
        const auto live_velocity_revision =
            candidate.maintenance_ledgers.size() > index
                ? svmp::FE::level_set::
                      levelSetP1PhaseVelocityContentRevision(
                          candidate.maintenance_ledgers[index]
                              .transport_stage.sampled_nodal_velocity)
                : 0u;
        svmp::FE::Real live_maximum_nodal_boundary_transfer{0.0};
        bool live_nodal_boundary_transfer_finite = true;
        if (candidate.maintenance_ledgers.size() > index) {
          for (const auto transfer :
               candidate.maintenance_ledgers[index]
                   .transport_stage.physical_boundary_mass_transfer) {
            live_nodal_boundary_transfer_finite =
                std::isfinite(transfer) &&
                live_nodal_boundary_transfer_finite;
            if (std::isfinite(transfer)) {
              live_maximum_nodal_boundary_transfer = std::max(
                  live_maximum_nodal_boundary_transfer,
                  std::abs(transfer));
            }
          }
        }
        local_ledger_ready =
            candidate.maintenance_ledgers.size() == requests.size() &&
            accepted_step != 0u &&
            std::isfinite(accepted_time) &&
            std::isfinite(time_step) &&
            time_step > svmp::FE::Real{0.0} &&
            state_revision != 0u &&
            request.runtime.conservative_phase_graph.has_value() &&
            request.configuration->volume_cut_request.has_value() &&
            candidate.maintenance_ledgers.size() > index &&
            candidate.maintenance_ledgers[index]
                .split_stage_provenance.has_value() &&
            candidate.maintenance_ledgers[index]
                    .split_stage_provenance->prospective_step ==
                accepted_step &&
            sameConservativePhaseStageRealBits(
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->step_end_time,
                accepted_time) &&
            sameConservativePhaseStageRealBits(
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->velocity_state_time,
                accepted_time) &&
            sameConservativePhaseStageRealBits(
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->time_step,
                time_step) &&
            sameConservativePhaseStageRealBits(
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->q_input_time,
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->step_start_time) &&
            sameConservativePhaseStageRealBits(
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->step_end_time,
                candidate.maintenance_ledgers[index]
                        .split_stage_provenance->step_start_time +
                    candidate.maintenance_ledgers[index]
                        .split_stage_provenance->time_step) &&
            live_graph_identity.dimension ==
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->operator_graph_identity
                    .dimension &&
            live_graph_identity.nodes ==
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->operator_graph_identity.nodes &&
            live_graph_identity.edges ==
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->operator_graph_identity.edges &&
            live_graph_identity.dof_layout_revision ==
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->operator_graph_identity
                    .dof_layout_revision &&
            live_graph_identity.content_revision ==
                candidate.maintenance_ledgers[index]
                    .split_stage_provenance->operator_graph_identity
                    .content_revision &&
            live_flux_ledger_digest != 0u &&
            live_velocity_revision != 0u &&
            candidate.maintenance_ledgers[index]
                    .split_stage_provenance->nodal_velocity_revision ==
                live_velocity_revision &&
            candidate.maintenance_ledgers[index]
                    .split_stage_provenance->final_flux_ledger_digest ==
                live_flux_ledger_digest &&
            std::isfinite(
                candidate.maintenance_ledgers[index]
                    .maximum_nodal_boundary_mass_transfer) &&
            live_nodal_boundary_transfer_finite &&
            sameConservativePhaseStageRealBits(
                candidate.maintenance_ledgers[index]
                    .maximum_nodal_boundary_mass_transfer,
                live_maximum_nodal_boundary_transfer) &&
            std::isfinite(
                candidate.maintenance_ledgers[index]
                    .boundary_mass_tolerance) &&
            candidate.maintenance_ledgers[index]
                    .boundary_mass_tolerance >= svmp::FE::Real{0.0} &&
            candidate.maintenance_ledgers[index]
                    .boundary_flux_policy ==
                request.configuration->transport.conservative_phase.boundary_flux_policy &&
            svmp::FE::level_set::
                levelSetConservativePhaseBoundaryFluxSatisfied(
                    candidate.maintenance_ledgers[index]
                        .boundary_flux_policy,
                    candidate.maintenance_ledgers[index]
                        .maximum_nodal_boundary_mass_transfer,
                    candidate.maintenance_ledgers[index]
                        .transport_stage.correction
                        .total_physical_boundary_mass_transfer,
                    candidate.maintenance_ledgers[index]
                        .boundary_mass_tolerance) &&
            candidate.maintenance_ledgers[index].transport_stage.success &&
            candidate.maintenance_ledgers[index].region_ledger.success;
        if (local_ledger_ready) {
          fingerprint = conservativePhaseArtifactFingerprint(
              request,
              candidate.maintenance_ledgers[index],
              accepted_step,
              accepted_time,
              time_step,
              state_revision);
        }
      } catch (...) {
        local_fingerprint_preparation_failure =
            std::current_exception();
      }
      requireCollectivePhasePreparation(
          local_fingerprint_preparation_failure,
          comm,
          "conservative phase artifact fingerprint preparation");
      local_preflight_failure =
          !local_ledger_ready || local_preflight_failure;
      const auto [minimum_fingerprint, maximum_fingerprint] =
          globalMinMaxUint64(fingerprint, comm);
      if (minimum_fingerprint != maximum_fingerprint) {
        local_preflight_failure = true;
      }
    }
  }
  if (globalAnyBool(local_preflight_failure, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase artifact preflight failed on at least one rank.");
  }
  if (!any_artifact_due) {
    return;
  }

  struct PublishedConservativePhaseArtifact {
    std::string phase_field_name{};
    svmp::FE::level_set::LevelSetConservativePhaseArtifactResult result{};
  };
  std::vector<PublishedConservativePhaseArtifact> published_artifacts;
  std::exception_ptr local_publication_storage_failure;
  try {
    published_artifacts.reserve(static_cast<std::size_t>(std::count(
        artifact_due.begin(), artifact_due.end(), true)));
  } catch (...) {
    local_publication_storage_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_publication_storage_failure,
      comm,
      "conservative phase artifact publication storage");

  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const auto& request = requests[index];
    if (!artifact_due[index]) {
      continue;
    }
    const auto& ledger = candidate.maintenance_ledgers[index];
    const auto& graph = *request.runtime.conservative_phase_graph;
    const auto& repair = ledger.reinitialization;
    const auto& reconciliation = ledger.reconciliation;
    svmp::FE::level_set::LevelSetConservativePhaseArtifactResult artifact;
    std::exception_ptr local_publication_failure;
    bool current_artifact_recorded = false;
    if (comm.rank() == 0) {
      try {
        svmp::FE::level_set::LevelSetConservativePhaseArtifactContext
            context;
        context.phase_field_name =
            request.configuration->transport.conservative_phase.liquid_indicator.field_name;
        context.level_set_field_name = request.configuration->transport.level_set.field_name;
        context.geometry_domain_id = request.configuration->volume_cut_request->domain_id;
        context.accepted_step = accepted_step;
        context.accepted_time = accepted_time;
        context.time_step = time_step;
        context.state_revision = state_revision;
        context.graph_dimension = graph.dimension;
        context.graph_nodes = graph.nodes;
        context.graph_edges = graph.edges.size();
        context.graph_geometry_revision = graph.geometry_revision;
        context.graph_topology_revision = graph.topology_revision;
        context.graph_ownership_revision = graph.ownership_revision;
        context.graph_numbering_revision = graph.numbering_revision;
        context.graph_dof_layout_revision = graph.dof_layout_revision;
        context.graph_content_revision =
            svmp::FE::level_set::levelSetP1PhaseGraphIdentity(graph)
                .content_revision;
        context.split_stage_provenance =
            ledger.split_stage_provenance;
        context.maximum_nodal_boundary_mass_transfer =
            ledger.maximum_nodal_boundary_mass_transfer;
        context.boundary_mass_tolerance =
            ledger.boundary_mass_tolerance;
        context.boundary_flux_policy =
            ledger.boundary_flux_policy;
        context.boundary_flux_scope =
            svmp::FE::level_set::
                levelSetConservativePhaseBoundaryFluxScope(
                    ledger.boundary_flux_policy);
        context.geometry_validated_before_commit = true;
        context.reinitialization_due = ledger.reinitialization_due;
        context.reinitialization_applied =
            ledger.reinitialization_applied;
        context.reinitialization = repair;
        context.reconciliation.success = reconciliation.success;
        context.reconciliation.target_reached =
            reconciliation.target_reached;
        context.reconciliation.limited_by_displacement =
            reconciliation.limited_by_displacement;
        context.reconciliation.limited_by_topology =
            reconciliation.limited_by_topology;
        context.reconciliation.iterations = reconciliation.iterations;
        context.reconciliation.line_search_evaluations =
            reconciliation.line_search_evaluations;
        context.reconciliation.geometry_refresh_requests =
            reconciliation.geometry_refresh_requests;
        context.reconciliation.geometry_rebuilds =
            reconciliation.geometry_rebuilds;
        context.reconciliation.rejected_geometry_trials =
            reconciliation.rejected_geometry_trials;
        context.reconciliation.contact_protected_nodes =
            reconciliation.contact_protected_nodes;
        context.reconciliation.allowed_interface_displacement =
            reconciliation.allowed_interface_displacement;
        context.reconciliation.accumulated_interface_displacement_bound =
            reconciliation.accumulated_interface_displacement_bound;
        context.reconciliation.initial_residual_norm =
            reconciliation.initial_residual_norm;
        context.reconciliation.final_residual_norm =
            reconciliation.final_residual_norm;
        context.reconciliation.maximum_final_nodal_residual =
            reconciliation.maximum_final_nodal_residual;
        context.reconciliation.final_total_residual =
            reconciliation.final_total_residual;
        context.reconciliation.maximum_removed_contact_increment =
            reconciliation.maximum_removed_contact_increment;
        context.reconciliation.last_rejected_trial_diagnostic =
            reconciliation.last_rejected_trial_diagnostic;
        context.reconciliation.diagnostic = reconciliation.diagnostic;
        context.raw_post_transport_phase_measure =
            ledger.raw_post_transport_phase_measure;
        context.post_limit_phase_measure = ledger.post_limit_phase_measure;
        context.raw_post_transport_geometry_measure =
            ledger.raw_post_transport_geometry_measure;
        context.post_reinitialization_phase_measure =
            ledger.post_limit_phase_measure;
        context.post_reinitialization_geometry_measure =
            ledger.post_reinitialization_geometry_measure;
        context.post_reinitialization_mismatch.maximum_nodal_residual =
            ledger.post_reinitialization_mismatch.maximum_nodal_residual;
        context.post_reinitialization_mismatch.residual_norm =
            ledger.post_reinitialization_mismatch.residual_norm;
        context.post_reinitialization_mismatch.total_residual =
            ledger.post_reinitialization_mismatch.total_residual;
        context.post_correction_phase_measure =
            ledger.post_correction_phase_measure;
        context.post_correction_geometry_measure =
            ledger.post_correction_geometry_measure;
        context.post_correction_mismatch.maximum_nodal_residual =
            ledger.post_correction_mismatch.maximum_nodal_residual;
        context.post_correction_mismatch.residual_norm =
            ledger.post_correction_mismatch.residual_norm;
        context.post_correction_mismatch.total_residual =
            ledger.post_correction_mismatch.total_residual;
        context.retained_assembly_geometry_measure =
            ledger.post_correction_geometry_measure;
        context.region_ledger = ledger.region_ledger;
        artifact =
            svmp::FE::level_set::writeLevelSetConservativePhaseArtifact(
                output_directory, context, ledger.transport_stage);
        if (artifact.success) {
          published_artifacts.push_back(
              PublishedConservativePhaseArtifact{
                  .phase_field_name =
                      request.configuration->transport.conservative_phase.liquid_indicator
                          .field_name,
                  .result = artifact,
              });
          current_artifact_recorded = true;
        }
      } catch (...) {
        local_publication_failure = std::current_exception();
      }
    }
    const bool publication_failed = globalAnyBool(
        comm.rank() == 0 &&
            (local_publication_failure != nullptr || !artifact.success),
        comm);
    if (publication_failed) {
      bool local_rollback_failed = false;
      if (comm.rank() == 0) {
        try {
          if (artifact.success && !current_artifact_recorded) {
            std::error_code removal_error;
            const bool removed =
                std::filesystem::remove(artifact.path, removal_error);
            local_rollback_failed =
                !removed || static_cast<bool>(removal_error);
          }
          for (const auto& published : published_artifacts) {
            std::error_code removal_error;
            const bool removed = std::filesystem::remove(
                published.result.path, removal_error);
            local_rollback_failed =
                !removed || static_cast<bool>(removal_error) ||
                local_rollback_failed;
          }
        } catch (...) {
          local_rollback_failed = true;
        }
      }
      if (globalAnyBool(local_rollback_failed, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase artifact publication and collective batch rollback both failed.");
      }
      if (comm.rank() == 0) {
        svmp::FE::rank_log::oopCout()
            << "[svMultiPhysics::Application] Conservative phase artifact"
            << " diagnostic=conservative_phase_flux_artifact"
            << " field='"
            << request.configuration->transport.conservative_phase.liquid_indicator.field_name
            << "' step=" << accepted_step
            << " outcome=failed"
            << " batch_rollback=complete"
            << " removed_prior_files=" << published_artifacts.size()
            << " reason='"
            << (local_publication_failure != nullptr
                    ? "exception during output-rank context construction or publication"
                    : artifact.diagnostic)
            << "'" << std::endl;
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase artifact publication failed on the output rank; earlier files from this accepted-step batch were removed.");
    }
  }

  if (comm.rank() == 0) {
    for (const auto& published : published_artifacts) {
      svmp::FE::rank_log::oopCout()
          << "[svMultiPhysics::Application] Conservative phase artifact"
          << " diagnostic=conservative_phase_flux_artifact"
          << " field='"
          << published.phase_field_name << "'"
          << " step=" << accepted_step
          << " outcome=written"
          << " path='" << published.result.path.string() << "'"
          << " bytes=" << published.result.bytes
          << " nodes=" << published.result.nodes
          << " edges=" << published.result.edges
          << " resolved_components="
          << published.result.resolved_components
          << " tracked_regions=" << published.result.tracked_regions
          << " subthreshold_component_present="
          << (published.result.subthreshold_component_present ? "true"
                                                              : "false")
          << std::endl;
    }
  }
}

// Moved from Application/Core/ApplicationDriver.cpp (lines 25914-27246 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.

ConservativePhaseCandidateResult applyConservativePhaseCandidates(
    const ConservativePhaseWorkflowHost& host,
    svmp::FE::timestepping::TimeHistory& history,
    std::span<ConservativePhaseRequest> requests,
    ConservativePhaseCandidateStageSnapshot& temporal_stage,
    std::span<const ConservativePhaseFixedGraphBinding>
        fixed_graph_bindings,
    std::uint64_t expected_attempt,
    const ConservativePhaseContactStageCallback& contact_stage_builder,
    const ConservativePhaseStageObserver& observe_stage)
{
  ConservativePhaseCandidateResult result;
  if (!host.fe_system || requests.empty()) {
    return result;
  }
  const bool any_enabled = std::any_of(
      requests.begin(), requests.end(), [](const auto& request) {
        return request.configuration->transport.conservative_phase.enabled;
      });
  if (!any_enabled) {
    return result;
  }
  if (!(history.dt() > 0.0) || !std::isfinite(history.dt())) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport requires a positive finite candidate time step.");
  }

  auto& system = *host.fe_system;
  const auto comm = activeFESystemCommunicator(system);
  requireConservativePhaseFixedGraphBindingsCurrent(
      system,
      requests,
      fixed_graph_bindings,
      comm,
      "candidate_application_fixed_background_preflight");
  bool local_stage_matches =
      temporal_stage.scheme ==
          svmp::FE::level_set::LevelSetP1PhaseSplitScheme::
              BackwardEulerExplicitIndicatorEndpointVelocity &&
      temporal_stage.temporal_order == 1 &&
      temporal_stage.prospective_step ==
          static_cast<std::uint64_t>(history.stepIndex() + 1) &&
      expected_attempt != 0u &&
      temporal_stage.attempt == expected_attempt &&
      temporal_stage.operator_state_revision != 0u &&
      temporal_stage.requests.size() == requests.size() &&
      sameConservativePhaseStageRealBits(
          temporal_stage.step_start_time, history.time()) &&
      sameConservativePhaseStageRealBits(
          temporal_stage.step_end_time, history.time() + history.dt()) &&
      sameConservativePhaseStageRealBits(
          temporal_stage.q_input_time, history.time()) &&
      sameConservativePhaseStageRealBits(
          temporal_stage.velocity_state_time,
          history.time() + history.dt()) &&
      sameConservativePhaseStageRealBits(
          temporal_stage.time_step, history.dt());
  std::exception_ptr local_request_schedule_failure;
  CanonicalLevelSetMaintenanceSchedule current_request_schedule;
  try {
    current_request_schedule =
        host.prospective_request_schedule(
            history.stepIndex() + 1);
    local_stage_matches =
        current_request_schedule.supported &&
        current_request_schedule.words ==
            temporal_stage.request_schedule_words &&
        local_stage_matches;
  } catch (...) {
    local_request_schedule_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_request_schedule_failure,
      comm,
      "conservative_phase_candidate_request_revalidation");
  if (local_stage_matches) {
    for (std::size_t index = 0u; index < requests.size(); ++index) {
      const auto& request = requests[index];
      const auto& staged_request = temporal_stage.requests[index];
      if (staged_request.enabled != request.configuration->transport.conservative_phase.enabled ||
          staged_request.phase_field_name !=
              request.configuration->transport.conservative_phase.liquid_indicator.field_name ||
          staged_request.velocity_field_name != request.configuration->transport.velocity.field_name ||
          staged_request.velocity_source != request.configuration->transport.velocity.source ||
          staged_request.material_interface_marker !=
              request.configuration->transport.velocity.material_interface_marker) {
        local_stage_matches = false;
        break;
      }
      if (!staged_request.enabled) {
        continue;
      }
      if (!request.runtime.conservative_phase_graph.has_value()) {
        local_stage_matches = false;
        break;
      }
      const auto& graph = *request.runtime.conservative_phase_graph;
      if (!sameConservativePhaseGraphIdentity(
              staged_request.graph_identity,
              svmp::FE::level_set::levelSetP1PhaseGraphIdentity(graph)) ||
          staged_request.graph_geometry_revision !=
              graph.geometry_revision ||
          staged_request.graph_topology_revision !=
              graph.topology_revision ||
          staged_request.graph_ownership_revision !=
              graph.ownership_revision ||
          staged_request.graph_numbering_revision !=
              graph.numbering_revision ||
          staged_request.sampled_nodal_velocity.size() != graph.nodes) {
        local_stage_matches = false;
        break;
      }
    }
  }
  if (globalAnyBool(!local_stage_matches, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase candidate is missing an exact current immutable Backward Euler operator-stage velocity snapshot.");
  }
  if (!collectiveConservativePhaseCandidateStageSnapshotAgrees(
          temporal_stage, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase candidate-stage metadata or immutable endpoint velocity diverged before application.");
  }

  std::vector<svmp::FE::Real> previous_solution;
  std::vector<svmp::FE::Real> candidate;
  std::vector<svmp::FE::Real> raw_transport_candidate;
  std::vector<svmp::FE::Real> accepted_phase_measures;
  std::vector<std::vector<svmp::FE::Real>> accepted_phase_masses;
  std::vector<ConservativePhaseGeometryReconciliationResult>
      reconciliation_reports;
  std::vector<std::vector<std::uint8_t>> contact_protected_nodes;
  std::exception_ptr local_candidate_storage_failure;
  try {
    result.original_solution =
        gatherFeOrderedSolution(history.u(), comm);
    previous_solution =
        gatherFeOrderedSolution(history.uPrev(), comm);
    if (result.original_solution.size() != previous_solution.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase transport requires equal endpoint and previous FE solution layouts.");
    }
    candidate = result.original_solution;
    raw_transport_candidate = result.original_solution;
    accepted_phase_measures.assign(
        requests.size(),
        std::numeric_limits<svmp::FE::Real>::quiet_NaN());
    accepted_phase_masses.resize(requests.size());
    reconciliation_reports.resize(requests.size());
    contact_protected_nodes.resize(requests.size());
    result.maintenance_ledgers.resize(requests.size());
  } catch (...) {
    local_candidate_storage_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_candidate_storage_failure,
      comm,
      "conservative_phase_candidate_dense_storage_before_stage");

  for (std::size_t request_index = 0u;
       request_index < requests.size(); ++request_index) {
    auto& request = requests[request_index];
    if (!request.configuration->transport.conservative_phase.enabled) {
      continue;
    }
    auto& staged_request = temporal_stage.requests[request_index];
    svmp::FE::level_set::LevelSetP1PhaseTransportGraph* graph_ptr =
        nullptr;
    svmp::FE::FieldId phase_field = svmp::FE::INVALID_FIELD_ID;
    std::size_t phase_offset{0u};
    std::size_t phase_count{0u};
    std::exception_ptr local_request_setup_failure;
    try {
      if (!request.runtime.conservative_phase_initialized) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase transport reached a candidate before its geometry projection was initialized.");
      }
      requireConservativePhaseGeometryBinding(system, request);
      if (!request.runtime.conservative_phase_graph.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase candidate lost its fixed transport graph.");
      }
      graph_ptr = &*request.runtime.conservative_phase_graph;
      phase_field = system.findFieldByName(
          request.configuration->transport.conservative_phase.liquid_indicator.field_name);
      if (phase_field == svmp::FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase candidate could not resolve its liquid-indicator field.");
      }
      const auto raw_phase_offset = system.fieldDofOffset(phase_field);
      const auto raw_phase_count =
          system.fieldDofHandler(phase_field).getNumDofs();
      if (raw_phase_offset < 0 || raw_phase_count < 0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase graph found an invalid FE solution layout.");
      }
      phase_offset = static_cast<std::size_t>(raw_phase_offset);
      phase_count = static_cast<std::size_t>(raw_phase_count);
      if (phase_count != graph_ptr->nodes ||
          phase_offset > previous_solution.size() ||
          phase_count > previous_solution.size() - phase_offset ||
          phase_offset > candidate.size() ||
          phase_count > candidate.size() - phase_offset) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase graph does not match the FE solution layout.");
      }
    } catch (...) {
      local_request_setup_failure = std::current_exception();
    }
    requireCollectivePhasePreparation(
        local_request_setup_failure,
        comm,
        "conservative_phase_request_setup_before_stage");
    auto& graph = *graph_ptr;
    const std::span<const svmp::FE::Real> previous_phase(
        previous_solution.data() + phase_offset, phase_count);
    std::pair<std::vector<svmp::FE::Real>,
              std::vector<svmp::FE::Real>> one_ring_bounds;
    std::vector<svmp::FE::Real> limited_phase;
    std::vector<svmp::FE::Real> raw_phase;
    std::exception_ptr local_stage_input_failure;
    try {
      one_ring_bounds = conservativePhaseOneRingBounds(
          graph, previous_phase);
      limited_phase.assign(graph.nodes, svmp::FE::Real{0.0});
      raw_phase.assign(graph.nodes, svmp::FE::Real{0.0});
      accepted_phase_masses[request_index].assign(
          graph.nodes, svmp::FE::Real{0.0});
    } catch (...) {
      local_stage_input_failure = std::current_exception();
    }
    requireCollectivePhasePreparation(
        local_stage_input_failure,
        comm,
        "conservative_phase_stage_input_and_output_allocation");
    const auto& lower = one_ring_bounds.first;
    const auto& upper = one_ring_bounds.second;

    svmp::FE::level_set::LevelSetP1PhaseStageOptions stage_options;
    stage_options.invariant_tolerance =
        request.configuration->transport.conservative_phase.invariant_tolerance;
    stage_options.component_activity_tolerance =
        request.configuration->transport.conservative_phase.component_activity_tolerance;
    stage_options.maximum_courant =
        request.configuration->transport.conservative_phase.maximum_courant;
    stage_options.enforce_courant_limit =
        request.configuration->transport.conservative_phase.enforce_courant_limit;
    stage_options.require_constant_preservation =
        request.configuration->transport.conservative_phase.require_constant_preservation;
    auto stage =
        svmp::FE::level_set::advanceLevelSetP1ConservativePhaseStage(
            graph,
            previous_phase,
            lower,
            upper,
            staged_request.sampled_nodal_velocity,
            static_cast<svmp::FE::Real>(history.dt()),
            stage_options);
    std::vector<std::array<svmp::FE::Real, 3>>{}.swap(
        staged_request.sampled_nodal_velocity);
    const bool any_stage_success = globalAnyBool(stage.success, comm);
    const bool any_stage_failure = globalAnyBool(!stage.success, comm);
    if (any_stage_success && any_stage_failure) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase graph-stage success status differs across the active FE communicator.");
    }
    if (any_stage_failure) {
      const bool local_courant_rejection =
          !stage.courant_satisfied &&
          request.configuration->transport.conservative_phase.enforce_courant_limit;
      const bool any_courant_rejection =
          globalAnyBool(local_courant_rejection, comm);
      const bool any_non_courant_failure =
          globalAnyBool(!local_courant_rejection, comm);
      if (any_courant_rejection && any_non_courant_failure) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase graph-stage rejection reason differs across the active FE communicator.");
      }
      if (any_courant_rejection) {
        svmp::FE::rank_log::oopCout()
            << "[svMultiPhysics::Application] Conservative phase candidate rejected"
            << " field='"
            << request.configuration->transport.conservative_phase.liquid_indicator.field_name
            << "' reason=courant_contract"
            << " courant=" << stage.maximum_courant
            << " maximum_courant="
            << request.configuration->transport.conservative_phase.maximum_courant
            << " dt=" << history.dt() << std::endl;
        result.accept_step = false;
        return result;
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase graph stage failed for field '" +
          request.configuration->transport.conservative_phase.liquid_indicator.field_name +
          "': " + stage.diagnostic);
    }
    svmp::FE::Real maximum_nodal_boundary_mass_transfer{0.0};
    bool local_nodal_boundary_flux_finite = true;
    for (const auto transfer : stage.physical_boundary_mass_transfer) {
      if (!std::isfinite(transfer)) {
        local_nodal_boundary_flux_finite = false;
        continue;
      }
      maximum_nodal_boundary_mass_transfer = std::max(
          maximum_nodal_boundary_mass_transfer, std::abs(transfer));
    }
    const auto total_boundary_mass_transfer =
        stage.correction.total_physical_boundary_mass_transfer;
    const auto boundary_mass_tolerance =
        request.configuration->transport.conservative_phase.invariant_tolerance *
        std::max(
            svmp::FE::Real{1.0},
            std::abs(stage.correction.total_previous_liquid_measure));
    const bool local_boundary_flux_satisfied =
        local_nodal_boundary_flux_finite &&
        svmp::FE::level_set::
            levelSetConservativePhaseBoundaryFluxSatisfied(
                request.configuration->transport.conservative_phase.boundary_flux_policy,
                maximum_nodal_boundary_mass_transfer,
                total_boundary_mass_transfer,
                boundary_mass_tolerance);
    if (globalAnyBool(!local_boundary_flux_satisfied, comm)) {
      const bool closed_domain_policy =
          request.configuration->transport.conservative_phase.boundary_flux_policy ==
          svmp::FE::level_set::
              LevelSetConservativePhaseBoundaryFluxPolicy::
                  ClosedDomainDiscreteQFluxOnly;
      throw std::runtime_error(
          std::string{
              "[svMultiPhysics::Application] Conservative phase "} +
          (closed_domain_policy
               ? "closed-domain policy rejected discrete q boundary flux above its invariant tolerance"
               : "globally balanced policy rejected non-finite or nonzero net discrete q boundary flux above its invariant tolerance") +
          " for field '" +
          request.configuration->transport.conservative_phase.liquid_indicator.field_name +
          "'; this is a phase-mass-flux gate, not a pointwise velocity-normal test.");
    }
    const svmp::FE::level_set::LevelSetP1PhaseSplitStageProvenance
        split_stage_provenance{
            .scheme = temporal_stage.scheme,
            .transport_mesh_policy =
                svmp::FE::level_set::LevelSetP1PhaseTransportMeshPolicy::
                    FixedBackground,
            .temporal_order = temporal_stage.temporal_order,
            .prospective_step = temporal_stage.prospective_step,
            .attempt = temporal_stage.attempt,
            .step_start_time = temporal_stage.step_start_time,
            .step_end_time = temporal_stage.step_end_time,
            .q_input_time = temporal_stage.q_input_time,
            .velocity_state_time = temporal_stage.velocity_state_time,
            .time_step = temporal_stage.time_step,
            .operator_state_revision =
                temporal_stage.operator_state_revision,
            .previous_q_revision =
                svmp::FE::level_set::
                    levelSetP1PhaseScalarContentRevision(previous_phase),
            .nodal_velocity_revision =
                svmp::FE::level_set::
                    levelSetP1PhaseVelocityContentRevision(
                        stage.sampled_nodal_velocity),
            .previous_graph_identity =
                fixed_graph_bindings[request_index].identity,
            .operator_graph_identity = staged_request.graph_identity,
            .final_flux_ledger_digest =
                svmp::FE::level_set::
                    levelSetP1PhaseFluxLedgerDigest(stage),
            .stage_options = stage.executed_options,
        };
    svmp::FE::level_set::LevelSetP1PhaseSplitStageValidationResult
        split_stage_validation;
    std::exception_ptr local_split_stage_validation_failure;
    try {
      split_stage_validation =
          svmp::FE::level_set::validateLevelSetP1PhaseSplitStage(
              graph, previous_phase, stage, split_stage_provenance);
    } catch (...) {
      local_split_stage_validation_failure = std::current_exception();
    }
    requireCollectivePhasePreparation(
        local_split_stage_validation_failure,
        comm,
        "conservative_phase_split_stage_validation");
    const bool any_split_stage_valid =
        globalAnyBool(split_stage_validation.valid, comm);
    const bool any_split_stage_invalid =
        globalAnyBool(!split_stage_validation.valid, comm);
    if (any_split_stage_valid && any_split_stage_invalid) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase split-stage validation status differs across the active FE communicator.");
    }
    if (any_split_stage_invalid) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase Backward Euler split-stage provenance validation failed for field '" +
          request.configuration->transport.conservative_phase.liquid_indicator.field_name +
          "': " + split_stage_validation.diagnostic);
    }
    svmp::FE::Real phase_measure{0.0};
    auto& maintenance_ledger =
        result.maintenance_ledgers[request_index];
    std::exception_ptr local_post_stage_materialization_failure;
    try {
      if (stage.correction.nodes.size() != graph.nodes) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase graph stage returned an incomplete nodal ledger.");
      }
      for (std::size_t node = 0u; node < graph.nodes; ++node) {
        raw_phase[node] =
            stage.correction.nodes[node].raw_target_liquid_indicator;
        limited_phase[node] =
            stage.correction.nodes[node].limited_liquid_indicator;
      }
      assignConservativePhaseSlice(
          system, phase_field, raw_phase, raw_transport_candidate);
      assignConservativePhaseSlice(
          system, phase_field, limited_phase, candidate);
      phase_measure = stage.correction.total_limited_liquid_measure;
      const auto measure_tolerance =
          request.configuration->transport.conservative_phase.geometry_measure_tolerance *
          std::max({svmp::FE::Real{1.0}, graph.physical_measure,
                    std::abs(phase_measure)});
      if (!std::isfinite(phase_measure) ||
          phase_measure < -measure_tolerance ||
          phase_measure > graph.physical_measure + measure_tolerance) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase stage produced an invalid liquid measure.");
      }
      accepted_phase_measures[request_index] = phase_measure;
      maintenance_ledger.split_stage_provenance =
          split_stage_validation.provenance;
      maintenance_ledger.maximum_nodal_boundary_mass_transfer =
          maximum_nodal_boundary_mass_transfer;
      maintenance_ledger.boundary_mass_tolerance =
          boundary_mass_tolerance;
      maintenance_ledger.boundary_flux_policy =
          request.configuration->transport.conservative_phase.boundary_flux_policy;
      maintenance_ledger.raw_post_transport_phase_measure =
          stage.correction.total_raw_target_liquid_measure;
      maintenance_ledger.post_limit_phase_measure = phase_measure;
      for (std::size_t node = 0u; node < graph.nodes; ++node) {
        accepted_phase_masses[request_index][node] =
            graph.lumped_control_volume[node] * limited_phase[node];
      }
    } catch (...) {
      local_post_stage_materialization_failure =
          std::current_exception();
    }
    requireCollectivePhasePreparation(
        local_post_stage_materialization_failure,
        comm,
        "conservative_phase_post_stage_materialization");

    std::vector<std::array<svmp::FE::Real, 3>> region_coordinates;
    if (!request.configuration->transport.conservative_phase.fixed_flux_regions.empty()) {
      region_coordinates = conservativePhaseNodeCoordinates(
          system, phase_field, graph);
    }
    std::vector<svmp::FE::level_set::LevelSetPhaseRegionDefinition>
        region_definitions;
    std::exception_ptr local_region_ledger_failure;
    try {
      if (!request.configuration->transport.conservative_phase.fixed_flux_regions.empty()) {
        region_definitions =
            svmp::FE::level_set::makeAxisAlignedBoxPhaseRegions(
                request.configuration->transport.conservative_phase.fixed_flux_regions,
                region_coordinates);
      }
      if (request.configuration->transport.conservative_phase
              .classify_nonprimary_components_as_satellites) {
        auto satellites = svmp::FE::level_set::
            makeNonprimaryComponentSatelliteRegions(stage.correction);
        region_definitions.insert(
            region_definitions.end(),
            std::make_move_iterator(satellites.begin()),
            std::make_move_iterator(satellites.end()));
      }
      maintenance_ledger.region_ledger =
          svmp::FE::level_set::buildLevelSetPhaseRegionLedgers(
              stage.correction,
              region_definitions,
              request.configuration->transport.conservative_phase.invariant_tolerance);
      if (!maintenance_ledger.region_ledger.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase fixed region ledger failed for field '" +
            request.configuration->transport.conservative_phase.liquid_indicator.field_name +
            "': " + maintenance_ledger.region_ledger.diagnostic);
      }

      svmp::FE::rank_log::oopCout()
        << "[svMultiPhysics::Application] Conservative phase staged"
        << " field='"
        << request.configuration->transport.conservative_phase.liquid_indicator.field_name << "'"
        << " step=" << history.stepIndex() + 1
        << " previous_measure="
        << stage.correction.total_previous_liquid_measure
        << " accepted_measure=" << phase_measure
        << " boundary_transfer="
        << stage.correction.total_physical_boundary_mass_transfer
        << " maximum_nodal_boundary_mass_transfer="
        << maximum_nodal_boundary_mass_transfer
        << " boundary_mass_tolerance=" << boundary_mass_tolerance
        << " boundary_flux_policy="
        << svmp::FE::level_set::
               levelSetConservativePhaseBoundaryFluxPolicyName(
                   request.configuration->transport.conservative_phase.boundary_flux_policy)
        << " boundary_safety_scope="
        << svmp::FE::level_set::
               levelSetConservativePhaseBoundaryFluxScope(
                   request.configuration->transport.conservative_phase.boundary_flux_policy)
        << " boundary_safety_limitation=blind_where_q_is_zero"
        << " divergence_source="
        << stage.correction.total_discrete_divergence_mass_source
        << " global_balance_residual="
        << stage.correction.global_mass_balance_residual
        << " max_local_balance_residual="
        << stage.correction.maximum_local_mass_balance_residual
        << " phase_components=" << stage.correction.components.size()
        << " component_activity_tolerance="
        << stage.correction.component_activity_tolerance
        << " subthreshold_component_present="
        << (stage.correction.subthreshold_component_present
                ? "true"
                : "false")
        << " subthreshold_limited_measure="
        << stage.correction.subthreshold_component
               .limited_liquid_measure
        << " max_component_balance_residual="
        << stage.correction.maximum_component_balance_residual
        << " tracked_regions="
        << maintenance_ledger.region_ledger.regions.size()
        << " max_region_balance_residual="
        << maintenance_ledger.region_ledger.maximum_balance_residual
        << " limited_edges=" << stage.correction.limited_edges
        << " courant=" << stage.maximum_courant
        << " temporal_split_scheme="
        << svmp::FE::level_set::levelSetP1PhaseSplitSchemeName(
               split_stage_validation.provenance.scheme)
        << " temporal_order="
        << split_stage_validation.provenance.temporal_order
        << " attempt=" << split_stage_validation.provenance.attempt
        << " q_input_time="
        << split_stage_validation.provenance.q_input_time
        << " velocity_state_time="
        << split_stage_validation.provenance.velocity_state_time
        << " operator_state_revision="
        << split_stage_validation.provenance.operator_state_revision
        << " previous_q_revision="
        << split_stage_validation.provenance.previous_q_revision
        << " nodal_velocity_revision="
        << split_stage_validation.provenance.nodal_velocity_revision
        << " final_flux_ledger_digest="
        << split_stage_validation.provenance.final_flux_ledger_digest
        << " stage_invariant_tolerance="
        << split_stage_validation.provenance
               .stage_options.invariant_tolerance
        << " stage_component_activity_tolerance="
        << split_stage_validation.provenance
               .stage_options.component_activity_tolerance
        << " stage_maximum_courant="
        << split_stage_validation.provenance
               .stage_options.maximum_courant
        << " stage_enforce_courant_limit="
        << (split_stage_validation.provenance
                    .stage_options.enforce_courant_limit
                ? "true"
                : "false")
        << " stage_require_constant_preservation="
        << (split_stage_validation.provenance
                    .stage_options.require_constant_preservation
                ? "true"
                : "false")
          << std::endl;
      maintenance_ledger.transport_stage = std::move(stage);
    } catch (...) {
      local_region_ledger_failure = std::current_exception();
    }
    requireCollectivePhasePreparation(
        local_region_ledger_failure,
        comm,
        "conservative_phase_region_ledger_materialization");
  }

  std::unique_ptr<ConservativePhaseGeometryTransaction> transaction;
  std::exception_ptr local_transaction_allocation_failure;
  try {
    transaction = host.begin_geometry_transaction();
  } catch (...) {
    local_transaction_allocation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_transaction_allocation_failure,
      comm,
      "conservative_phase_geometry_transaction_allocation");
  try {
    const auto raw_transport_refresh_report =
        transaction->refresh(raw_transport_candidate);
    if (observe_stage) {
      observe_stage(
          svmp::FE::level_set::MaintenanceSubstage::Transport,
          result.original_solution,
          raw_transport_candidate);
    }
    const auto post_limit_refresh_report =
        transaction->refresh(candidate);
    if (observe_stage) {
      observe_stage(
          svmp::FE::level_set::MaintenanceSubstage::
              Limiting,
          raw_transport_candidate,
          candidate);
    }
    for (std::size_t request_index = 0u;
         request_index < requests.size(); ++request_index) {
      auto& request = requests[request_index];
      if (!request.configuration->transport.conservative_phase.enabled) {
        continue;
      }
      const auto projection = projectCurrentConservativePhaseGeometry(
          system, request, /*allow_graph_rebuild=*/false);
      if (!projection.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase raw post-transport geometry projection failed for field '" +
            request.configuration->transport.conservative_phase.liquid_indicator.field_name +
            "': " + projection.diagnostic);
      }
      result.maintenance_ledgers[request_index]
          .raw_post_transport_geometry_measure =
          projection.retained_liquid_measure;
    }

    if (contact_stage_builder) {
      result.contact_stage =
          contact_stage_builder(candidate, transaction.get());
    }

    const auto before_reinitialization_candidate = candidate;
    bool any_reinitialization_applied = false;
    for (std::size_t request_index = 0u;
         request_index < requests.size(); ++request_index) {
      auto& request = requests[request_index];
      if (!request.configuration->transport.conservative_phase.enabled) {
        continue;
      }
      auto& maintenance_ledger =
          result.maintenance_ledgers[request_index];
      maintenance_ledger.reinitialization_due =
          svmp::FE::level_set::shouldReinitializeLevelSet(
              request.configuration->transport.reinitialization,
              history.stepIndex() + 1);
      const auto field =
          system.findFieldByName(request.configuration->transport.level_set.field_name);
      if (field == svmp::FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase maintenance could not find level-set field '" +
            request.configuration->transport.level_set.field_name + "'.");
      }
      const auto wall_contact_constraints = host.wall_contact_constraints(
          history,
          field,
          static_cast<svmp::FE::Real>(history.time() + history.dt()),
          result.contact_stage.stages,
          result.contact_stage.constraints);
      auto& graph = requireCurrentConservativePhaseGraph(
          system, request, /*allow_graph_rebuild=*/false);
      contact_protected_nodes[request_index] =
          conservativePhaseContactProtectedNodes(
              system,
              request,
              graph,
              wall_contact_constraints);
      if (!maintenance_ledger.reinitialization_due) {
        continue;
      }
      const auto staged_reinitialization =
          host.stage_projection_reinitialization(
              history,
              request_index,
              field,
              static_cast<svmp::FE::Real>(history.time() + history.dt()),
              candidate,
              result.contact_stage.stages,
              result.contact_stage.constraints,
              result.contact_stage.stage_solution);
      maintenance_ledger.reinitialization =
          staged_reinitialization.repair;
      maintenance_ledger.reinitialization_applied =
          staged_reinitialization.applied;
      any_reinitialization_applied =
          any_reinitialization_applied ||
          staged_reinitialization.applied;
      if (!staged_reinitialization.repair.converged) {
        svmp::FE::rank_log::oopCout()
            << "[svMultiPhysics::Application] Conservative phase candidate rejected"
            << " field='"
            << request.configuration->transport.conservative_phase.liquid_indicator.field_name
            << "' reason=reinitialization_nonconverged"
            << " iterations="
            << staged_reinitialization.repair.iterations
            << " max_iteration_residual="
            << staged_reinitialization.repair.max_iteration_residual
            << " max_signed_distance_error="
            << staged_reinitialization.repair.max_signed_distance_error
            << " max_wall_constrained_signed_distance_error="
            << staged_reinitialization.repair
                   .max_wall_constrained_signed_distance_error
            << " max_interface_displacement="
            << staged_reinitialization.repair.max_interface_displacement
            << " max_contact_line_displacement="
            << staged_reinitialization.repair.max_contact_line_displacement
            << " max_contact_angle_change_radians="
            << staged_reinitialization.repair
                   .max_contact_angle_change_radians
            << " diagnostic='"
            << staged_reinitialization.repair.diagnostic << "'"
            << " dt=" << history.dt() << std::endl;
        transaction->rollback();
        result.accept_step = false;
        return result;
      }
    }

    const auto post_reinitialization_refresh_report =
        transaction->refresh(candidate);
    if (observe_stage && any_reinitialization_applied) {
      observe_stage(
          svmp::FE::level_set::MaintenanceSubstage::
              Reinitialization,
          before_reinitialization_candidate,
          candidate);
    }
    const auto before_reconciliation_candidate = candidate;
    bool any_reconciliation = false;
    for (std::size_t request_index = 0u;
         request_index < requests.size(); ++request_index) {
      auto& request = requests[request_index];
      if (!request.configuration->transport.conservative_phase.enabled) {
        continue;
      }
      const auto projection = projectCurrentConservativePhaseGeometry(
          system, request, /*allow_graph_rebuild=*/false);
      if (!projection.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase post-reinitialization geometry projection failed for field '" +
            request.configuration->transport.conservative_phase.liquid_indicator.field_name +
            "': " + projection.diagnostic);
      }
      auto& maintenance_ledger =
          result.maintenance_ledgers[request_index];
      maintenance_ledger.post_reinitialization_geometry_measure =
          projection.retained_liquid_measure;
      maintenance_ledger.post_reinitialization_mismatch =
          conservativePhaseMomentMismatch(
              projection.liquid_phase_mass,
              accepted_phase_masses[request_index]);
    }

    for (std::size_t request_index = 0u;
         request_index < requests.size(); ++request_index) {
      auto& request = requests[request_index];
      if (!request.configuration->transport.conservative_phase.enabled ||
          !request.configuration->transport.conservative_phase.reconcile_geometry) {
        continue;
      }
      any_reconciliation = true;
      auto& maintenance_ledger =
          result.maintenance_ledgers[request_index];
      auto& graph = requireCurrentConservativePhaseGraph(
          system, request, /*allow_graph_rebuild=*/false);
      const auto projection = projectCurrentConservativePhaseGeometry(
          system, request, /*allow_graph_rebuild=*/false);
      if (!projection.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase stationary-equilibrium projection failed for field '" +
            request.configuration->transport.conservative_phase.liquid_indicator.field_name +
            "': " + projection.diagnostic);
      }
      const auto& transport_nodes =
          maintenance_ledger.transport_stage.correction.nodes;
      if (transport_nodes.size() != graph.nodes ||
          projection.liquid_phase_mass.size() != graph.nodes ||
          projection.liquid_indicator.size() != graph.nodes) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase stationary-equilibrium reconciliation found incompatible transport and geometry layouts.");
      }
      std::vector<svmp::FE::Real> previous_phase_masses(
          graph.nodes, svmp::FE::Real{0.0});
      for (std::size_t node = 0u; node < graph.nodes; ++node) {
        if (transport_nodes[node].node !=
                static_cast<svmp::FE::GlobalIndex>(node) ||
            !std::isfinite(
                transport_nodes[node].previous_liquid_indicator)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Conservative phase stationary-equilibrium reconciliation found a malformed previous-phase ledger.");
        }
        previous_phase_masses[node] =
            graph.lumped_control_volume[node] *
            transport_nodes[node].previous_liquid_indicator;
      }
      const auto stationary_geometry_mismatch =
          conservativePhaseMomentMismatch(
              projection.liquid_phase_mass, previous_phase_masses);
      const auto transported_geometry_mismatch =
          conservativePhaseMomentMismatch(
              projection.liquid_phase_mass,
              accepted_phase_masses[request_index]);
      const auto equilibrium_tolerance =
          request.configuration->transport.conservative_phase.geometry_measure_tolerance *
          std::max(
              {svmp::FE::Real{1.0}, graph.physical_measure,
               std::abs(projection.retained_liquid_measure),
               std::abs(accepted_phase_measures[request_index]),
               std::abs(maintenance_ledger.transport_stage.correction
                            .total_previous_liquid_measure)});
      const bool stationary_geometry =
          stationary_geometry_mismatch.maximum_nodal_residual <=
              equilibrium_tolerance &&
          std::abs(stationary_geometry_mismatch.total_residual) <=
              equilibrium_tolerance;
      const bool transported_total_compatible =
          std::abs(transported_geometry_mismatch.total_residual) <=
              equilibrium_tolerance &&
          std::abs(projection.retained_liquid_measure -
                   accepted_phase_measures[request_index]) <=
              equilibrium_tolerance;
      const bool phase_projection_needed =
          transported_geometry_mismatch.maximum_nodal_residual >
          svmp::FE::Real{0.0};

      ConservativePhaseGeometryReconciliationResult reconciliation;
      // A stationary cut geometry is the authoritative equilibrium for q.
      // Tangential graph fluxes can redistribute projected nodal moments
      // while leaving their total unchanged; moving phi to reproduce that
      // null-mode redistribution would create a spurious normal motion.
      // Restore q from the unchanged geometry and retain the transport stage
      // unchanged as the flux provenance for this accepted step.
      if (stationary_geometry && transported_total_compatible &&
          phase_projection_needed) {
        const auto phase_field = system.findFieldByName(
            request.configuration->transport.conservative_phase.liquid_indicator.field_name);
        if (phase_field == svmp::FE::INVALID_FIELD_ID) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Conservative phase stationary-equilibrium reconciliation could not resolve its phase field.");
        }
        assignConservativePhaseSlice(
            system, phase_field, projection.liquid_indicator, candidate);
        accepted_phase_masses[request_index] =
            projection.liquid_phase_mass;
        accepted_phase_measures[request_index] =
            projection.retained_liquid_measure;
        reconciliation.success = true;
        reconciliation.target_reached = true;
        reconciliation.initial_residual_norm =
            transported_geometry_mismatch.residual_norm;
        reconciliation.final_residual_norm = svmp::FE::Real{0.0};
        reconciliation.maximum_final_nodal_residual =
            svmp::FE::Real{0.0};
        reconciliation.final_total_residual = svmp::FE::Real{0.0};
        reconciliation.diagnostic =
            "stationary_geometry_equilibrium_projection";
        svmp::FE::rank_log::oopCout()
            << "[svMultiPhysics::Application] Conservative phase stationary geometry equilibrium preserved"
            << " diagnostic=" << reconciliation.diagnostic
            << " field='"
            << request.configuration->transport.conservative_phase.liquid_indicator.field_name
            << "' initial_local_residual_norm="
            << transported_geometry_mismatch.residual_norm
            << " maximum_initial_nodal_residual="
            << transported_geometry_mismatch.maximum_nodal_residual
            << " initial_total_residual="
            << transported_geometry_mismatch.total_residual
            << " stationary_geometry_maximum_nodal_residual="
            << stationary_geometry_mismatch.maximum_nodal_residual
            << " stationary_geometry_total_residual="
            << stationary_geometry_mismatch.total_residual
            << " tolerance=" << equilibrium_tolerance << std::endl;
      } else {
        reconciliation = reconcileConservativePhaseGeometry(
            host,
            request,
            accepted_phase_masses[request_index],
            candidate,
            *transaction,
            contact_protected_nodes[request_index]);
      }
      reconciliation_reports[request_index] = reconciliation;
      maintenance_ledger.reconciliation = reconciliation;
      if (!reconciliation.success || !reconciliation.target_reached) {
        svmp::FE::rank_log::oopCout()
            << "[svMultiPhysics::Application] Conservative phase candidate rejected"
            << " field='"
            << request.configuration->transport.conservative_phase.liquid_indicator.field_name
            << "' reason=local_geometry_reconciliation_contract"
            << " iterations=" << reconciliation.iterations
            << " line_search_evaluations="
            << reconciliation.line_search_evaluations
            << " geometry_refresh_requests="
            << reconciliation.geometry_refresh_requests
            << " geometry_rebuilds="
            << reconciliation.geometry_rebuilds
            << " rejected_geometry_trials="
            << reconciliation.rejected_geometry_trials
            << " contact_protected_nodes="
            << reconciliation.contact_protected_nodes
            << " maximum_removed_contact_increment="
            << reconciliation.maximum_removed_contact_increment
            << " initial_local_residual_norm="
            << reconciliation.initial_residual_norm
            << " final_local_residual_norm="
            << reconciliation.final_residual_norm
            << " max_final_nodal_residual="
            << reconciliation.maximum_final_nodal_residual
            << " total_final_residual="
            << reconciliation.final_total_residual
            << " accumulated_interface_displacement_bound="
            << reconciliation.accumulated_interface_displacement_bound
            << " allowed_interface_displacement="
            << reconciliation.allowed_interface_displacement
            << " limited_by_displacement="
            << (reconciliation.limited_by_displacement ? "true" : "false")
            << " limited_by_topology="
            << (reconciliation.limited_by_topology ? "true" : "false")
            << " diagnostic='" << reconciliation.diagnostic << "'"
            << " dt=" << history.dt() << std::endl;
        transaction->rollback();
        result.accept_step = false;
        return result;
      }
    }
    if (observe_stage && any_reconciliation) {
      observe_stage(
          svmp::FE::level_set::MaintenanceSubstage::
              GeometryReconciliation,
          before_reconciliation_candidate,
          candidate);
    }
    for (std::size_t request_index = 0u;
         request_index < requests.size(); ++request_index) {
      auto& request = requests[request_index];
      if (!request.configuration->transport.conservative_phase.enabled) {
        continue;
      }
      const auto projection = projectCurrentConservativePhaseGeometry(
          system, request, /*allow_graph_rebuild=*/false);
      if (!projection.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase authoritative geometry projection failed for field '" +
            request.configuration->transport.conservative_phase.liquid_indicator.field_name +
            "': " + projection.diagnostic);
      }
      const auto phase_measure = accepted_phase_measures[request_index];
      const auto mismatch = std::abs(
          projection.retained_liquid_measure - phase_measure);
      const auto tolerance =
          request.configuration->transport.conservative_phase.geometry_measure_tolerance *
          std::max({svmp::FE::Real{1.0},
                    std::abs(projection.retained_liquid_measure),
                    std::abs(phase_measure)});
      const auto local_mismatch = conservativePhaseMomentMismatch(
          projection.liquid_phase_mass,
          accepted_phase_masses[request_index]);
      auto& maintenance_ledger =
          result.maintenance_ledgers[request_index];
      maintenance_ledger.post_correction_phase_measure = phase_measure;
      maintenance_ledger.post_correction_geometry_measure =
          projection.retained_liquid_measure;
      maintenance_ledger.post_correction_mismatch = local_mismatch;
      if (request.configuration->transport.conservative_phase.reconcile_geometry &&
          (mismatch > tolerance ||
           local_mismatch.maximum_nodal_residual > tolerance ||
           std::abs(local_mismatch.total_residual) > tolerance)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase local geometry reconciliation did not match the transported nodal moments for field '" +
            request.configuration->transport.conservative_phase.liquid_indicator.field_name +
            "'.");
      }
      const auto& reconciliation =
          reconciliation_reports[request_index];
      svmp::FE::rank_log::oopCout()
          << "[svMultiPhysics::Application] Conservative phase geometry validated"
          << " field='"
          << request.configuration->transport.conservative_phase.liquid_indicator.field_name << "'"
          << " step=" << history.stepIndex() + 1
          << " phase_measure=" << phase_measure
          << " retained_geometry_measure="
          << projection.retained_liquid_measure
          << " measure_mismatch=" << mismatch
          << " max_nodal_moment_mismatch="
          << local_mismatch.maximum_nodal_residual
          << " nodal_moment_residual_norm="
          << local_mismatch.residual_norm
          << " reconciliation_iterations="
          << reconciliation.iterations
          << " reconciliation_line_search_evaluations="
          << reconciliation.line_search_evaluations
          << " reconciliation_geometry_refresh_requests="
          << reconciliation.geometry_refresh_requests
          << " reconciliation_geometry_rebuilds="
          << reconciliation.geometry_rebuilds
          << " reconciliation_rejected_geometry_trials="
          << reconciliation.rejected_geometry_trials
          << " reconciliation_initial_residual_norm="
          << reconciliation.initial_residual_norm
          << " reconciliation_final_residual_norm="
          << reconciliation.final_residual_norm
          << " reconciliation_diagnostic='"
          << reconciliation.diagnostic << "'"
          << " interface_displacement_bound="
          << reconciliation.accumulated_interface_displacement_bound
          << " interface_displacement_limit="
          << reconciliation.allowed_interface_displacement
          << " tolerance=" << tolerance
          << " reconciliation_enabled="
          << (request.configuration->transport.conservative_phase.reconcile_geometry ? "true"
                                                             : "false")
          << " cut_context_revision="
          << projection.cut_context_revision
          << " cut_context_refreshed="
          << (raw_transport_refresh_report.refreshed ||
                      post_limit_refresh_report.refreshed ||
                      post_reinitialization_refresh_report.refreshed ||
                      reconciliation.geometry_rebuilds > 0
                  ? "true"
                  : "false")
          << std::endl;
      const auto& repair = maintenance_ledger.reinitialization;
      const auto& transport = maintenance_ledger.transport_stage;
      if (!maintenance_ledger.split_stage_provenance.has_value()) {
        throw std::logic_error(
            "[svMultiPhysics::Application] Conservative phase maintenance ledger is missing its validated temporal split-stage provenance.");
      }
      const auto& split_stage =
          *maintenance_ledger.split_stage_provenance;
      svmp::FE::Real ledger_maximum_nodal_boundary_transfer{0.0};
      for (const auto transfer :
           transport.physical_boundary_mass_transfer) {
        ledger_maximum_nodal_boundary_transfer = std::max(
            ledger_maximum_nodal_boundary_transfer,
            std::abs(transfer));
      }
      svmp::FE::rank_log::oopCout()
          << std::setprecision(17)
          << "[svMultiPhysics::Application] Conservative phase maintenance ledger"
          << " diagnostic=conservative_phase_maintenance_ledger"
          << " field='"
          << request.configuration->transport.conservative_phase.liquid_indicator.field_name << "'"
          << " step=" << history.stepIndex() + 1
          << " raw_post_transport_phase_measure="
          << maintenance_ledger.raw_post_transport_phase_measure
          << " post_limit_phase_measure="
          << maintenance_ledger.post_limit_phase_measure
          << " raw_post_transport_geometry_measure="
          << maintenance_ledger.raw_post_transport_geometry_measure
          << " post_reinitialization_phase_measure="
          << maintenance_ledger.post_limit_phase_measure
          << " post_reinitialization_geometry_measure="
          << maintenance_ledger.post_reinitialization_geometry_measure
          << " post_reinitialization_max_nodal_mismatch="
          << maintenance_ledger.post_reinitialization_mismatch
                 .maximum_nodal_residual
          << " post_correction_phase_measure="
          << maintenance_ledger.post_correction_phase_measure
          << " post_correction_geometry_measure="
          << maintenance_ledger.post_correction_geometry_measure
          << " post_correction_max_nodal_mismatch="
          << maintenance_ledger.post_correction_mismatch
                 .maximum_nodal_residual
          << " retained_assembly_measure="
          << projection.retained_liquid_measure
          << " temporal_split_scheme="
          << svmp::FE::level_set::levelSetP1PhaseSplitSchemeName(
                 split_stage.scheme)
          << " temporal_order=" << split_stage.temporal_order
          << " prospective_step=" << split_stage.prospective_step
          << " attempt=" << split_stage.attempt
          << " step_start_time=" << split_stage.step_start_time
          << " step_end_time=" << split_stage.step_end_time
          << " q_input_time=" << split_stage.q_input_time
          << " velocity_state_time=" << split_stage.velocity_state_time
          << " operator_state_revision="
          << split_stage.operator_state_revision
          << " previous_q_revision="
          << split_stage.previous_q_revision
          << " nodal_velocity_revision="
          << split_stage.nodal_velocity_revision
          << " transport_graph_content_revision="
          << split_stage.operator_graph_identity.content_revision
          << " final_flux_ledger_digest="
          << split_stage.final_flux_ledger_digest
          << " stage_invariant_tolerance="
          << split_stage.stage_options.invariant_tolerance
          << " stage_component_activity_tolerance="
          << split_stage.stage_options.component_activity_tolerance
          << " stage_maximum_courant="
          << split_stage.stage_options.maximum_courant
          << " stage_enforce_courant_limit="
          << (split_stage.stage_options.enforce_courant_limit
                  ? "true"
                  : "false")
          << " stage_require_constant_preservation="
          << (split_stage.stage_options.require_constant_preservation
                  ? "true"
                  : "false")
          << " total_physical_boundary_mass_transfer="
          << transport.correction
                 .total_physical_boundary_mass_transfer
          << " maximum_nodal_physical_boundary_mass_transfer="
          << ledger_maximum_nodal_boundary_transfer
          << " boundary_mass_tolerance="
          << maintenance_ledger.boundary_mass_tolerance
          << " boundary_flux_policy="
          << svmp::FE::level_set::
                 levelSetConservativePhaseBoundaryFluxPolicyName(
                     maintenance_ledger.boundary_flux_policy)
          << " boundary_safety_scope="
          << svmp::FE::level_set::
                 levelSetConservativePhaseBoundaryFluxScope(
                     maintenance_ledger.boundary_flux_policy)
          << " boundary_safety_limitation=blind_where_q_is_zero"
          << " transport_nodes="
          << transport.correction.nodes.size()
          << " transport_edges="
          << transport.correction.edges.size()
          << " transport_components="
          << transport.correction.components.size()
          << " transport_component_activity_tolerance="
          << transport.correction.component_activity_tolerance
          << " transport_subthreshold_component_present="
          << (transport.correction.subthreshold_component_present
                  ? "true"
                  : "false")
          << " transport_subthreshold_nodes="
          << transport.correction.subthreshold_component.nodes
          << " transport_subthreshold_limited_measure="
          << transport.correction.subthreshold_component
                 .limited_liquid_measure
          << " transport_component_balance_satisfied="
          << (transport.correction.component_balance_satisfied
                  ? "true"
                  : "false")
          << " transport_component_measure_closure_satisfied="
          << (transport.correction.component_measure_closure_satisfied
                  ? "true"
                  : "false")
          << " transport_max_component_balance_residual="
          << transport.correction.maximum_component_balance_residual
          << " transport_limited_component_closure_residual="
          << transport.correction
                 .limited_component_measure_closure_residual
          << " transport_limited_component_transfer_closure_residual="
          << transport.correction
                 .limited_component_transfer_closure_residual
          << " reinitialization_due="
          << (maintenance_ledger.reinitialization_due ? "true" : "false")
          << " reinitialization_applied="
          << (maintenance_ledger.reinitialization_applied ? "true"
                                                           : "false")
          << " reinitialization_converged="
          << (repair.converged ? "true" : "false")
          << " reinitialization_iterations=" << repair.iterations
          << " reinitialization_max_abs_update="
          << repair.max_abs_update
          << " reinitialization_max_interface_displacement="
          << repair.max_interface_displacement
          << " reinitialization_wall_contact_constraints="
          << repair.wall_contact_constraints
          << " reinitialization_max_contact_line_displacement="
          << repair.max_contact_line_displacement
          << " reinitialization_max_contact_angle_change_radians="
          << repair.max_contact_angle_change_radians
          << " reconciliation_iterations="
          << reconciliation.iterations
          << " reconciliation_interface_displacement_bound="
          << reconciliation.accumulated_interface_displacement_bound
          << " reconciliation_contact_protected_nodes="
          << reconciliation.contact_protected_nodes
          << " reconciliation_maximum_removed_contact_increment="
          << reconciliation.maximum_removed_contact_increment
          << " reconciliation_initial_residual_norm="
          << reconciliation.initial_residual_norm
          << " reconciliation_final_residual_norm="
          << reconciliation.final_residual_norm
          << " reconciliation_diagnostic='"
          << reconciliation.diagnostic << "'"
          << std::endl;
      const auto emit_component_ledger =
          [&](const auto& component, std::string_view classification) {
        svmp::FE::rank_log::oopCout()
            << std::setprecision(17)
            << "[svMultiPhysics::Application] Conservative phase component ledger"
            << " diagnostic=conservative_phase_component_ledger"
            << " field='"
            << request.configuration->transport.conservative_phase.liquid_indicator.field_name
            << "'"
            << " step=" << history.stepIndex() + 1
            << " classification=" << classification
            << " component_id=" << component.component_id
            << " nodes=" << component.nodes
            << " previous_liquid_measure="
            << component.previous_liquid_measure
            << " low_order_liquid_measure="
            << component.low_order_liquid_measure
            << " raw_target_liquid_measure="
            << component.raw_target_liquid_measure
            << " limited_liquid_measure="
            << component.limited_liquid_measure
            << " physical_boundary_mass_transfer="
            << component.physical_boundary_mass_transfer
            << " discrete_divergence_mass_source="
            << component.discrete_divergence_mass_source
            << " low_order_interior_mass_transfer="
            << component.low_order_interior_mass_transfer
            << " raw_antidiffusive_mass_transfer="
            << component.raw_antidiffusive_mass_transfer
            << " limited_antidiffusive_mass_transfer="
            << component.limited_antidiffusive_mass_transfer
            << " low_order_balance_residual="
            << component.low_order_balance_residual
            << " raw_target_balance_residual="
            << component.raw_target_balance_residual
            << " limited_balance_residual="
            << component.limited_balance_residual
            << std::endl;
      };
      for (const auto& component : transport.correction.components) {
        emit_component_ledger(component, "resolved");
      }
      if (transport.correction.subthreshold_component_present) {
        emit_component_ledger(
            transport.correction.subthreshold_component,
            "subthreshold");
      }
      for (const auto& region : maintenance_ledger.region_ledger.regions) {
        svmp::FE::rank_log::oopCout()
            << std::setprecision(17)
            << "[svMultiPhysics::Application] Conservative phase region ledger"
            << " diagnostic=conservative_phase_region_ledger"
            << " field='"
            << request.configuration->transport.conservative_phase.liquid_indicator.field_name
            << "'"
            << " step=" << history.stepIndex() + 1
            << " region='" << region.name << "'"
            << " kind="
            << svmp::FE::level_set::levelSetPhaseRegionKindName(
                   region.kind)
            << " nodes=" << region.member_nodes.size()
            << " internal_edges=" << region.internal_edges
            << " crossing_edges=" << region.crossing_edges.size()
            << " previous_liquid_measure="
            << region.previous_liquid_measure
            << " low_order_liquid_measure="
            << region.low_order_liquid_measure
            << " raw_target_liquid_measure="
            << region.raw_target_liquid_measure
            << " limited_liquid_measure="
            << region.limited_liquid_measure
            << " physical_boundary_mass_transfer="
            << region.physical_boundary_mass_transfer
            << " discrete_divergence_mass_source="
            << region.discrete_divergence_mass_source
            << " low_order_crossing_mass_transfer="
            << region.low_order_crossing_mass_transfer
            << " raw_crossing_antidiffusive_mass_transfer="
            << region.raw_crossing_antidiffusive_mass_transfer
            << " limited_crossing_antidiffusive_mass_transfer="
            << region.limited_crossing_antidiffusive_mass_transfer
            << " low_order_balance_residual="
            << region.low_order_balance_residual
            << " raw_target_balance_residual="
            << region.raw_target_balance_residual
            << " limited_balance_residual="
            << region.limited_balance_residual
            << " maximum_internal_pair_cancellation_residual="
            << region.maximum_internal_pair_cancellation_residual
            << std::endl;
      }
    }
    result.changed = candidate != result.original_solution;
    if (result.changed) {
      scatterFeOrderedSolution(history.u(), candidate);
      history.updateGhosts();
    }
    result.geometry_transaction = std::move(transaction);
  } catch (...) {
    transaction->rollback();
    throw;
  }
  return result;
}

void rollbackConservativePhaseCandidate(
    svmp::FE::timestepping::TimeHistory& history,
    ConservativePhaseCandidateResult& result)
{
  std::exception_ptr first_failure;
  if (result.geometry_transaction) {
    try {
      result.geometry_transaction->rollback();
      if (!result.geometry_transaction->active()) {
        result.geometry_transaction.reset();
      }
    } catch (...) {
      first_failure = std::current_exception();
    }
  }
  if (result.changed && !result.original_solution.empty()) {
    try {
      scatterFeOrderedSolution(
          history.u(), result.original_solution);
      history.updateGhosts();
      result.changed = false;
    } catch (...) {
      if (!first_failure) {
        first_failure = std::current_exception();
      }
    }
  } else {
    result.changed = false;
  }
  if (first_failure) {
    std::rethrow_exception(first_failure);
  }
}

} // namespace level_set
} // namespace FE
} // namespace svmp

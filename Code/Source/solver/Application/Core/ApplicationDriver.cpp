#include "Application/Core/ApplicationDriver.h"

#include "Application/Core/ActiveDomainOutput.h"
#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/LevelSetCurvatureSamples.h"
#include "Application/Core/LevelSetMaintenanceHistory.h"
#include "Application/Core/LevelSetVelocityExtensionMap.h"
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
#include "FE/LevelSet/LevelSetCellEvaluator.h"
#include "FE/LevelSet/LevelSetConservativePhaseOperator.h"
#include "FE/LevelSet/LevelSetConservativePhaseArtifact.h"
#include "FE/LevelSet/LevelSetConservativePhaseRegions.h"
#include "FE/LevelSet/LevelSetConservativePhaseState.h"
#include "FE/LevelSet/LevelSetImplicitCutQuadratureBackend.h"
#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"
#include "FE/LevelSet/LevelSetReinitialization.h"
#include "FE/LevelSet/LevelSetTransport.h"
#include "FE/LevelSet/LevelSetVelocityExtensionConstraint.h"
#include "FE/LevelSet/LevelSetVolume.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/Interfaces/GeneratedActiveBoundaryDomain.h"
#include "FE/Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "FE/Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "FE/PostProcessing/DerivedResultTypes.h"
#include "FE/PostProcessing/DerivedResultEvaluator.h"
#include "FE/Systems/CutIntegrationInvalidation.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/TimeStepping/NewtonSolver.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"
#include "FE/TimeStepping/TimeSteppingUtils.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Core/MeshComm.h"
#include "Mesh/Topology/CellTopology.h"
#include "Mesh/Topology/DistributedTopology.h"
#include "Physics/Core/PhysicsModule.h"
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
#include <exception>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <iterator>
#include <limits>
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
#include <unordered_map>
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
namespace {

void mixMaintenanceRevision(
    std::uint64_t& revision,
    std::uint64_t value) noexcept
{
  constexpr std::uint64_t prime = 1099511628211ull;
  for (std::size_t byte = 0u; byte < sizeof(value); ++byte) {
    revision ^= (value >> (byte * 8u)) & 0xffu;
    revision *= prime;
  }
}

enum class MaintenanceRevisionKind : std::uint8_t {
  Snapshot,
  MeshTopology,
  CutTopology
};

std::uint64_t maintenanceRevisionSet(
    const std::vector<LevelSetAuthoritativeFunctionalValue>& values,
    MaintenanceRevisionKind kind) noexcept
{
  if (values.empty()) {
    return 0u;
  }
  std::uint64_t revision = 1469598103934665603ull;
  for (const auto& value : values) {
    mixMaintenanceRevision(
        revision, static_cast<std::uint64_t>(value.interface_marker));
    switch (kind) {
      case MaintenanceRevisionKind::Snapshot:
        mixMaintenanceRevision(revision, value.snapshot_revision);
        break;
      case MaintenanceRevisionKind::MeshTopology:
        mixMaintenanceRevision(revision, value.mesh_topology_revision);
        break;
      case MaintenanceRevisionKind::CutTopology:
        mixMaintenanceRevision(revision, value.cut_topology_revision);
        break;
    }
  }
  return revision == 0u ? 1u : revision;
}

void validateMaintenanceFunctionalValues(
    const std::vector<LevelSetAuthoritativeFunctionalValue>& values)
{
  int previous_marker = -1;
  for (const auto& value : values) {
    const std::array<double, 8> scalars{
        value.liquid_volume,
        value.liquid_gas_area,
        value.wetted_wall_area,
        value.contact_measure,
        value.surface_energy,
        value.young_wall_energy,
        value.volume_constraint_potential,
        value.total_potential};
    if (value.interface_marker < 0 ||
        value.interface_marker <= previous_marker ||
        value.snapshot_revision == 0u ||
        value.cut_topology_revision == 0u ||
        !std::all_of(scalars.begin(), scalars.end(), [](double scalar) {
          return std::isfinite(scalar);
        })) {
      throw std::invalid_argument(
          "Level-set maintenance work values require unique ordered markers, "
          "complete snapshot and cut-topology revisions, and finite scalars.");
    }
    previous_marker = value.interface_marker;
  }
}

LevelSetMaintenanceWorkAttempt makeMaintenanceWorkAttempt(
    const LevelSetMaintenanceWorkTransaction& transaction,
    LevelSetMaintenanceWorkStatus status,
    const std::vector<LevelSetMaintenanceWorkRow>& rows) noexcept
{
  double numerical_work = 0.0;
  for (const auto& row : rows) {
    numerical_work += row.numerical_work;
  }
  return LevelSetMaintenanceWorkAttempt{
      .transaction_id = transaction.transaction_id,
      .status = status,
      .step = transaction.step,
      .attempt = transaction.attempt,
      .time = transaction.time,
      .dt = transaction.dt,
      .declared_stage = transaction.declared_stage,
      .extension_map_revision = transaction.extension_map_revision,
      .row_count = rows.size(),
      .numerical_work = numerical_work,
      .accepted_numerical_work =
          status == LevelSetMaintenanceWorkStatus::Accepted
              ? numerical_work
              : 0.0,
  };
}

} // namespace

void LevelSetMaintenanceWorkLedger::beginTransaction(
    LevelSetMaintenanceWorkTransaction transaction)
{
  if (active_transaction_.has_value()) {
    throw std::logic_error(
        "A level-set maintenance work transaction is already active.");
  }
  if (!trial_rows_.empty()) {
    throw std::logic_error(
        "A level-set maintenance work transaction has unpublished trial rows.");
  }
  if (transaction.transaction_id == 0u || transaction.attempt == 0u ||
      !std::isfinite(transaction.time) ||
      !std::isfinite(transaction.dt) || transaction.dt < 0.0 ||
      (transaction.extension_map_revision.has_value() &&
       *transaction.extension_map_revision == 0u)) {
    throw std::invalid_argument(
        "Level-set maintenance work transaction metadata is incomplete.");
  }
  if (transaction.transaction_id <= last_transaction_id_) {
    throw std::invalid_argument(
        "Level-set maintenance work transaction identifiers must increase.");
  }
  // Reserve both possible publication destinations before making the
  // transaction active. Commit and rejection can then publish their outcome
  // without allocating after geometry has been committed or rolled back.
  accepted_attempts_.reserve(accepted_attempts_.size() + 1u);
  rejected_attempts_.reserve(rejected_attempts_.size() + 1u);
  last_transaction_id_ = transaction.transaction_id;
  active_transaction_ = std::move(transaction);
}

void LevelSetMaintenanceWorkLedger::stageRow(
    LevelSetMaintenanceWorkSubstage substage,
    std::uint64_t algebraic_state_revision_before,
    std::uint64_t algebraic_state_revision_after,
    std::vector<LevelSetAuthoritativeFunctionalValue> before,
    std::vector<LevelSetAuthoritativeFunctionalValue> after,
    std::optional<std::uint64_t> extension_map_revision_after)
{
  if (!active_transaction_.has_value()) {
    throw std::logic_error(
        "Level-set maintenance work rows require an active transaction.");
  }
  if (algebraic_state_revision_before == 0u ||
      algebraic_state_revision_after == 0u) {
    throw std::invalid_argument(
        "Level-set maintenance work rows require complete algebraic state revisions.");
  }
  if (extension_map_revision_after.has_value() &&
      *extension_map_revision_after == 0u) {
    throw std::invalid_argument(
        "Level-set maintenance work rows require a nonzero extension-map revision when present.");
  }
  const auto by_marker = [](const auto& left, const auto& right) {
    return left.interface_marker < right.interface_marker;
  };
  std::sort(before.begin(), before.end(), by_marker);
  std::sort(after.begin(), after.end(), by_marker);
  validateMaintenanceFunctionalValues(before);
  validateMaintenanceFunctionalValues(after);
  if (before.size() != after.size()) {
    throw std::invalid_argument(
        "Level-set maintenance work rows require matching functional coverage.");
  }

  double numerical_work = 0.0;
  for (std::size_t index = 0u; index < before.size(); ++index) {
    if (before[index].interface_marker != after[index].interface_marker) {
      throw std::invalid_argument(
          "Level-set maintenance work rows require matching functional markers.");
    }
    numerical_work +=
        after[index].total_potential - before[index].total_potential;
  }
  if (!std::isfinite(numerical_work)) {
    throw std::invalid_argument(
        "Level-set maintenance numerical work is not finite.");
  }

  const auto& transaction = *active_transaction_;
  const auto extension_map_revision_before =
      trial_rows_.empty()
          ? transaction.extension_map_revision
          : trial_rows_.back().extension_map_revision_after;
  if (!trial_rows_.empty()) {
    const auto& previous = trial_rows_.back();
    if (previous.algebraic_state_revision_after !=
            algebraic_state_revision_before ||
        previous.after != before) {
      throw std::invalid_argument(
          "Level-set maintenance work rows must form one continuous algebraic and functional state chain.");
    }
  }
  const auto resolved_extension_map_revision_after =
      extension_map_revision_after.has_value()
          ? extension_map_revision_after
          : extension_map_revision_before;

  // Preflight both terminal row destinations while candidate geometry is
  // still rollback-capable. Neither commitTransaction nor rejectTransaction
  // then needs to allocate after the physical outcome is known.
  const auto terminal_row_count =
      accepted_rows_.size() + trial_rows_.size() + 1u;
  accepted_rows_.reserve(terminal_row_count);
  rejected_rows_.reserve(
      rejected_rows_.size() + trial_rows_.size() + 1u);
  trial_rows_.reserve(trial_rows_.size() + 1u);
  trial_rows_.push_back(LevelSetMaintenanceWorkRow{
      .transaction_id = transaction.transaction_id,
      .status = LevelSetMaintenanceWorkStatus::Trial,
      .substage = substage,
      .step = transaction.step,
      .attempt = transaction.attempt,
      .time = transaction.time,
      .dt = transaction.dt,
      .algebraic_state_revision_before =
          algebraic_state_revision_before,
      .algebraic_state_revision_after =
          algebraic_state_revision_after,
      .snapshot_set_revision_before =
          maintenanceRevisionSet(
              before, MaintenanceRevisionKind::Snapshot),
      .snapshot_set_revision_after =
          maintenanceRevisionSet(
              after, MaintenanceRevisionKind::Snapshot),
      .mesh_topology_set_revision_before =
          maintenanceRevisionSet(
              before, MaintenanceRevisionKind::MeshTopology),
      .mesh_topology_set_revision_after =
          maintenanceRevisionSet(
              after, MaintenanceRevisionKind::MeshTopology),
      .cut_topology_set_revision_before =
          maintenanceRevisionSet(
              before, MaintenanceRevisionKind::CutTopology),
      .cut_topology_set_revision_after =
          maintenanceRevisionSet(
              after, MaintenanceRevisionKind::CutTopology),
      .extension_map_revision_before =
          extension_map_revision_before,
      .extension_map_revision_after =
          resolved_extension_map_revision_after,
      .declared_stage = transaction.declared_stage,
      .before = std::move(before),
      .after = std::move(after),
      .numerical_work = numerical_work,
      // Trial and rejected rows never contribute to the accepted-state
      // account. This channel is a maintenance term only, not a complete
      // time-discrete energy identity.
      .accepted_numerical_work = 0.0,
  });
}

void LevelSetMaintenanceWorkLedger::commitTransaction()
{
  if (!active_transaction_.has_value()) {
    throw std::logic_error(
        "No level-set maintenance work transaction is active.");
  }
  if (accepted_attempts_.capacity() < accepted_attempts_.size() + 1u ||
      accepted_rows_.capacity() <
          accepted_rows_.size() + trial_rows_.size()) {
    throw std::logic_error(
        "Level-set maintenance work commit was not allocation-preflighted.");
  }
  const auto attempt = makeMaintenanceWorkAttempt(
      *active_transaction_,
      LevelSetMaintenanceWorkStatus::Accepted,
      trial_rows_);
  for (auto& row : trial_rows_) {
    row.status = LevelSetMaintenanceWorkStatus::Accepted;
    row.accepted_numerical_work = row.numerical_work;
    accepted_rows_.push_back(std::move(row));
  }
  accepted_attempts_.push_back(attempt);
  trial_rows_.clear();
  active_transaction_.reset();
}

void LevelSetMaintenanceWorkLedger::rejectTransaction()
{
  if (!active_transaction_.has_value()) {
    throw std::logic_error(
        "No level-set maintenance work transaction is active.");
  }
  if (rejected_attempts_.capacity() < rejected_attempts_.size() + 1u ||
      rejected_rows_.capacity() <
          rejected_rows_.size() + trial_rows_.size()) {
    throw std::logic_error(
        "Level-set maintenance work rejection was not allocation-preflighted.");
  }
  const auto attempt = makeMaintenanceWorkAttempt(
      *active_transaction_,
      LevelSetMaintenanceWorkStatus::Rejected,
      trial_rows_);
  for (auto& row : trial_rows_) {
    row.status = LevelSetMaintenanceWorkStatus::Rejected;
    row.accepted_numerical_work = 0.0;
    rejected_rows_.push_back(std::move(row));
  }
  rejected_attempts_.push_back(attempt);
  trial_rows_.clear();
  active_transaction_.reset();
}

bool LevelSetMaintenanceWorkLedger::transactionActive() const noexcept
{
  return active_transaction_.has_value();
}

const std::vector<LevelSetMaintenanceWorkRow>&
LevelSetMaintenanceWorkLedger::trialRows() const noexcept
{
  return trial_rows_;
}

const std::vector<LevelSetMaintenanceWorkRow>&
LevelSetMaintenanceWorkLedger::acceptedRows() const noexcept
{
  return accepted_rows_;
}

const std::vector<LevelSetMaintenanceWorkRow>&
LevelSetMaintenanceWorkLedger::rejectedRows() const noexcept
{
  return rejected_rows_;
}

const std::vector<LevelSetMaintenanceWorkAttempt>&
LevelSetMaintenanceWorkLedger::acceptedAttempts() const noexcept
{
  return accepted_attempts_;
}

const std::vector<LevelSetMaintenanceWorkAttempt>&
LevelSetMaintenanceWorkLedger::rejectedAttempts() const noexcept
{
  return rejected_attempts_;
}

} // namespace application::core

namespace {

using Clock = std::chrono::steady_clock;
using application::core::ActiveCutVolumeRequest;
using application::core::LevelSetActiveSide;
using application::core::NearestPointIndex;
using application::core::NearestPointRecord;
using application::core::activeCutVolumeRequests;

struct VelocityExtensionSample {
  std::array<double, 3> point{0.0, 0.0, 0.0};
  std::vector<double> value{};
};

double secondsSince(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
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
    output << "]}\n";
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

  application::core::oopCout()
      << "[svMultiPhysics::Application] Wrote effective configuration: "
      << output_path.string() << std::endl;
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

std::vector<VelocityExtensionSample>
gatherVelocityExtensionSamples(std::vector<VelocityExtensionSample> local_samples,
                               std::size_t component_count,
                               const svmp::MeshComm& comm)
{
  if (component_count == 0u) {
    return local_samples;
  }

#ifdef MESH_HAS_MPI
  if (!comm.is_parallel()) {
    return local_samples;
  }

  if (component_count >
      static_cast<std::size_t>(std::numeric_limits<int>::max() - 3)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Too many velocity-extension components for MPI gather.");
  }
  if (local_samples.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Too many local velocity-extension samples for MPI gather.");
  }

  const int width = static_cast<int>(component_count + 3u);
  const int local_count = static_cast<int>(local_samples.size());
  std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
  MPI_Allgather(&local_count, 1, MPI_INT,
                counts.data(), 1, MPI_INT,
                comm.native());

  std::vector<int> displs(counts.size(), 0);
  int total_count = 0;
  for (std::size_t r = 0; r < counts.size(); ++r) {
    if (counts[r] < 0 ||
        counts[r] > (std::numeric_limits<int>::max() / width)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Invalid velocity-extension sample count in MPI gather.");
    }
    displs[r] = total_count * width;
    total_count += counts[r];
  }

  std::vector<double> send(
      static_cast<std::size_t>(local_count) * static_cast<std::size_t>(width),
      0.0);
  for (std::size_t i = 0; i < local_samples.size(); ++i) {
    const auto base = i * static_cast<std::size_t>(width);
    send[base + 0u] = local_samples[i].point[0];
    send[base + 1u] = local_samples[i].point[1];
    send[base + 2u] = local_samples[i].point[2];
    for (std::size_t c = 0; c < component_count; ++c) {
      const auto value =
          c < local_samples[i].value.size() ? local_samples[i].value[c] : 0.0;
      send[base + 3u + c] = std::isfinite(value) ? value : 0.0;
    }
  }

  std::vector<int> recv_counts(counts.size(), 0);
  for (std::size_t r = 0; r < counts.size(); ++r) {
    recv_counts[r] = counts[r] * width;
  }
  std::vector<double> recv(
      static_cast<std::size_t>(total_count) * static_cast<std::size_t>(width),
      0.0);
  MPI_Allgatherv(send.empty() ? nullptr : send.data(),
                 local_count * width,
                 MPI_DOUBLE,
                 recv.empty() ? nullptr : recv.data(),
                 recv_counts.data(),
                 displs.data(),
                 MPI_DOUBLE,
                 comm.native());

  std::vector<VelocityExtensionSample> gathered;
  gathered.reserve(static_cast<std::size_t>(total_count));
  for (int i = 0; i < total_count; ++i) {
    const auto base =
        static_cast<std::size_t>(i) * static_cast<std::size_t>(width);
    VelocityExtensionSample sample;
    sample.point = {{recv[base + 0u], recv[base + 1u], recv[base + 2u]}};
    sample.value.assign(component_count, 0.0);
    for (std::size_t c = 0; c < component_count; ++c) {
      sample.value[c] = recv[base + 3u + c];
    }
    gathered.push_back(std::move(sample));
  }
  return gathered;
#else
  return local_samples;
#endif
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

std::vector<int> communicatorWideBoundaryMarkers(
    const svmp::FE::assembly::IMeshAccess& mesh,
    const svmp::MeshComm& comm)
{
  auto markers = svmp::FE::interfaces::boundaryMarkers(mesh);

#ifdef MESH_HAS_MPI
  if (!comm.is_parallel()) {
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

svmp::FE::interfaces::FreeSurfaceGeometryOwnershipCollective
snapshotOwnershipCollective(
    const svmp::MeshComm& comm)
{
  svmp::FE::interfaces::FreeSurfaceGeometryOwnershipCollective collective;
  collective.rank = comm.rank();
  collective.size = comm.size();
  collective.all_gather_owned_rule_identity_values =
      [&comm](std::span<const std::uint64_t> local_values) {
        std::vector<std::uint64_t> gathered(local_values.begin(),
                                            local_values.end());
#ifdef MESH_HAS_MPI
        if (!comm.is_parallel()) {
          return gathered;
        }
        if (local_values.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
          throw std::runtime_error(
              "Free-surface snapshot ownership stream exceeds MPI limits.");
        }
        const int local_count = static_cast<int>(local_values.size());
        std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
        MPI_Allgather(&local_count,
                      1,
                      MPI_INT,
                      counts.data(),
                      1,
                      MPI_INT,
                      comm.native());
        std::vector<int> displacements(counts.size(), 0);
        int total_values = 0;
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
          if (counts[rank] < 0 ||
              counts[rank] >
                  std::numeric_limits<int>::max() - total_values) {
            throw std::runtime_error(
                "Free-surface snapshot ownership gather exceeds MPI limits.");
          }
          displacements[rank] = total_values;
          total_values += counts[rank];
        }
        gathered.assign(static_cast<std::size_t>(total_values), 0u);
#ifdef MPI_UINT64_T
        const MPI_Datatype identity_type = MPI_UINT64_T;
#else
        const MPI_Datatype identity_type = MPI_UNSIGNED_LONG_LONG;
#endif
        MPI_Allgatherv(local_values.empty() ? nullptr : local_values.data(),
                       local_count,
                       identity_type,
                       gathered.empty() ? nullptr : gathered.data(),
                       counts.data(),
                       displacements.data(),
                       identity_type,
                       comm.native());
#else
        if (comm.is_parallel()) {
          throw std::runtime_error(
              "Free-surface snapshot ownership validation requires MPI in a parallel run.");
        }
#endif
        return gathered;
      };
  collective.all_gather_revision_values =
      collective.all_gather_owned_rule_identity_values;
  return collective;
}

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

using application::core::WallCompatibleVelocityExtensionResult;
using application::core::WallVelocityExtensionConstraint;
using application::core::estimateSymmetricConditionNumber;
using application::core::extendVelocityInLevelSetNormalBand;
using application::core::globalOwnedVelocityExtensionMaskCount;
using application::core::globalVelocityExtensionGeometrySampleCount;
using application::core::kVelocityExtensionCoefficientTolerance;
using application::core::kVelocityExtensionMaxRegressionCondition;
using application::core::kVelocityExtensionMaxWetToDryAmplification;
using application::core::kVelocityExtensionRowTolerance;
using application::core::markVelocityExtensionTraceSupportCells;
using application::core::nodalVelocityExtensionInterfaceCells;
using application::core::synchronizeVelocityExtensionTraceSupportMask;
using application::core::velocityExtensionEdgeAdjacency;


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

bool generalizedAlphaPdeRateInitializationRequested(
    bool active_cut_domain_present)
{
  // Active-cut systems used to default this solve off because their dt-only
  // operator contains exact zero rows outside the retained physical domain.
  // TimeLoop now regularizes precisely those rows, so an active cut is not a
  // reason to replace the PDE-consistent initial rate with zero.  Keep the
  // argument explicit so this policy remains covered at the application
  // boundary if active-cut startup handling changes again.
  (void)active_cut_domain_present;
  return parseBoolEnv("SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT", true);
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

svmp::FE::level_set::LevelSetVelocitySource
parseLevelSetVelocitySource(const std::string& raw)
{
  const auto value = normalized_token(raw);
  using Source = svmp::FE::level_set::LevelSetVelocitySource;
  if (value.empty() || value == "coupled" || value == "coupledfield" ||
      value == "unknown") {
    return Source::CoupledField;
  }
  if (value == "prescribed" || value == "prescribeddata" ||
      value == "data") {
    return Source::PrescribedData;
  }
  if (value == "constant" || value == "constantvector") {
    return Source::ConstantVector;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Level-set Velocity_source must be one of "
      "'coupled_field', 'prescribed_data', or 'constant_vector'.");
}

svmp::FE::level_set::LevelSetPhaseSide
parseLevelSetPhaseSide(const std::string& raw)
{
  const auto value = normalized_token(raw);
  using Side = svmp::FE::level_set::LevelSetPhaseSide;
  if (value == "negative" || value == "minus") {
    return Side::Negative;
  }
  if (value == "positive" || value == "plus") {
    return Side::Positive;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Conservative_phase_liquid_side must be "
      "'negative' or 'positive'.");
}

std::array<svmp::FE::Real, 3> parseLevelSetVector3(
    const std::string& raw,
    const char* parameter_name)
{
  std::string normalized = raw;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  std::istringstream input(normalized);
  std::array<svmp::FE::Real, 3> result{};
  for (auto& value : result) {
    double parsed = 0.0;
    if (!(input >> parsed) || !std::isfinite(parsed)) {
      throw std::runtime_error(
          std::string("[svMultiPhysics::Application] ") + parameter_name +
          " must contain exactly three finite numeric components.");
    }
    value = static_cast<svmp::FE::Real>(parsed);
  }
  std::string trailing;
  if (input >> trailing) {
    throw std::runtime_error(
        std::string("[svMultiPhysics::Application] ") + parameter_name +
        " must contain exactly three finite numeric components.");
  }
  return result;
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

struct LevelSetMaintenanceRequest {
  struct OpenBoundary {
    std::string face_name{};
    bool inflow{false};
    std::optional<svmp::FE::Real> literal_inflow_value{};
  };

  std::string level_set_field_name{"level_set"};
  double isovalue{0.0};
  svmp::FE::level_set::LevelSetTransportForm transport_form{
      svmp::FE::level_set::LevelSetTransportForm::Advective};
  svmp::FE::level_set::LevelSetVelocityOptions velocity{};
  svmp::FE::level_set::LevelSetBoundPreservingOptions bound_preserving{};
  svmp::FE::level_set::LevelSetConservativePhaseOptions conservative_phase{};
  std::vector<OpenBoundary> open_boundaries{};
  svmp::FE::level_set::LevelSetReinitializationOptions reinitialization{};
  svmp::FE::level_set::LevelSetVolumeCorrectionOptions volume_correction{};
  std::optional<ActiveCutVolumeRequest> volume_cut_request{};
  bool curvature_projection_enabled{false};
  std::string curvature_field_name{};
  int curvature_projection_cadence_steps{1};
  svmp::FE::level_set::LevelSetCurvatureProjectionOptions curvature_projection{};
  bool volume_target_initialized{false};
  svmp::FE::Real volume_target{0.0};
  svmp::FE::Real volume_correction_reference_minimum_edge_length{0.0};
  svmp::FE::Real cumulative_volume_correction_interface_displacement{0.0};
  svmp::FE::Real cumulative_volume_correction_contact_line_displacement{0.0};
  bool conservative_phase_initialized{false};
  std::optional<
      svmp::FE::level_set::LevelSetP1PhaseTransportGraph>
      conservative_phase_graph{};
};

struct CurvatureProjectionCacheEntry {
  bool valid{false};
  std::uint64_t signature{0};
  bool fast_valid{false};
  std::uint64_t fast_signature{0};
  bool cut_context_signature_valid{false};
  const svmp::FE::assembly::CutIntegrationContext* cut_context_signature_context{nullptr};
  std::uint64_t cut_context_signature_context_revision{0};
  std::optional<int> cut_context_signature_marker{};
  std::uint64_t cut_context_signature_request_key{0};
  std::uint64_t cut_context_signature{0};
  std::size_t cut_context_signature_cache_hits{0};
  std::size_t cut_context_signature_cache_misses{0};
  std::uint64_t target_prescribed_revision{0};
  svmp::FE::level_set::LevelSetCurvatureProjectionWorkspace workspace{};
  svmp::FE::level_set::LevelSetCurvatureProjectionResult last_result{};
  std::vector<svmp::FE::Real> last_curvature_vertex_values{};
};

struct CurvatureProjectionCache {
  std::map<std::string, CurvatureProjectionCacheEntry> entries;
};

struct LevelSetAdvectionVelocityRequest {
  struct WallConstraint {
    std::string face_name{};
    // Preserve the raw Navier--Stokes Effective_direction metadata until the
    // mesh dimension is known.  Empty (and all-zero/all-active) directions
    // have the same all-component semantics as the production NS translator
    // for a homogeneous Dirichlet condition.
    std::vector<int> effective_direction{};
  };

  std::string level_set_field_name{"level_set"};
  std::string domain_id{"free_surface"};
  std::string source_velocity_field_name{"Velocity"};
  std::string target_velocity_field_name{"LevelSetAdvectionVelocity"};
  std::string operator_tag{"level_set"};
  std::string extension_method{"wall_compatible_normal"};
  int extension_band_layers{4};
  bool enforce_wall_impermeability{true};
  std::vector<std::string> wall_face_names{};
  std::vector<WallConstraint> wall_constraints{};
  double isovalue{0.0};
  int requested_interface_marker{-1};
  std::optional<std::size_t> active_cut_request_index{};
  LevelSetActiveSide active_side{LevelSetActiveSide::Negative};
};

struct AcceptedVelocityExtensionMapRecord {
  std::string level_set_field_name{};
  std::string source_velocity_field_name{};
  std::string target_velocity_field_name{};
  std::string geometry_domain_id{};
  std::string operator_tag{};
  std::string extension_method{};
  double isovalue{0.0};
  int extension_band_layers{0};
  bool enforce_wall_impermeability{false};
  LevelSetActiveSide retained_side{LevelSetActiveSide::Negative};
  std::shared_ptr<const application::core::VelocityExtensionMapSnapshot>
      snapshot{};
};

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

std::array<bool, 3> strongZeroVelocityComponentMask(
    std::span<const int> effective_direction,
    int dimension)
{
  if (dimension <= 0 || dimension > 3) {
    throw std::invalid_argument(
        "strong zero-velocity component mask requires dimension 1, 2, or 3");
  }
  std::array<bool, 3> mask{true, true, true};
  if (effective_direction.empty()) {
    return mask;
  }

  std::array<bool, 3> requested{false, false, false};
  int active_count = 0;
  for (int d = 0; d < dimension; ++d) {
    const auto component = static_cast<std::size_t>(d);
    requested[component] =
        component < effective_direction.size() &&
        effective_direction[component] != 0;
    active_count += requested[component] ? 1 : 0;
  }
  // Match NavierStokesRegister.cpp exactly for a zero-valued steady
  // Dirichlet BC: only a nonempty proper subset changes the default
  // all-component constraint.
  if (active_count > 0 && active_count < dimension) {
    mask = requested;
  }
  return mask;
}

struct ActiveCutContextRefreshReport {
  bool refreshed{false};
  std::uint64_t topology_key{0};
  std::uint64_t request_policy_key{0};
  std::uint64_t value_revision{0};
  std::map<int, std::uint64_t> evaluated_state_source_revisions{};
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
  // Cache/refresh counters and backend elapsed time are summed rank-local work
  // telemetry and deliberately include generated ghost-cell work.  Physical
  // domain counts and measures above are owned-cell global totals.
  std::size_t generated_cell_cache_hits{0};
  std::size_t generated_cell_cache_misses{0};
  std::size_t generated_cell_cache_unchanged_dof_hits{0};
  std::size_t generated_cell_refresh_candidates{0};
  std::size_t generated_cell_directly_affected{0};
  std::size_t generated_cell_affected_neighborhood{0};
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

double globalSumDouble(double local, const svmp::MeshComm& comm);
std::size_t globalSumSize(std::size_t local, const svmp::MeshComm& comm);
std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::MeshComm& comm);
bool globalAnyBool(bool local, const svmp::MeshComm& comm);

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
          token != "closestinterfacepoint" &&
          token != "wallcompatiblenormal" &&
          token != "normalextension" &&
          token != "graphnormalextension") {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Unsupported wet-extension level-set "
            "advection velocity method '" + trim_copy(*extension_method) +
            "'. The supported method is wall_compatible_normal; legacy nearest-* "
            "names are accepted as aliases for the corrected method.");
      }
      request.extension_method = "wall_compatible_normal";
    }
    if (const auto band_layers = first_defined_int_parameter(
            eq_params,
            {"Wet_extension_band_layers", "WetExtensionBandLayers",
             "Advection_velocity_extension_band_layers",
             "AdvectionVelocityExtensionBandLayers"})) {
      if (*band_layers <= 0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wet-extension band layers must be positive.");
      }
      request.extension_band_layers = *band_layers;
    }
    if (const auto wall_compatible = first_defined_bool_parameter(
            eq_params,
            {"Wet_extension_enforce_wall_impermeability",
             "WetExtensionEnforceWallImpermeability"})) {
      request.enforce_wall_impermeability = *wall_compatible;
    }
    bool explicit_wall_faces = false;
    if (const auto wall_faces = first_defined_parameter(
            eq_params,
            {"Wet_extension_wall_faces", "WetExtensionWallFaces",
             "Advection_velocity_extension_wall_faces",
             "AdvectionVelocityExtensionWallFaces"})) {
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
        if (!explicit_wall_faces) {
          const auto boundary_params = boundary->get_parameter_list();
          if (const auto contact_walls = first_defined_parameter(
                  boundary_params,
                  {"Contact_line_wall_faces", "ContactLineWallFaces",
                   "Wall_boundary_faces", "WallBoundaryFaces"})) {
            append_wall_faces(*contact_walls);
          }
        }
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
            first_defined_double_parameter(eq_params, {"Level_set_isovalue",
                                                      "LevelSetIsovalue",
                                                      "Interface_isovalue",
                                                      "InterfaceIsovalue"})) {
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

std::optional<ActiveCutVolumeRequest> matchingActiveCutVolumeRequest(
    const std::vector<ActiveCutVolumeRequest>& active_requests,
    const std::string& level_set_field_name,
    double isovalue)
{
  constexpr double kIsovalueTolerance = 1.0e-12;
  for (const auto& active_request : active_requests) {
    if (active_request.level_set_field_name != level_set_field_name) {
      continue;
    }
    if (std::abs(active_request.isovalue - isovalue) <= kIsovalueTolerance) {
      return active_request;
    }
  }
  for (const auto& active_request : active_requests) {
    if (active_request.level_set_field_name == level_set_field_name) {
      return active_request;
    }
  }
  return std::nullopt;
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

std::vector<double> orientedLevelSetForVelocityExtension(
    std::span<const double> level_set_values,
    double isovalue,
    LevelSetActiveSide retained_side)
{
  if (!std::isfinite(isovalue)) {
    throw std::invalid_argument(
        "velocity-extension orientation requires a finite isovalue");
  }
  std::vector<double> oriented(level_set_values.size(), 0.0);
  for (std::size_t vertex = 0u; vertex < level_set_values.size(); ++vertex) {
    const double centered = level_set_values[vertex] - isovalue;
    oriented[vertex] = retained_side == LevelSetActiveSide::Negative
                           ? centered
                           : -centered;
    if (!std::isfinite(oriented[vertex])) {
      throw std::invalid_argument(
          "velocity-extension orientation received a non-finite level-set value");
    }
  }
  return oriented;
}

const char* activeSideName(LevelSetActiveSide side) noexcept
{
  return side == LevelSetActiveSide::Negative
             ? "LevelSetNegative"
             : "LevelSetPositive";
}

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

svmp::FE::geometry::CutIntegrationSide cutIntegrationSide(
    LevelSetActiveSide side) noexcept
{
  return side == LevelSetActiveSide::Negative
             ? svmp::FE::geometry::CutIntegrationSide::Negative
             : svmp::FE::geometry::CutIntegrationSide::Positive;
}

svmp::FE::geometry::CutIntegrationSide oppositeCutIntegrationSide(
    svmp::FE::geometry::CutIntegrationSide side) noexcept
{
  return side == svmp::FE::geometry::CutIntegrationSide::Negative
             ? svmp::FE::geometry::CutIntegrationSide::Positive
             : svmp::FE::geometry::CutIntegrationSide::Negative;
}

const char* retainedVolumeSidesName(
    application::core::ActiveCutVolumeRetention retention) noexcept
{
  return retention == application::core::ActiveCutVolumeRetention::ActiveAndInactive
             ? "active_and_inactive"
             : "active_only";
}

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
  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << "CUT_RULE_DUMP step=" << step << " n_rules=" << rules.size() << "\n";
  for (std::size_t i = 0; i < rules.size() && i < metadata.size(); ++i) {
    const auto& rule = rules[i];
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
    oss << "CUT_RULE step=" << step << " idx=" << i << " cell=" << cell
        << " side=" << side << " mside=" << side_name(meta.side)
        << " full=" << (rule.full_cell_equivalent ? 1 : 0)
        << " measure=" << rule.measure
        << " parent_measure=" << rule.parent_measure
        << " vf=" << rule.volume_fraction
        << " mvf=" << meta.volume_fraction
        << " n_qp=" << rule.points.size() << "\n";
    for (const auto& q : rule.points) {
      oss << "CUT_QP step=" << step << " cell=" << cell << " side=" << side
          << " x=" << q.point[0] << " y=" << q.point[1] << " z=" << q.point[2]
          << " w=" << q.weight << "\n";
    }
  }
  application::core::oopCout() << oss.str() << std::flush;
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

std::optional<int> configuredInterfaceVelocityMarker(
    const svmp::FE::systems::FESystem& system,
    const LevelSetAdvectionVelocityRequest& request)
{
  const auto* cut_context = system.cutIntegrationContext();
  if (request.requested_interface_marker >= 0) {
    return request.requested_interface_marker;
  }

  // Reconstruct the lifecycle's stable marker for the usual no-explicit-
  // marker configuration only when this request has keyed active-cut
  // provenance.  Integer-only system/context registration is insufficient:
  // an unregistered nodal request could otherwise match an unrelated marker.
  // The lifecycle registry rejects hash collisions rather than probing them.
  const auto phi_field = system.findFieldByName(request.level_set_field_name);
  if (request.active_cut_request_index.has_value() &&
      phi_field != svmp::FE::INVALID_FIELD_ID) {
    svmp::FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = svmp::FE::interfaces::LevelSetInterfaceSource::fromField(
        phi_field);
    key.domain_id = request.domain_id;
    key.isovalue = request.isovalue;
    key.requested_marker = -1;
    const int stable_marker =
        svmp::FE::interfaces::stableGeneratedInterfaceMarker(key);
    if (system.isGeneratedEmbeddedInterfaceMarkerRegistered(stable_marker) ||
        (cut_context != nullptr &&
         (cut_context->hasExpectedGeneratedSourceValueRevision(stable_marker) ||
          cut_context->hasGeneratedVolumeMarker(stable_marker) ||
          cut_context->hasGeneratedInterfaceMarker(stable_marker)))) {
      return stable_marker;
    }
  }

  if (cut_context != nullptr &&
      request.active_cut_request_index.has_value()) {
    const auto& markers = cut_context->generatedVolumeMarkers();
    const auto index = *request.active_cut_request_index;
    if (index < markers.size()) {
      return markers[index];
    } else {
      const auto& interface_markers =
          cut_context->generatedInterfaceMarkers();
      if (index < interface_markers.size()) {
        return interface_markers[index];
      }
    }
  }
  return std::nullopt;
}

std::optional<int> interfaceVelocitySampleMarker(
    const svmp::FE::systems::FESystem& system,
    const LevelSetAdvectionVelocityRequest& request)
{
  const auto* cut_context = system.cutIntegrationContext();
  if (cut_context == nullptr) {
    return std::nullopt;
  }
  const auto marker = configuredInterfaceVelocityMarker(system, request);
  if (!marker.has_value() ||
      (!cut_context->hasGeneratedVolumeMarker(*marker) &&
       !cut_context->hasGeneratedInterfaceMarker(*marker))) {
    return std::nullopt;
  }
  return marker;
}

bool hasAuthoritativeInterfaceVelocityContext(
    const svmp::FE::systems::FESystem& system,
    const LevelSetAdvectionVelocityRequest& request)
{
  const auto marker = configuredInterfaceVelocityMarker(system, request);
  const auto* cut_context = system.cutIntegrationContext();
  if (!marker.has_value()) {
    return false;
  }

  // The expected-source revision is registered even when this rank has no
  // retained rules for the generated interface.  Likewise, the FESystem
  // marker registration survives a locally empty cut context.  Either is an
  // authoritative declaration: a rank-local empty candidate set must remain
  // empty and must not be replaced by a nodal crossing search.
  return system.isGeneratedEmbeddedInterfaceMarkerRegistered(*marker) ||
         (cut_context != nullptr &&
          (cut_context->hasExpectedGeneratedSourceValueRevision(*marker) ||
           cut_context->hasGeneratedVolumeMarker(*marker) ||
           cut_context->hasGeneratedInterfaceMarker(*marker)));
}

std::vector<svmp::FE::MeshIndex> interfaceVelocitySampleCandidateCells(
    const svmp::FE::systems::FESystem& system,
    const LevelSetAdvectionVelocityRequest& request)
{
  const auto* cut_context = system.cutIntegrationContext();
  const auto marker = interfaceVelocitySampleMarker(system, request);
  if (cut_context == nullptr || !marker.has_value()) {
    return {};
  }
  cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
      system.meshAccess());

  std::vector<svmp::FE::MeshIndex> cells;
  if (cut_context->hasGeneratedVolumeMarker(*marker)) {
    cells = activeCutCellsForMarkerAndSide(
        *cut_context, *marker, request.active_side);
  }
  if (cut_context->hasGeneratedInterfaceMarker(*marker)) {
    for (const auto* rule : cut_context->interfaceRulesForMarker(*marker)) {
      if (rule == nullptr ||
          rule->kind != svmp::FE::geometry::CutQuadratureKind::Interface ||
          rule->provenance.parent_entity < 0) {
        continue;
      }
      cells.push_back(rule->provenance.parent_entity);
    }
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
  case Point::OuterFixedPointState:
    return "outer_fixed_point";
  case Point::ProjectedOuterFixedPointState:
    return "projected_outer_fixed_point";
  case Point::EndpointCandidateState:
    return "endpoint_candidate";
  case Point::ProjectedEndpointCandidateState:
    return "projected_endpoint_candidate";
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
  case Point::RestoredOuterFixedPointState:
    return "restored_outer_fixed_point";
  case Point::RestoredProjectedOuterFixedPointState:
    return "restored_projected_outer_fixed_point";
  case Point::RestoredTimeStepState:
    return "restored_time_step";
  case Point::RestoredProjectedTimeStepState:
    return "restored_projected_time_step";
  case Point::FinalResidualAssembly:
    return "final_residual";
  }
  return "unknown";
}

bool allowCachedCurvatureAfterProjectionFailure(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
{
  using Point = svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  return point == Point::LineSearchTrialResidual &&
         parseBoolEnv("SVMP_CURVATURE_REUSE_CACHE_ON_TRIAL_FAILURE", false);
}

bool synchronizeTransientLineSearchTrials(
    bool residual_defining_state_changes)
{
  // R(u, G(u)) is the nonlinear residual when generated cuts, projected
  // curvature, velocity extension, or state-dependent constraints are part of
  // the problem definition.  Backtracking must evaluate that deterministic
  // residual at each trial.  Freezing G at the previous iterate instead
  // globalizes a different Picard surrogate and can accept a step that
  // increases the actual residual.  Keep an explicit opt-out for controlled
  // legacy comparisons.
  return parseBoolEnv("SVMP_SYNC_LINE_SEARCH_TRIALS",
                      residual_defining_state_changes);
}

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
        << solve_kind
        << " nonlinear solve diagnostic=cut_topology_change_nonsmooth_event"
        << " event_class=nonsmooth_cut_topology_change"
        << " newton_consistency=not_expected"
        << " jacobian_validity=piecewise_smooth_topology_only sync_point="
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
        << " generated_cell_work_scope=summed_rank_local_including_ghost_cells"
        << " generated_cell_cache_hits="
        << report.generated_cell_cache_hits
        << " generated_cell_cache_misses="
        << report.generated_cell_cache_misses
        << " generated_cell_cache_unchanged_dof_hits="
        << report.generated_cell_cache_unchanged_dof_hits
        << " generated_cell_refresh_candidates="
        << report.generated_cell_refresh_candidates
        << " generated_cell_directly_affected="
        << report.generated_cell_directly_affected
        << " generated_cell_affected_neighborhood="
        << report.generated_cell_affected_neighborhood
        << " generated_domain_cache_hits="
        << report.generated_domain_cache_hits
        << " linear_full_cell_fast_path_cells="
        << report.linear_full_cell_fast_path_count
        << " process_vm_kb=" << report.process_vm_kb
        << " process_rss_kb=" << report.process_rss_kb
        << " basis_cache_entries=" << report.basis_cache_entries
        << " cut_adjacent_scope=summed_rank_local_including_ghost_cells"
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

  application::core::oopCout()
      << "[svMultiPhysics::Application] WARNING generated level-set interface "
      << "uses corner-linearized cut geometry"
      << " marker=" << result.interface_marker
      << " field='" << request.level_set_field_name << "'"
      << " domain_id='" << request.domain_id << "'"
      << " corner_linearized_cells="
      << global_corner_linearized_cell_count
      << " cell_count=" << global_cell_count
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
        application::core::collectCutVolumeMeasures(mesh, rules);
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
    application::core::oopCout()
        << "[svMultiPhysics::Application] Wet volume diagnostic"
        << " step=" << step
        << " time=" << time
        << " field='" << diagnostic.level_set_field_name << "'"
        << " domain_id='" << diagnostic.domain_id << "'"
        << " marker=" << diagnostic.marker
        << " free_surface_snapshot_revision_key="
        << diagnostic.free_surface_snapshot_revision_key
        << " source_value_revision=" << diagnostic.source_value_revision
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
  const auto active_requests = activeCutVolumeRequests(params);
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

    if (const auto velocity_field = first_defined_parameter(
            eq_params,
            {"Velocity_field_name", "VelocityFieldName",
             "Advection_velocity_field", "AdvectionVelocityField"})) {
      request.velocity.field_name = trim_copy(*velocity_field);
    }
    if (const auto velocity_source = first_defined_parameter(
            eq_params, {"Velocity_source", "VelocitySource"})) {
      request.velocity.source = parseLevelSetVelocitySource(*velocity_source);
    }
    const bool wet_extension_enabled =
        first_defined_bool_parameter(
            eq_params,
            {"Use_wet_extension_advection_velocity",
             "UseWetExtensionAdvectionVelocity",
             "Update_advection_velocity_from_wet_region",
             "UpdateAdvectionVelocityFromWetRegion"})
            .value_or(false) ||
        first_defined_parameter(
            eq_params,
            {"Advection_velocity_from_field",
             "AdvectionVelocityFromField",
             "Source_velocity_field_name",
             "SourceVelocityFieldName",
             "Physical_velocity_field_name",
             "PhysicalVelocityFieldName"})
            .has_value();
    if (wet_extension_enabled) {
      // The translator promotes the generated coefficient to an algebraic
      // unknown.  Maintenance/safety checks must read that solved field from
      // the coupled state rather than looking for a prescribed buffer.
      request.velocity.source =
          svmp::FE::level_set::LevelSetVelocitySource::CoupledField;
    }
    if (const auto constant_velocity = first_defined_parameter(
            eq_params,
            {"Constant_velocity", "ConstantVelocity",
             "Velocity_value", "VelocityValue"})) {
      request.velocity.source =
          svmp::FE::level_set::LevelSetVelocitySource::ConstantVector;
      request.velocity.constant_value = parseLevelSetVector3(
          *constant_velocity, "Constant_velocity");
    }

    if (const auto enabled = first_defined_bool_parameter(
            eq_params,
            {"Enable_conservative_phase_transport",
             "EnableConservativePhaseTransport",
             "Conservative_phase_transport",
             "ConservativePhaseTransport"})) {
      request.conservative_phase.enabled = *enabled;
    }
    if (const auto field = first_defined_parameter(
            eq_params,
            {"Conservative_phase_field_name",
             "ConservativePhaseFieldName",
             "Liquid_indicator_field_name",
             "LiquidIndicatorFieldName"})) {
      request.conservative_phase.liquid_indicator.field_name =
          trim_copy(*field);
    }
    if (const auto auto_register = first_defined_bool_parameter(
            eq_params,
            {"Auto_register_conservative_phase_field",
             "AutoRegisterConservativePhaseField"})) {
      request.conservative_phase.liquid_indicator.auto_register_field =
          *auto_register;
    }
    if (const auto side = first_defined_parameter(
            eq_params,
            {"Conservative_phase_liquid_side",
             "ConservativePhaseLiquidSide"})) {
      request.conservative_phase.liquid_side =
          parseLevelSetPhaseSide(*side);
    }
    if (const auto tolerance = first_defined_double_parameter(
            eq_params,
            {"Conservative_phase_invariant_tolerance",
             "ConservativePhaseInvariantTolerance"})) {
      request.conservative_phase.invariant_tolerance =
          static_cast<svmp::FE::Real>(*tolerance);
    }
    if (const auto tolerance = first_defined_double_parameter(
            eq_params,
            {"Conservative_phase_component_activity_tolerance",
             "ConservativePhaseComponentActivityTolerance"})) {
      request.conservative_phase.component_activity_tolerance =
          static_cast<svmp::FE::Real>(*tolerance);
    }
    if (const auto maximum_courant = first_defined_double_parameter(
            eq_params,
            {"Conservative_phase_maximum_courant",
             "ConservativePhaseMaximumCourant"})) {
      request.conservative_phase.maximum_courant =
          static_cast<svmp::FE::Real>(*maximum_courant);
    }
    if (const auto enforce = first_defined_bool_parameter(
            eq_params,
            {"Conservative_phase_enforce_courant_limit",
             "ConservativePhaseEnforceCourantLimit"})) {
      request.conservative_phase.enforce_courant_limit = *enforce;
    }
    if (const auto preserve = first_defined_bool_parameter(
            eq_params,
            {"Conservative_phase_require_constant_preservation",
             "ConservativePhaseRequireConstantPreservation"})) {
      request.conservative_phase.require_constant_preservation = *preserve;
    }
    if (const auto write = first_defined_bool_parameter(
            eq_params,
            {"Conservative_phase_write_flux_artifacts",
             "ConservativePhaseWriteFluxArtifacts"})) {
      request.conservative_phase.write_flux_artifacts = *write;
    }
    if (const auto cadence = first_defined_int_parameter(
            eq_params,
            {"Conservative_phase_flux_artifact_cadence_steps",
             "ConservativePhaseFluxArtifactCadenceSteps"})) {
      request.conservative_phase.flux_artifact_cadence_steps = *cadence;
    }
    if (const auto classify = first_defined_bool_parameter(
            eq_params,
            {"Conservative_phase_classify_nonprimary_components_as_satellites",
             "ConservativePhaseClassifyNonprimaryComponentsAsSatellites"})) {
      request.conservative_phase
          .classify_nonprimary_components_as_satellites = *classify;
    }
    if (const auto regions = first_defined_parameter(
            eq_params,
            {"Conservative_phase_fixed_flux_regions",
             "ConservativePhaseFixedFluxRegions"})) {
      request.conservative_phase.fixed_flux_regions =
          svmp::FE::level_set::parseLevelSetPhaseRegionBoxes(*regions);
    }
    if (const auto tolerance = first_defined_double_parameter(
            eq_params,
            {"Conservative_phase_impermeable_normal_velocity_tolerance",
             "ConservativePhaseImpermeableNormalVelocityTolerance"})) {
      request.conservative_phase.impermeable_normal_velocity_tolerance =
          static_cast<svmp::FE::Real>(*tolerance);
    }
    if (const auto reconcile = first_defined_bool_parameter(
            eq_params,
            {"Conservative_phase_reconcile_geometry",
             "ConservativePhaseReconcileGeometry"})) {
      request.conservative_phase.reconcile_geometry = *reconcile;
    }
    if (const auto tolerance = first_defined_double_parameter(
            eq_params,
            {"Conservative_phase_geometry_measure_tolerance",
             "ConservativePhaseGeometryMeasureTolerance"})) {
      request.conservative_phase.geometry_measure_tolerance =
          static_cast<svmp::FE::Real>(*tolerance);
    }
    if (const auto iterations = first_defined_int_parameter(
            eq_params,
            {"Conservative_phase_geometry_correction_max_iterations",
             "ConservativePhaseGeometryCorrectionMaxIterations"})) {
      request.conservative_phase.geometry_correction_max_iterations =
          *iterations;
    }
    if (const auto fraction = first_defined_double_parameter(
            eq_params,
            {"Conservative_phase_maximum_geometry_displacement_fraction",
             "ConservativePhaseMaximumGeometryDisplacementFraction"})) {
      request.conservative_phase.maximum_geometry_displacement_fraction =
          static_cast<svmp::FE::Real>(*fraction);
    }

    if (const auto enabled = first_defined_bool_parameter(
            eq_params,
            {"Enable_bound_preserving_limiter",
             "EnableBoundPreservingLimiter",
             "Bound_preserving_limiter", "BoundPreservingLimiter"})) {
      request.bound_preserving.enabled = *enabled;
    }
    if (const auto tolerance = first_defined_double_parameter(
            eq_params,
            {"Bound_preserving_bound_tolerance",
             "BoundPreservingBoundTolerance"})) {
      request.bound_preserving.bound_tolerance =
          static_cast<svmp::FE::Real>(*tolerance);
    }
    if (const auto tolerance = first_defined_double_parameter(
            eq_params,
            {"Bound_preserving_sign_tolerance",
             "BoundPreservingSignTolerance"})) {
      request.bound_preserving.sign_tolerance =
          static_cast<svmp::FE::Real>(*tolerance);
    }
    if (const auto tolerance = first_defined_double_parameter(
            eq_params,
            {"Bound_preserving_courant_tolerance",
             "BoundPreservingCourantTolerance"})) {
      request.bound_preserving.courant_tolerance =
          static_cast<svmp::FE::Real>(*tolerance);
    }
    if (const auto maximum_courant = first_defined_double_parameter(
            eq_params,
            {"Bound_preserving_maximum_courant",
             "BoundPreservingMaximumCourant"})) {
      request.bound_preserving.maximum_courant =
          static_cast<svmp::FE::Real>(*maximum_courant);
    }
    if (const auto enabled = first_defined_bool_parameter(
            eq_params,
            {"Bound_preserving_enforce_courant_limit",
             "BoundPreservingEnforceCourantLimit"})) {
      request.bound_preserving.enforce_courant_limit = *enabled;
    }
    if (const auto enabled = first_defined_bool_parameter(
            eq_params,
            {"Bound_preserving_enforce_impermeable_boundaries",
             "BoundPreservingEnforceImpermeableBoundaries"})) {
      request.bound_preserving.enforce_impermeable_boundaries = *enabled;
    }
    if (const auto tolerance = first_defined_double_parameter(
            eq_params,
            {"Bound_preserving_impermeable_normal_velocity_tolerance",
             "BoundPreservingImpermeableNormalVelocityTolerance"})) {
      request.bound_preserving.impermeable_normal_velocity_tolerance =
          static_cast<svmp::FE::Real>(*tolerance);
    }

    for (auto* boundary : eq->boundary_conditions) {
      if (boundary == nullptr || !boundary->type.defined() ||
          !boundary->name.defined()) {
        continue;
      }
      const auto boundary_type = normalized_token(boundary->type.value());
      const bool inflow =
          boundary_type == "levelsetinflow" || boundary_type == "inflow" ||
          boundary_type == "levelsetdirichlet";
      const bool outflow =
          boundary_type == "levelsetoutflow" || boundary_type == "outflow";
      if (!inflow && !outflow) {
        continue;
      }
      LevelSetMaintenanceRequest::OpenBoundary open_boundary;
      open_boundary.face_name = trim_copy(boundary->name.value());
      open_boundary.inflow = inflow;
      if (inflow) {
        const auto boundary_params = boundary->get_parameter_list();
        if (const auto value = first_defined_double_parameter(
                boundary_params, {"Value", "Level_set_value"})) {
          open_boundary.literal_inflow_value =
              static_cast<svmp::FE::Real>(*value);
        }
      }
      request.open_boundaries.push_back(std::move(open_boundary));
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
    if (const auto displacement = first_defined_double_parameter(
            eq_params,
            {"Reinitialization_max_zero_set_displacement",
             "ReinitializationMaxZeroSetDisplacement"})) {
      request.reinitialization.max_zero_set_displacement =
          static_cast<svmp::FE::Real>(*displacement);
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
    if (const auto relative_error = first_defined_double_parameter(
            eq_params,
            {"Volume_correction_minimum_relative_error",
             "VolumeCorrectionMinimumRelativeError"})) {
      request.volume_correction.minimum_relative_volume_error =
          static_cast<svmp::FE::Real>(*relative_error);
    }
    if (const auto displacement_fraction = first_defined_double_parameter(
            eq_params,
            {"Volume_correction_maximum_interface_displacement_fraction",
             "VolumeCorrectionMaximumInterfaceDisplacementFraction"})) {
      request.volume_correction.maximum_interface_displacement_fraction =
          static_cast<svmp::FE::Real>(*displacement_fraction);
    }
    if (const auto cumulative_displacement_fraction =
            first_defined_double_parameter(
                eq_params,
                {"Volume_correction_maximum_cumulative_interface_displacement_fraction",
                 "VolumeCorrectionMaximumCumulativeInterfaceDisplacementFraction"})) {
      request.volume_correction
          .maximum_cumulative_interface_displacement_fraction =
          static_cast<svmp::FE::Real>(
              *cumulative_displacement_fraction);
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
                {"Curvature_projection_least_squares_rank_tolerance",
                 "CurvatureProjectionLeastSquaresRankTolerance",
                 "Curvature_projection_normal_equation_tolerance",
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
    if (const auto max_neighbor =
            first_defined_int_parameter(
                eq_params,
                {"Curvature_projection_max_neighbor_fallback_vertices",
                 "CurvatureProjectionMaxNeighborFallbackVertices",
                 "Projected_curvature_max_neighbor_fallback_vertices",
                 "ProjectedCurvatureMaxNeighborFallbackVertices"})) {
      request.curvature_projection.max_neighbor_fallback_vertices =
          *max_neighbor;
    }
    if (const auto max_zero =
            first_defined_int_parameter(
                eq_params,
                {"Curvature_projection_max_zero_fallback_vertices",
                 "CurvatureProjectionMaxZeroFallbackVertices",
                 "Projected_curvature_max_zero_fallback_vertices",
                 "ProjectedCurvatureMaxZeroFallbackVertices"})) {
      request.curvature_projection.max_zero_fallback_vertices = *max_zero;
    }
    if (const auto weight =
            first_defined_double_parameter(
                eq_params,
                {"Curvature_projection_supplemental_sample_weight",
                 "CurvatureProjectionSupplementalSampleWeight",
                 "Projected_curvature_supplemental_sample_weight",
                 "ProjectedCurvatureSupplementalSampleWeight",
                 "Curvature_projection_interface_sample_weight",
                 "CurvatureProjectionInterfaceSampleWeight"})) {
      request.curvature_projection.supplemental_sample_weight =
          static_cast<svmp::FE::Real>(*weight);
    }
    if (const auto width =
            first_defined_double_parameter(
                eq_params,
                {"Curvature_projection_narrow_band_width",
                 "CurvatureProjectionNarrowBandWidth",
                 "Projected_curvature_narrow_band_width",
                 "ProjectedCurvatureNarrowBandWidth",
                 "Curvature_projection_interface_band_width",
                 "CurvatureProjectionInterfaceBandWidth"})) {
      request.curvature_projection.narrow_band_width =
          static_cast<svmp::FE::Real>(*width);
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
    if (const auto mode =
            first_defined_parameter(
                eq_params,
                {"Curvature_projection_smoothing_mode",
                 "CurvatureProjectionSmoothingMode",
                 "Projected_curvature_smoothing_mode",
                 "ProjectedCurvatureSmoothingMode",
                 "Curvature_projection_regularization",
                 "CurvatureProjectionRegularization"})) {
      request.curvature_projection.smoothing_mode =
          svmp::FE::level_set::parseLevelSetCurvatureSmoothingMode(*mode);
    }
    request.curvature_projection.isovalue =
        static_cast<svmp::FE::Real>(request.isovalue);
    request.volume_cut_request = matchingActiveCutVolumeRequest(
        active_requests,
        request.level_set_field_name,
        request.isovalue);

    if (request.bound_preserving.enabled ||
        request.conservative_phase.enabled ||
        request.reinitialization.enabled ||
        request.volume_correction.enabled ||
        request.curvature_projection_enabled) {
      requests.push_back(std::move(request));
    }
  }
  return requests;
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
            {"Level_set_field_name", "LevelSetFieldName",
             "Level_set_field", "LevelSetField", "Field_name"})) {
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
    if (request.reinitialization.enabled || request.volume_correction.enabled) {
      transport_maintained_fields.insert(request.level_set_field_name);
    }
    application::core::oopCout()
        << "[svMultiPhysics::Application] Level-set maintenance diagnostic"
        << " field='" << request.level_set_field_name << "'"
        << " reinitialization="
        << (request.reinitialization.enabled ? "enabled" : "disabled")
        << " bound_preserving="
        << (request.bound_preserving.enabled ? "enabled" : "disabled")
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
        << " volume_correction_maximum_interface_displacement_fraction="
        << request.volume_correction.maximum_interface_displacement_fraction
        << " volume_correction_maximum_cumulative_interface_displacement_fraction="
        << request.volume_correction
               .maximum_cumulative_interface_displacement_fraction
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
    application::core::oopCout()
        << "[svMultiPhysics::Application] WARNING accepted pressure update "
        << "diagnostic skipped: solution size mismatch"
        << " diagnostic=accepted_pressure_update_guard_skipped"
        << " phase='" << phase << "'"
        << " previous_size=" << previous_solution.size()
        << " current_size=" << current_solution.size() << std::endl;
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
    application::core::oopCout()
        << "[svMultiPhysics::Application] WARNING accepted pressure update "
        << "diagnostic skipped: pressure field not found"
        << " diagnostic=accepted_pressure_update_guard_skipped"
        << " phase='" << phase << "'"
        << " field='" << pressure_field_name << "'" << std::endl;
    return false;
  }

  const auto& pressure_dofs = system.fieldDofHandler(pressure_field);
  const auto* pressure_entity_map = pressure_dofs.getEntityDofMap();
  if (pressure_entity_map == nullptr) {
    application::core::oopCout()
        << "[svMultiPhysics::Application] WARNING accepted pressure update "
        << "diagnostic skipped: pressure field has no entity DOF map"
        << " diagnostic=accepted_pressure_update_guard_skipped"
        << " phase='" << phase << "'"
        << " field='" << pressure_field_name << "'" << std::endl;
    return false;
  }

  const auto requests = activeCutVolumeRequests(params);
  const auto* cut_context = system.cutIntegrationContext();
  if (requests.empty() || cut_context == nullptr) {
    application::core::oopCout()
        << "[svMultiPhysics::Application] WARNING accepted pressure update "
        << "diagnostic skipped: active cut context unavailable"
        << " diagnostic=accepted_pressure_update_guard_skipped"
        << " phase='" << phase << "'"
        << " active_requests=" << requests.size()
        << " has_cut_context=" << (cut_context != nullptr ? 1 : 0)
        << std::endl;
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
    application::core::oopCout()
        << "[svMultiPhysics::Application] WARNING accepted pressure update "
        << "diagnostic skipped: pressure field DOFs outside solution vector"
        << " diagnostic=accepted_pressure_update_guard_skipped"
        << " phase='" << phase << "'"
        << " field='" << pressure_field_name << "'"
        << " pressure_offset=" << pressure_offset
        << " pressure_dofs=" << pressure_field_dofs
        << " solution_size=" << current_solution.size() << std::endl;
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

  application::core::oopCout()
      << "[svMultiPhysics::Application] Accepted pressure update diagnostic"
      << " diagnostic=accepted_pressure_update_guard"
      << " phase='" << phase << "'"
      << " step=" << step
      << " time=" << time
      << " dt=" << dt
      << " field='" << pressure_field_name << "'"
      << " retained_active_volume_rules=" << global_retained_rules
      << " active_supported_vertices=" << global_supported_vertices
      << " compared_vertex_pressure_dofs=" << global_compared_vertices
      << " skipped_nonvertex_pressure_vertices=" << global_skipped_vertices
      << " local_worst_vertex=" << local_vertex
      << " local_worst_dof=" << local_dof
      << " local_abs_pressure_delta_pa=" << local_max_abs_delta
      << " global_abs_pressure_delta_pa=" << global_max_abs_delta
      << " local_pressure_delta_pa=" << local_signed_delta
      << " local_from_pressure_pa=" << local_from_pressure
      << " local_to_pressure_pa=" << local_to_pressure
      << " support_class="
      << activePressureUpdateSupportClass(local_max_wet,
                                          local_min_positive_wet)
      << " incident_wet_fraction_max=" << local_max_wet
      << " incident_wet_fraction_min_positive=" << local_min_positive_wet
      << " threshold_pa=" << threshold
      << " triggered=" << (triggered ? 1 : 0)
      << std::endl;

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

std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    std::uint64_t minimum = 0u;
    std::uint64_t maximum = 0u;
#ifdef MPI_UINT64_T
    const MPI_Datatype datatype = MPI_UINT64_T;
#else
    const MPI_Datatype datatype = MPI_UNSIGNED_LONG_LONG;
#endif
    MPI_Allreduce(&local, &minimum, 1, datatype, MPI_MIN, comm.native());
    MPI_Allreduce(&local, &maximum, 1, datatype, MPI_MAX, comm.native());
    return {minimum, maximum};
  }
#else
  (void)comm;
#endif

  return {local, local};
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

std::uint64_t acceptedContactStageRevision(
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::uint64_t snapshot_revision,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::span<const svmp::FE::Real> stage_solution)
{
  constexpr std::uint64_t offset = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  std::uint64_t hash = offset;
  const auto mix = [&hash](std::uint64_t value) {
    hash ^= value;
    hash *= prime;
  };
  const auto real_bits = [](svmp::FE::Real value) {
    if (value == svmp::FE::Real{0.0}) {
      value = svmp::FE::Real{0.0};
    }
    std::uint64_t bits = 0u;
    static_assert(sizeof(value) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
  };
  mix(previous_state_revision);
  mix(endpoint_state_revision);
  mix(snapshot_revision);
  mix(real_bits(stage_time));
  mix(real_bits(stage_alpha_f));
  mix(static_cast<std::uint64_t>(stage_solution.size()));
  for (const auto value : stage_solution) {
    mix(real_bits(value));
  }
  return hash == 0u ? 1u : hash;
}

std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
evaluateAcceptedFreeSurfaceContactStages(
    application::core::SimulationComponents& sim,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::span<const svmp::FE::Real> stage_solution)
{
  const auto declarations =
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations();
  const auto dynamic_count = std::count_if(
      declarations.begin(), declarations.end(), [](const auto& declaration) {
        return !declaration.parameters.dynamic_contact_coefficients.empty();
      });
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const double local_dynamic_count =
      static_cast<double>(dynamic_count);
  if (globalMinDouble(local_dynamic_count, comm) !=
      globalMaxDouble(local_dynamic_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic contact-stage declarations differ across the FE communicator.");
  }
  if (dynamic_count == 0) {
    return {};
  }
  if (!std::isfinite(stage_time) || !std::isfinite(stage_alpha_f) ||
      !(stage_alpha_f > svmp::FE::Real{0.0}) ||
      stage_alpha_f > svmp::FE::Real{1.0} ||
      previous_state_revision == 0u || endpoint_state_revision == 0u) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage provenance is incomplete.");
  }
  const double local_solution_size =
      static_cast<double>(stage_solution.size());
  if (globalMinDouble(local_solution_size, comm) !=
      globalMaxDouble(local_solution_size, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Contact-stage solution layouts differ across the FE communicator.");
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (globalMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage recording requires an authoritative geometry snapshot on every rank.");
  }

  std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
      stages;
  stages.reserve(static_cast<std::size_t>(dynamic_count));
  for (const auto& declaration : declarations) {
    if (declaration.parameters.dynamic_contact_coefficients.empty()) {
      continue;
    }
    const double local_marker =
        static_cast<double>(declaration.interface_marker);
    const double local_velocity_field =
        static_cast<double>(declaration.velocity_field);
    const double local_contact_count = static_cast<double>(
        declaration.parameters.dynamic_contact_coefficients.size());
    if (globalMinDouble(local_marker, comm) !=
            globalMaxDouble(local_marker, comm) ||
        globalMinDouble(local_velocity_field, comm) !=
            globalMaxDouble(local_velocity_field, comm) ||
        globalMinDouble(local_contact_count, comm) !=
            globalMaxDouble(local_contact_count, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic contact-stage declaration ordering differs across the FE communicator.");
    }
    for (const auto& coefficient :
         declaration.parameters.dynamic_contact_coefficients) {
      const double boundary_marker =
          static_cast<double>(coefficient.boundary_marker);
      const double equilibrium_angle =
          static_cast<double>(
              coefficient.equilibrium_contact_angle_radians);
      const double mobility = static_cast<double>(coefficient.mobility);
      const double slip_length =
          static_cast<double>(coefficient.slip_length);
      const double dynamic_viscosity =
          static_cast<double>(coefficient.dynamic_viscosity);
      if (globalMinDouble(boundary_marker, comm) !=
              globalMaxDouble(boundary_marker, comm) ||
          globalMinDouble(equilibrium_angle, comm) !=
              globalMaxDouble(equilibrium_angle, comm) ||
          globalMinDouble(mobility, comm) !=
              globalMaxDouble(mobility, comm) ||
          globalMinDouble(slip_length, comm) !=
              globalMaxDouble(slip_length, comm) ||
          globalMinDouble(dynamic_viscosity, comm) !=
              globalMaxDouble(dynamic_viscosity, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Dynamic contact-stage wall coefficients differ across the FE communicator for marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
    }
    const bool local_marker_available =
        context->hasFreeSurfaceGeometrySnapshotForMarker(
            declaration.interface_marker);
    if (globalMinDouble(local_marker_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage marker " +
          std::to_string(declaration.interface_marker) +
          " is missing an authoritative geometry snapshot on at least one rank.");
    }
    const auto snapshot_revision =
        context->freeSurfaceGeometrySnapshotRevisionForMarker(
            declaration.interface_marker);
    const auto [minimum_revision, maximum_revision] =
        globalMinMaxUint64(snapshot_revision, comm);
    if (minimum_revision != maximum_revision) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage geometry revision differs across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     snapshot_revision;
        });
    if (globalMinDouble(found != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    bool local_snapshot_current = true;
    try {
      context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
          sim.fe_system->meshAccess());
    } catch (const std::exception&) {
      local_snapshot_current = false;
    }
    if (globalMinDouble(local_snapshot_current ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage snapshot is stale or incomplete on at least one rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshot = **found;
    const auto& velocity_record =
        sim.fe_system->fieldRecord(declaration.velocity_field);
    const auto velocity_offset =
        sim.fe_system->fieldDofOffset(declaration.velocity_field);
    const int dimension = sim.fe_system->meshAccess().dimension();
    if (!velocity_record.space || velocity_offset < 0 ||
        velocity_record.components < dimension) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage velocity field is incompatible with the free-surface declaration.");
    }
    svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator
        velocity;
    velocity.value =
        [&sim,
         &velocity_record,
         velocity_offset,
         velocity_field = declaration.velocity_field,
         stage_solution](
            svmp::FE::GlobalIndex cell,
            const std::array<svmp::FE::Real, 3>& reference_point,
            const svmp::FE::geometry::CutQuadratureProvenance&) {
          const auto cell_dofs =
              sim.fe_system->fieldDofHandler(velocity_field).getCellDofs(cell);
          std::vector<svmp::FE::Real> coefficients;
          coefficients.reserve(cell_dofs.size());
          for (const auto dof : cell_dofs) {
            if (dof < 0) {
              throw std::runtime_error(
                  "[svMultiPhysics::Application] Accepted contact-stage velocity cell has a negative DOF.");
            }
            const auto index = static_cast<std::size_t>(
                velocity_offset + dof);
            if (index >= stage_solution.size()) {
              throw std::runtime_error(
                  "[svMultiPhysics::Application] Accepted contact-stage solution is too small for the velocity field.");
            }
            coefficients.push_back(stage_solution[index]);
          }
          const svmp::FE::spaces::FunctionSpace::Value reference_value{
              reference_point[0], reference_point[1], reference_point[2]};
          const auto value = velocity_record.space->evaluate(
              reference_value, coefficients);
          return std::array<svmp::FE::Real, 3>{
              value[0], value[1], value[2]};
        };
    std::optional<
        svmp::FE::interfaces::FreeSurfaceDynamicContactState>
        local_state;
    try {
      local_state =
          svmp::FE::interfaces::evaluateFreeSurfaceDynamicContactState(
              snapshot, declaration.parameters, velocity);
    } catch (const std::exception&) {
      local_state.reset();
    }
    if (globalMinDouble(local_state.has_value() ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic contact-stage evaluation failed on at least one rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    auto state = std::move(*local_state);
    const double local_wall_count =
        static_cast<double>(state.walls.size());
    if (globalMinDouble(local_wall_count, comm) !=
        globalMaxDouble(local_wall_count, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic contact-stage wall ordering differs across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }

    const auto global_sum = [&comm](svmp::FE::Real local) {
      return static_cast<svmp::FE::Real>(
          globalSumDouble(static_cast<double>(local), comm));
    };
    for (auto& wall : state.walls) {
      wall.owned_quadrature_point_count = globalSumSize(
          wall.owned_quadrature_point_count, comm);
      wall.owned_advancing_point_count = globalSumSize(
          wall.owned_advancing_point_count, comm);
      wall.owned_receding_point_count = globalSumSize(
          wall.owned_receding_point_count, comm);
      wall.owned_stationary_point_count = globalSumSize(
          wall.owned_stationary_point_count, comm);
      wall.owned_contact_measure =
          global_sum(wall.owned_contact_measure);
      wall.dynamic_angle_integral =
          global_sum(wall.dynamic_angle_integral);
      wall.dynamic_cosine_integral =
          global_sum(wall.dynamic_cosine_integral);
      wall.contact_speed_integral =
          global_sum(wall.contact_speed_integral);
      wall.contact_speed_squared_integral =
          global_sum(wall.contact_speed_squared_integral);
      wall.constitutive_residual_integral =
          global_sum(wall.constitutive_residual_integral);
      wall.absolute_constitutive_residual_integral =
          global_sum(wall.absolute_constitutive_residual_integral);
      wall.line_friction_dissipation =
          global_sum(wall.line_friction_dissipation);
      wall.owned_wetted_wall_quadrature_point_count = globalSumSize(
          wall.owned_wetted_wall_quadrature_point_count, comm);
      wall.owned_wetted_wall_measure =
          global_sum(wall.owned_wetted_wall_measure);
      wall.wall_slip_speed_integral =
          global_sum(wall.wall_slip_speed_integral);
      wall.wall_slip_speed_squared_integral =
          global_sum(wall.wall_slip_speed_squared_integral);
      wall.wall_slip_dissipation =
          global_sum(wall.wall_slip_dissipation);
      for (std::size_t component = 0; component < 3u; ++component) {
        wall.wall_tangential_velocity_integral[component] =
            global_sum(
                wall.wall_tangential_velocity_integral[component]);
        wall.contact_position_integral[component] =
            global_sum(wall.contact_position_integral[component]);
        wall.wall_normal_integral[component] =
            global_sum(wall.wall_normal_integral[component]);
        wall.footprint_direction_integral[component] =
            global_sum(wall.footprint_direction_integral[component]);
        wall.contact_line_tangent_integral[component] =
            global_sum(wall.contact_line_tangent_integral[component]);
      }
    }
    svmp::FE::interfaces::finalizeFreeSurfaceDynamicContactState(state);
    stages.push_back(
        svmp::FE::systems::FreeSurfaceAcceptedContactStageState{
            .stage_time = stage_time,
            .stage_alpha_f = stage_alpha_f,
            .previous_state_revision = previous_state_revision,
            .endpoint_state_revision = endpoint_state_revision,
            .stage_state_revision = acceptedContactStageRevision(
                previous_state_revision,
                endpoint_state_revision,
                snapshot_revision,
                stage_time,
                stage_alpha_f,
                stage_solution),
            .geometry_revision = snapshot.revision(),
            .state = std::move(state),
        });
  }
  return stages;
}

std::vector<svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
evaluateCurrentFreeSurfaceDiscreteFunctionals(
    application::core::SimulationComponents& sim)
{
  const auto declarations =
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations();
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const auto local_declaration_count =
      static_cast<double>(declarations.size());
  if (globalMinDouble(local_declaration_count, comm) !=
      globalMaxDouble(local_declaration_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface functional declarations differ across the FE communicator.");
  }
  if (declarations.empty()) {
    return {};
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  const auto context_available = context != nullptr ? 1.0 : 0.0;
  if (globalMinDouble(context_available, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface functional evaluation requires an authoritative cut-integration context on every rank.");
  }

  std::vector<svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
      states;
  states.reserve(declarations.size());
  for (const auto& declaration : declarations) {
    const std::array<double, 6> declaration_metadata{
        static_cast<double>(declaration.interface_marker),
        static_cast<double>(declaration.level_set_field),
        static_cast<double>(declaration.velocity_field),
        static_cast<double>(declaration.parameters.liquid_side),
        static_cast<double>(declaration.parameters.surface_tension),
        static_cast<double>(declaration.parameters.volume_multiplier)};
    bool declaration_metadata_consistent = true;
    for (const auto value : declaration_metadata) {
      const bool local_finite = std::isfinite(value);
      if (globalMinDouble(local_finite ? 1.0 : 0.0, comm) != 1.0) {
        declaration_metadata_consistent = false;
        continue;
      }
      declaration_metadata_consistent =
          globalMinDouble(value, comm) == globalMaxDouble(value, comm) &&
          declaration_metadata_consistent;
    }
    if (!declaration_metadata_consistent) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface functional declaration ordering or scalar parameters differ across the FE communicator.");
    }
    const bool local_marker_available =
        context->hasFreeSurfaceGeometrySnapshotForMarker(
            declaration.interface_marker);
    if (globalMinDouble(local_marker_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface functional marker " +
          std::to_string(declaration.interface_marker) +
          " is missing an authoritative geometry snapshot on at least one rank.");
    }
    const auto snapshot_revision =
        context->freeSurfaceGeometrySnapshotRevisionForMarker(
            declaration.interface_marker);
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     snapshot_revision;
        });
    const auto [minimum_revision, maximum_revision] =
        globalMinMaxUint64(snapshot_revision, comm);
    if (minimum_revision != maximum_revision) {
      std::ostringstream diagnostic;
      diagnostic
          << "[svMultiPhysics::Application] Free-surface geometry revision differs across the FE communicator for marker "
          << declaration.interface_marker
          << ": local=" << snapshot_revision
          << " minimum=" << minimum_revision
          << " maximum=" << maximum_revision;
      if (found != snapshots.end()) {
        const auto& revision = (*found)->revision();
        diagnostic
            << " source_layout=" << revision.source_layout_revision
            << " source_value=" << revision.source_value_revision
            << " mesh_geometry=" << revision.mesh_geometry_revision
            << " mesh_topology=" << revision.mesh_topology_revision
            << " ownership=" << revision.ownership_revision
            << " numbering=" << revision.numbering_revision
            << " quadrature_policy=" << revision.quadrature_policy_key;
      }
      diagnostic << ".";
      throw std::runtime_error(diagnostic.str());
    }
    const bool local_snapshot_available = found != snapshots.end();
    if (globalMinDouble(local_snapshot_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface geometry snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
        sim.fe_system->meshAccess());
    const auto& snapshot = **found;
    auto global_state =
        svmp::FE::interfaces::evaluateFreeSurfaceDiscreteFunctional(
            snapshot, declaration.parameters);

    const auto wall_count = static_cast<double>(global_state.walls.size());
    if (globalMinDouble(wall_count, comm) !=
        globalMaxDouble(wall_count, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface functional wall sets differ across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    for (const auto& wall : global_state.walls) {
      const auto marker = static_cast<double>(wall.boundary_marker);
      if (globalMinDouble(marker, comm) != globalMaxDouble(marker, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Free-surface functional wall ordering differs across the FE communicator for marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
      const auto angle_present =
          wall.equilibrium_contact_angle_radians.has_value() ? 1.0 : 0.0;
      if (globalMinDouble(angle_present, comm) !=
          globalMaxDouble(angle_present, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Free-surface functional wall coefficient presence differs across the FE communicator for marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
      if (wall.equilibrium_contact_angle_radians.has_value()) {
        const auto angle = static_cast<double>(
            *wall.equilibrium_contact_angle_radians);
        if (globalMinDouble(angle, comm) != globalMaxDouble(angle, comm)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Free-surface functional wall angles differ across the FE communicator for marker " +
              std::to_string(declaration.interface_marker) + ".");
        }
      }
    }

    const auto global_sum = [&comm](svmp::FE::Real local) {
      return static_cast<svmp::FE::Real>(
          globalSumDouble(static_cast<double>(local), comm));
    };
    global_state.owned_liquid_volume =
        global_sum(global_state.owned_liquid_volume);
    global_state.owned_liquid_gas_area =
        global_sum(global_state.owned_liquid_gas_area);
    global_state.owned_wetted_wall_area =
        global_sum(global_state.owned_wetted_wall_area);
    global_state.owned_contact_measure =
        global_sum(global_state.owned_contact_measure);
    global_state.liquid_gas_surface_energy =
        global_sum(global_state.liquid_gas_surface_energy);
    global_state.young_wall_energy =
        global_sum(global_state.young_wall_energy);
    global_state.volume_constraint_potential =
        global_sum(global_state.volume_constraint_potential);
    global_state.total_potential =
        global_sum(global_state.total_potential);
    for (auto& wall : global_state.walls) {
      wall.owned_wetted_wall_area =
          global_sum(wall.owned_wetted_wall_area);
      wall.owned_contact_measure =
          global_sum(wall.owned_contact_measure);
      wall.young_wall_energy = global_sum(wall.young_wall_energy);
    }
    states.push_back(
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState{
            .interface_marker = declaration.interface_marker,
            .geometry_revision = snapshot.revision(),
            .state = std::move(global_state),
            .contact_stage = std::nullopt,
        });
  }
  return states;
}

std::uint64_t levelSetMaintenanceCutTopologyRevision(
    const svmp::FE::interfaces::FreeSurfaceGeometrySnapshot& snapshot,
    const svmp::MeshComm& comm)
{
  std::vector<std::uint64_t> local_rule_revisions;
  local_rule_revisions.reserve(snapshot.rules().size());
  for (const auto& rule : snapshot.rules()) {
    if (!rule.locally_owned) {
      continue;
    }
    std::uint64_t revision = kCutContextHashOffset;
    mixCutContextHash(
        revision, static_cast<std::uint64_t>(rule.role));
    mixCutContextHash(
        revision, static_cast<std::uint64_t>(rule.retention));
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(rule.physical_boundary_marker)));
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(
            rule.reference_rule.geometric_dimension));
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(
            rule.reference_rule.provenance.parent_entity_global_id));
    mixCutContextHash(
        revision,
        rule.reference_rule.provenance.cut_topology_revision);
    mixCutContextHash(
        revision,
        rule.physical_rule.cut_topology_revision);
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(rule.component_id));
    mixCutContextHash(revision, rule.topology_id);
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(
            rule.source_fragment_stable_ids.size()));
    for (const auto stable_id : rule.source_fragment_stable_ids) {
      mixCutContextHash(revision, stable_id);
    }
    local_rule_revisions.push_back(
        revision == 0u ? 1u : revision);
  }

  const auto collective = snapshotOwnershipCollective(comm);
  auto global_rule_revisions =
      collective.all_gather_owned_rule_identity_values(
          local_rule_revisions);
  std::sort(
      global_rule_revisions.begin(), global_rule_revisions.end());
  std::uint64_t revision = kCutContextHashOffset;
  mixCutContextHash(
      revision,
      static_cast<std::uint64_t>(
          snapshot.revision().interface_marker));
  mixCutContextHash(
      revision,
      static_cast<std::uint64_t>(global_rule_revisions.size()));
  for (const auto rule_revision : global_rule_revisions) {
    mixCutContextHash(revision, rule_revision);
  }
  return revision == 0u ? 1u : revision;
}

std::vector<application::core::LevelSetAuthoritativeFunctionalValue>
levelSetMaintenanceFunctionalValues(
    application::core::SimulationComponents& sim,
    std::span<const svmp::FE::systems::
                        AcceptedFreeSurfaceDiscreteFunctionalState> states)
{
  std::vector<application::core::LevelSetAuthoritativeFunctionalValue>
      values;
  values.reserve(states.size());
  if (states.empty()) {
    return values;
  }
  if (!sim.fe_system || !sim.fe_system->cutIntegrationContext()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance work provenance requires an authoritative cut-integration context.");
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  const auto& snapshots = context->freeSurfaceGeometrySnapshots();
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  for (const auto& state : states) {
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& snapshot) {
          return snapshot &&
                 snapshot->revision().snapshot_revision_key ==
                     state.geometry_revision.snapshot_revision_key;
        });
    if (found == snapshots.end()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set maintenance work could not resolve its authoritative geometry snapshot.");
    }
    values.push_back(
        application::core::LevelSetAuthoritativeFunctionalValue{
            .interface_marker = state.interface_marker,
            .snapshot_revision =
                state.geometry_revision.snapshot_revision_key,
            .mesh_topology_revision =
                state.geometry_revision.mesh_topology_revision,
            .cut_topology_revision =
                levelSetMaintenanceCutTopologyRevision(
                    **found, comm),
            .liquid_volume = state.state.owned_liquid_volume,
            .liquid_gas_area = state.state.owned_liquid_gas_area,
            .wetted_wall_area = state.state.owned_wetted_wall_area,
            .contact_measure = state.state.owned_contact_measure,
            .surface_energy = state.state.liquid_gas_surface_energy,
            .young_wall_energy = state.state.young_wall_energy,
            .volume_constraint_potential =
                state.state.volume_constraint_potential,
            .total_potential = state.state.total_potential,
        });
  }
  std::sort(
      values.begin(), values.end(), [](const auto& left, const auto& right) {
        return left.interface_marker < right.interface_marker;
      });
  return values;
}

std::uint64_t levelSetMaintenanceAlgebraicRevision(
    std::span<const svmp::FE::Real> state) noexcept
{
  std::uint64_t revision = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto mix = [&](std::uint64_t value) {
    for (std::size_t byte = 0u; byte < sizeof(value); ++byte) {
      revision ^= (value >> (byte * 8u)) & 0xffu;
      revision *= prime;
    }
  };
  mix(static_cast<std::uint64_t>(state.size()));
  for (const auto value : state) {
    std::uint64_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    mix(bits);
  }
  return revision == 0u ? 1u : revision;
}

std::uint64_t collectiveLevelSetMaintenanceAlgebraicRevision(
    std::span<const svmp::FE::Real> state,
    const svmp::MeshComm& comm)
{
  const auto revision = levelSetMaintenanceAlgebraicRevision(state);
  const auto [minimum, maximum] =
      globalMinMaxUint64(revision, comm);
  if (minimum != maximum) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance algebraic state revision differs across the FE communicator.");
  }
  return revision;
}

const char* levelSetMaintenanceWorkStatusName(
    application::core::LevelSetMaintenanceWorkStatus status) noexcept
{
  using Status = application::core::LevelSetMaintenanceWorkStatus;
  switch (status) {
    case Status::Trial:
      return "trial";
    case Status::Accepted:
      return "accepted";
    case Status::Rejected:
      return "rejected";
  }
  return "unknown";
}

const char* levelSetMaintenanceWorkSubstageName(
    application::core::LevelSetMaintenanceWorkSubstage substage) noexcept
{
  using Substage = application::core::LevelSetMaintenanceWorkSubstage;
  switch (substage) {
    case Substage::Transport:
      return "transport";
    case Substage::Limiting:
      return "limiting";
    case Substage::Reinitialization:
      return "reinitialization";
    case Substage::GeometryReconciliation:
      return "geometry_reconciliation";
    case Substage::GlobalCorrection:
      return "global_correction";
  }
  return "unknown";
}

const char* levelSetMaintenanceDeclaredStageName(
    application::core::LevelSetMaintenanceDeclaredStage stage) noexcept
{
  using Stage = application::core::LevelSetMaintenanceDeclaredStage;
  switch (stage) {
    case Stage::ProspectiveAcceptedEndpoint:
      return "prospective_accepted_endpoint";
    case Stage::AcceptedEndpointPostStep:
      return "accepted_endpoint_post_step";
  }
  return "unknown";
}

void logLevelSetMaintenanceWorkRows(
    std::span<const application::core::LevelSetMaintenanceWorkRow> rows)
{
  for (const auto& row : rows) {
    std::ostringstream log;
    log << std::setprecision(17)
        << "[svMultiPhysics::Application] Level-set maintenance work row"
        << " diagnostic=level_set_maintenance_work"
        << " transaction_id=" << row.transaction_id
        << " status=" << levelSetMaintenanceWorkStatusName(row.status)
        << " substage="
        << levelSetMaintenanceWorkSubstageName(row.substage)
        << " step=" << row.step
        << " attempt=" << row.attempt
        << " time=" << row.time
        << " dt=" << row.dt
        << " declared_stage="
        << levelSetMaintenanceDeclaredStageName(row.declared_stage)
        << " algebraic_state_revision_before="
        << row.algebraic_state_revision_before
        << " algebraic_state_revision_after="
        << row.algebraic_state_revision_after
        << " snapshot_set_revision_before="
        << row.snapshot_set_revision_before
        << " snapshot_set_revision_after="
        << row.snapshot_set_revision_after
        << " mesh_topology_set_revision_before="
        << row.mesh_topology_set_revision_before
        << " mesh_topology_set_revision_after="
        << row.mesh_topology_set_revision_after
        << " cut_topology_set_revision_before="
        << row.cut_topology_set_revision_before
        << " cut_topology_set_revision_after="
        << row.cut_topology_set_revision_after
        << " extension_map_revision_before="
        << row.extension_map_revision_before.value_or(0u)
        << " extension_map_revision_after="
        << row.extension_map_revision_after.value_or(0u)
        << " authoritative_functional_count=" << row.before.size()
        << " numerical_work_sign=potential_after_minus_before"
        << " numerical_work=" << row.numerical_work
        << " accepted_numerical_work="
        << row.accepted_numerical_work
        << " account_scope=maintenance_surface_wall_volume_only"
        << " complete_time_discrete_energy_identity=false";
    for (std::size_t index = 0u; index < row.before.size(); ++index) {
      const auto& before = row.before[index];
      const auto& after = row.after[index];
      log << " interface_marker=" << before.interface_marker
          << " before_snapshot_revision="
          << before.snapshot_revision
          << " after_snapshot_revision=" << after.snapshot_revision
          << " before_mesh_topology_revision="
          << before.mesh_topology_revision
          << " after_mesh_topology_revision="
          << after.mesh_topology_revision
          << " before_cut_topology_revision="
          << before.cut_topology_revision
          << " after_cut_topology_revision="
          << after.cut_topology_revision
          << " before_liquid_volume=" << before.liquid_volume
          << " after_liquid_volume=" << after.liquid_volume
          << " before_liquid_gas_area=" << before.liquid_gas_area
          << " after_liquid_gas_area=" << after.liquid_gas_area
          << " before_wetted_wall_area=" << before.wetted_wall_area
          << " after_wetted_wall_area=" << after.wetted_wall_area
          << " before_contact_measure=" << before.contact_measure
          << " after_contact_measure=" << after.contact_measure
          << " before_surface_energy=" << before.surface_energy
          << " after_surface_energy=" << after.surface_energy
          << " before_young_wall_energy="
          << before.young_wall_energy
          << " after_young_wall_energy=" << after.young_wall_energy
          << " before_volume_constraint_potential="
          << before.volume_constraint_potential
          << " after_volume_constraint_potential="
          << after.volume_constraint_potential
          << " before_total_potential=" << before.total_potential
          << " after_total_potential=" << after.total_potential;
    }
    application::core::oopCout() << log.str() << std::endl;
  }
}

void logLevelSetMaintenanceWorkAttempts(
    std::span<const
        application::core::LevelSetMaintenanceWorkAttempt> attempts)
{
  for (const auto& attempt : attempts) {
    std::ostringstream log;
    log << std::setprecision(17)
        << "[svMultiPhysics::Application] Level-set maintenance work attempt"
        << " diagnostic=level_set_maintenance_work_attempt"
        << " transaction_id=" << attempt.transaction_id
        << " status="
        << levelSetMaintenanceWorkStatusName(attempt.status)
        << " step=" << attempt.step
        << " attempt=" << attempt.attempt
        << " time=" << attempt.time
        << " dt=" << attempt.dt
        << " declared_stage="
        << levelSetMaintenanceDeclaredStageName(
               attempt.declared_stage)
        << " extension_map_revision="
        << attempt.extension_map_revision.value_or(0u)
        << " row_count=" << attempt.row_count
        << " numerical_work=" << attempt.numerical_work
        << " accepted_numerical_work="
        << attempt.accepted_numerical_work
        << " accepted_account_publication="
        << (attempt.status ==
                    application::core::
                        LevelSetMaintenanceWorkStatus::Accepted
                ? "accepted_rows"
                : "zero_for_rejection")
        << " complete_time_discrete_energy_identity=false";
    application::core::oopCout() << log.str() << std::endl;
  }
}

void recordAcceptedFreeSurfaceDiscreteFunctionals(
    application::core::SimulationComponents& sim,
    std::uint64_t accepted_step,
    svmp::FE::Real accepted_time,
    svmp::FE::Real dt,
    std::uint64_t state_revision,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        contact_stages = {})
{
  const auto declarations =
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations();
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const auto local_declaration_count =
      static_cast<double>(declarations.size());
  if (globalMinDouble(local_declaration_count, comm) !=
      globalMaxDouble(local_declaration_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted free-surface functional declarations differ across the FE communicator.");
  }
  if (declarations.empty()) {
    if (!contact_stages.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Contact-stage states were supplied without free-surface functional declarations.");
    }
    return;
  }
  const double local_contact_stage_count =
      static_cast<double>(contact_stages.size());
  if (globalMinDouble(local_contact_stage_count, comm) !=
      globalMaxDouble(local_contact_stage_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage coverage differs across the FE communicator.");
  }
  std::map<
      int,
      const svmp::FE::systems::FreeSurfaceAcceptedContactStageState*>
      contact_stage_by_marker;
  bool local_contact_stages_valid = true;
  for (const auto& stage : contact_stages) {
    const int marker = stage.geometry_revision.interface_marker;
    const double local_marker = static_cast<double>(marker);
    if (globalMinDouble(local_marker, comm) !=
        globalMaxDouble(local_marker, comm)) {
      local_contact_stages_valid = false;
    }
    local_contact_stages_valid =
        marker >= 0 &&
        contact_stage_by_marker.emplace(marker, &stage).second &&
        local_contact_stages_valid;
  }
  if (globalMinDouble(local_contact_stages_valid ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage states must have communicator-consistent unique nonnegative interface markers.");
  }
  auto accepted_states = evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
  if (accepted_states.size() != declarations.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted free-surface functional evaluation returned incomplete declaration coverage.");
  }
  for (std::size_t index = 0; index < declarations.size(); ++index) {
    const auto& declaration = declarations[index];
    auto& accepted_state = accepted_states[index];
    if (accepted_state.interface_marker != declaration.interface_marker) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted free-surface functional evaluation changed declaration order.");
    }
    const auto contact_stage =
        contact_stage_by_marker.find(declaration.interface_marker);
    const bool dynamic_contact_declared =
        !declaration.parameters.dynamic_contact_coefficients.empty();
    if (dynamic_contact_declared !=
        (contact_stage != contact_stage_by_marker.end())) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage coverage does not match the free-surface declaration for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    if (contact_stage != contact_stage_by_marker.end()) {
      accepted_state.contact_stage = *contact_stage->second;
    }
  }

  const auto declared_dynamic_count = std::count_if(
      declarations.begin(), declarations.end(), [](const auto& declaration) {
        return !declaration.parameters.dynamic_contact_coefficients.empty();
      });
  if (contact_stage_by_marker.size() !=
      static_cast<std::size_t>(declared_dynamic_count)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage states include an undeclared interface marker.");
  }

  sim.fe_system->recordAcceptedFreeSurfaceDiscreteFunctionals(
      accepted_step,
      accepted_time,
      dt,
      state_revision,
      accepted_states);
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

svmp::FE::level_set::LevelSetVolumeOptions levelSetVolumeOptionsForMaintenance(
    const LevelSetMaintenanceRequest& request)
{
  svmp::FE::level_set::LevelSetVolumeOptions options{};
  options.isovalue = static_cast<svmp::FE::Real>(request.isovalue);
  if (!request.volume_cut_request.has_value()) {
    return options;
  }

  const auto& cut_request = *request.volume_cut_request;
  if (cut_request.geometry_mode !=
      svmp::FE::level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit) {
    return options;
  }

  options.use_generated_interface_quadrature = true;
  options.level_set_field_name = request.level_set_field_name;
  options.generated_domain_id =
      cut_request.domain_id.empty()
          ? std::string{"volume_correction"}
          : cut_request.domain_id + "_volume_correction";
  options.requested_interface_marker = cut_request.requested_interface_marker;
  options.quadrature_order = cut_request.quadrature_order;
  options.interface_quadrature_order = cut_request.interface_quadrature_order;
  options.volume_quadrature_order = cut_request.volume_quadrature_order;
  options.geometry_mode = cut_request.geometry_mode;
  options.implicit_cut_quadrature_backend = cut_request.implicit_cut_backend;
  options.implicit_cut_fallback_policy =
      cut_request.implicit_cut_fallback_policy;
  options.geometry_tangent_policy = cut_request.geometry_tangent_policy;
  options.implicit_cut_root_tolerance =
      static_cast<svmp::FE::Real>(
          cut_request.implicit_cut_root_tolerance);
  options.implicit_cut_root_coordinate_tolerance =
      static_cast<svmp::FE::Real>(
          cut_request.implicit_cut_root_coordinate_tolerance);
  options.implicit_cut_root_max_iterations =
      cut_request.implicit_cut_root_max_iterations;
  options.implicit_cut_max_subdivision_depth =
      cut_request.implicit_cut_max_subdivision_depth;
  options.affected_cell_neighborhood_layers =
      cut_request.affected_cell_neighborhood_layers;
  options.allow_corner_linearized_geometry =
      cut_request.allow_corner_linearized_geometry;
  options.require_production_qualified_implicit_cut_backend =
      cut_request.require_production_qualified_implicit_cut_backend;
  return options;
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

    const auto volume_options =
        levelSetVolumeOptionsForMaintenance(request);
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
        << request.volume_target
        << " volume_measure_source="
        << (volume_options.use_generated_interface_quadrature
                ? "generated_interface_quadrature"
                : "corner_linearized")
        << std::endl;
  }
}

svmp::FE::level_set::LevelSetBoundaryOptions
resolveLevelSetOpenBoundaries(
    const application::core::SimulationComponents& sim,
    const LevelSetMaintenanceRequest& request)
{
  svmp::FE::level_set::LevelSetBoundaryOptions boundaries;
  if (request.open_boundaries.empty()) {
    return boundaries;
  }
  if (!sim.primary_mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Bound-preserving level-set transport "
        "cannot resolve open boundary names without the active mesh.");
  }
  for (const auto& open : request.open_boundaries) {
    const auto marker = sim.primary_mesh->label_from_name(open.face_name);
    if (marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Bound-preserving level-set transport "
          "could not resolve open boundary face '" + open.face_name + "'.");
    }
    if (open.inflow) {
      if (!open.literal_inflow_value.has_value() ||
          !std::isfinite(*open.literal_inflow_value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Bound-preserving level-set transport "
            "currently supports finite literal LevelSetInflow data only; face='" +
            open.face_name + "'.");
      }
      boundaries.inflow.push_back(
          svmp::FE::level_set::LevelSetInflowBoundary{
              .boundary_marker = marker,
              .value = *open.literal_inflow_value});
    } else {
      boundaries.outflow.push_back(
          svmp::FE::level_set::LevelSetOutflowBoundary{
              .boundary_marker = marker});
    }
  }
  return boundaries;
}

struct LevelSetBoundPreservingApplicationResult {
  bool accept_step{true};
  bool changed{false};
};

using LevelSetMaintenanceStageObserver = std::function<void(
    application::core::LevelSetMaintenanceWorkSubstage,
    std::span<const svmp::FE::Real>,
    std::span<const svmp::FE::Real>)>;

LevelSetBoundPreservingApplicationResult
applyLevelSetBoundPreservingCandidates(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    const std::vector<LevelSetMaintenanceRequest>& requests,
    double /*generalized_alpha_gamma*/,
    const LevelSetMaintenanceStageObserver& observe_stage = {})
{
  LevelSetBoundPreservingApplicationResult application_result;
  if (!sim.fe_system || requests.empty()) {
    return application_result;
  }

  const bool any_enabled = std::any_of(
      requests.begin(), requests.end(), [](const auto& request) {
        return request.bound_preserving.enabled;
      });
  if (!any_enabled) {
    return application_result;
  }
  if (!(history.dt() > 0.0) || !std::isfinite(history.dt())) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Bound-preserving level-set transport "
        "requires a positive finite candidate time step.");
  }

  const auto previous_solution = gatherFeOrderedSolution(history.uPrev());
  const auto raw_candidate = gatherFeOrderedSolution(history.u());
  auto candidate = raw_candidate;

  for (const auto& request : requests) {
    if (!request.bound_preserving.enabled) {
      continue;
    }
    const auto field =
        sim.fe_system->findFieldByName(request.level_set_field_name);
    if (field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Bound-preserving level-set transport "
          "could not find field '" + request.level_set_field_name + "'.");
    }
    const auto boundaries = resolveLevelSetOpenBoundaries(sim, request);
    svmp::FE::systems::SystemStateView state;
    state.time = history.time() + history.dt();
    state.dt = history.dt();
    state.dt_prev = history.dtPrev();
    state.u = std::span<const svmp::FE::Real>(candidate);
    state.u_prev = std::span<const svmp::FE::Real>(previous_solution);

    const auto safety =
        svmp::FE::level_set::evaluateLevelSetTransportSafety(
            *sim.fe_system,
            request.velocity,
            boundaries,
            request.bound_preserving,
            state,
            static_cast<svmp::FE::Real>(history.dt()));
    const bool wall_violation =
        request.bound_preserving.enforce_impermeable_boundaries &&
        safety.maximum_boundary_normal_velocity_ratio >
            request.bound_preserving.impermeable_normal_velocity_tolerance;
    const bool courant_violation =
        request.bound_preserving.enforce_courant_limit &&
        safety.maximum_courant >
            request.bound_preserving.maximum_courant +
                request.bound_preserving.courant_tolerance;
    if (wall_violation) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Bound-preserving level-set transport "
          "rejected an incompatible impermeable-wall velocity for field '" +
          request.level_set_field_name + "': marker=" +
          std::to_string(safety.worst_boundary_marker) +
          " normal_velocity=" +
          std::to_string(safety.maximum_boundary_normal_velocity) +
          " normalized_flux=" +
          std::to_string(safety.maximum_boundary_normal_velocity_ratio) +
          ". " + safety.diagnostic);
    }
    if (courant_violation) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] Level-set candidate rejected"
          << " field='" << request.level_set_field_name << "'"
          << " reason=bound_preserving_courant_contract"
          << " courant=" << safety.maximum_courant
          << " maximum_courant="
          << request.bound_preserving.maximum_courant
          << " minimum_cell_length=" << safety.minimum_cell_length
          << " maximum_speed=" << safety.maximum_speed
          << " dt=" << history.dt() << std::endl;
      application_result.accept_step = false;
      return application_result;
    }
    if (!safety.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set transport safety check "
          "failed for field '" + request.level_set_field_name + "': " +
          safety.diagnostic);
    }

    std::vector<svmp::FE::Real> limited;
    const auto limiter =
        svmp::FE::level_set::applyLevelSetBoundPreservingLimiter(
            *sim.fe_system,
            field,
            boundaries,
            request.bound_preserving,
            std::span<const svmp::FE::Real>(previous_solution),
            std::span<const svmp::FE::Real>(candidate),
            safety.maximum_courant,
            limited);
    if (!limiter.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Bound-preserving level-set limiter "
          "failed for field '" + request.level_set_field_name + "': " +
          limiter.diagnostic);
    }
    application::core::oopCout()
        << "[svMultiPhysics::Application] Level-set bound-preserving gate"
        << " field='" << request.level_set_field_name << "'"
        << " step=" << history.stepIndex() + 1
        << " courant=" << safety.maximum_courant
        << " wall_normal_ratio="
        << safety.maximum_boundary_normal_velocity_ratio
        << " limited_dofs=" << limiter.limited_dofs
        << " max_unrelaxed_violation="
        << limiter.maximum_unrelaxed_bound_violation
        << " max_correction=" << limiter.maximum_correction
        << " positive_sign_flips_prevented="
        << limiter.positive_patch_sign_flips_prevented
        << " negative_sign_flips_prevented="
        << limiter.negative_patch_sign_flips_prevented
        << " conservative=false" << std::endl;
    if (limiter.applied) {
      if (observe_stage) {
        observe_stage(
            application::core::LevelSetMaintenanceWorkSubstage::
                Limiting,
            candidate,
            limited);
      }
      // Mutating a converged Newton candidate after its final residual check
      // would accept a state that was never a solution of the coupled
      // transport/capillary operator.  Reject without touching history so an
      // adaptive TimeLoop can reduce dt and solve again.  Fixed-step runs fail
      // closed if they cannot produce an in-bounds nonlinear solution.
      application::core::oopCout()
          << "[svMultiPhysics::Application] Level-set candidate rejected"
          << " field='" << request.level_set_field_name << "'"
          << " reason=bound_preserving_limiter_requires_nonlinear_retry"
          << " limited_dofs=" << limiter.limited_dofs
          << " max_unrelaxed_violation="
          << limiter.maximum_unrelaxed_bound_violation
          << " max_correction=" << limiter.maximum_correction
          << " dt=" << history.dt() << std::endl;
      application_result.accept_step = false;
      application_result.changed = true;
      return application_result;
    }
  }
  return application_result;
}

void accountAppliedLevelSetVolumeCorrection(
    LevelSetMaintenanceRequest& request,
    const svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult& result)
{
  if (!result.correction_applied) {
    return;
  }
  const auto finite_nonnegative = [](svmp::FE::Real value) {
    return std::isfinite(value) && value >= svmp::FE::Real{0.0};
  };
  if (!(result.minimum_edge_length > svmp::FE::Real{0.0}) ||
      !std::isfinite(result.minimum_edge_length) ||
      !finite_nonnegative(result.max_interface_displacement) ||
      !finite_nonnegative(result.max_contact_line_displacement) ||
      !finite_nonnegative(result.max_contact_angle_change_radians) ||
      !result.negative_component_topology_preserved ||
      result.negative_component_volume_transfers.empty() ||
      !std::isfinite(result.total_component_volume_transfer) ||
      !finite_nonnegative(
          result.total_absolute_component_volume_transfer) ||
      !finite_nonnegative(
          result.maximum_absolute_component_volume_transfer)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Applied level-set volume correction "
        "reported invalid geometry or component-transfer metrics.");
  }

  auto reference_edge_length =
      request.volume_correction_reference_minimum_edge_length;
  if (!(reference_edge_length > svmp::FE::Real{0.0}) ||
      !std::isfinite(reference_edge_length)) {
    reference_edge_length = result.minimum_edge_length;
  } else {
    reference_edge_length =
        std::min(reference_edge_length, result.minimum_edge_length);
  }
  const auto allowed_cumulative_displacement =
      request.volume_correction
          .maximum_cumulative_interface_displacement_fraction *
      reference_edge_length;
  const auto next_interface_displacement =
      request.cumulative_volume_correction_interface_displacement +
      result.max_interface_displacement;
  const auto next_contact_line_displacement =
      request.cumulative_volume_correction_contact_line_displacement +
      result.max_contact_line_displacement;
  const auto prospective_budget_consumption =
      std::max(next_interface_displacement, next_contact_line_displacement);
  if (!finite_nonnegative(allowed_cumulative_displacement) ||
      !finite_nonnegative(next_interface_displacement) ||
      !finite_nonnegative(next_contact_line_displacement) ||
      !finite_nonnegative(prospective_budget_consumption)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set volume correction "
        "cumulative-displacement budget is invalid or overflowed.");
  }
  const auto comparison_tolerance =
      svmp::FE::Real{64.0} * std::numeric_limits<svmp::FE::Real>::epsilon() *
      std::max({allowed_cumulative_displacement,
                prospective_budget_consumption,
                reference_edge_length});
  if (prospective_budget_consumption >
      allowed_cumulative_displacement + comparison_tolerance) {
    const auto* limiting_path =
        next_contact_line_displacement > next_interface_displacement
            ? "contact_line"
            : "interface";
    std::ostringstream message;
    message << std::setprecision(17)
            << "[svMultiPhysics::Application] Level-set volume correction "
               "would exceed the cumulative interface/contact-line "
               "displacement budget for field '"
            << request.level_set_field_name
            << "': previous_interface="
            << request.cumulative_volume_correction_interface_displacement
            << " event_interface=" << result.max_interface_displacement
            << " prospective_interface=" << next_interface_displacement
            << " previous_contact_line="
            << request.cumulative_volume_correction_contact_line_displacement
            << " event_contact_line=" << result.max_contact_line_displacement
            << " prospective_contact_line="
            << next_contact_line_displacement
            << " limiting_path=" << limiting_path
            << " prospective_budget_consumption="
            << prospective_budget_consumption
            << " allowed=" << allowed_cumulative_displacement
            << " reference_minimum_edge_length=" << reference_edge_length
            << " maximum_cumulative_fraction="
            << request.volume_correction
                   .maximum_cumulative_interface_displacement_fraction;
    throw std::runtime_error(message.str());
  }

  request.volume_correction_reference_minimum_edge_length =
      reference_edge_length;
  request.cumulative_volume_correction_interface_displacement =
      next_interface_displacement;
  request.cumulative_volume_correction_contact_line_displacement =
      next_contact_line_displacement;
}

svmp::FE::level_set::LevelSetWallContactConstraint
makeAcceptedSnapshotWallConstraint(
    const svmp::FE::interfaces::FreeSurfaceGeometryRuleRecord& record,
    svmp::FE::level_set::LevelSetWallContactConstraintKind kind,
    int interface_marker,
    std::uint64_t snapshot_revision,
    const svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters&
        parameters,
    int dimension)
{
  using Constraint =
      svmp::FE::level_set::LevelSetWallContactConstraint;
  using Kind =
      svmp::FE::level_set::LevelSetWallContactConstraintKind;
  using Point = std::array<svmp::FE::Real, 3>;

  const auto fail = [&](std::string_view reason) -> Constraint {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact geometry cannot "
        "provide a prescribed level-set frame for interface marker " +
        std::to_string(interface_marker) + ", boundary marker " +
        std::to_string(record.physical_boundary_marker) + ": " +
        std::string(reason) + ".");
  };
  const auto parent =
      record.reference_rule.provenance.parent_entity_global_id;
  if (record.role !=
          svmp::FE::interfaces::FreeSurfaceGeometryRuleRole::Contact ||
      record.retention !=
          svmp::FE::interfaces::FreeSurfaceGeometryRetention::Retained ||
      parent == svmp::FE::INVALID_GLOBAL_INDEX || snapshot_revision == 0u ||
      record.reference_rule.provenance.free_surface_snapshot_revision_key !=
          snapshot_revision ||
      record.physical_rule.free_surface_snapshot_revision_key !=
          snapshot_revision) {
    return fail("the retained rule has incomplete or stale provenance");
  }

  Constraint constraint{
      .kind = kind,
      .interface_marker = interface_marker,
      .boundary_marker = record.physical_boundary_marker,
      .parent_cell_global_id = parent,
      .geometry_revision = snapshot_revision,
  };
  if (kind != Kind::PrescribedAngle) {
    return constraint;
  }
  if (dimension != 2 && dimension != 3) {
    return fail("the mesh dimension is not two or three");
  }

  const auto coefficient = std::find_if(
      parameters.young_wall_coefficients.begin(),
      parameters.young_wall_coefficients.end(),
      [&](const auto& candidate) {
        return candidate.boundary_marker ==
               record.physical_boundary_marker;
      });
  if (coefficient == parameters.young_wall_coefficients.end()) {
    return fail("the prescribed Young datum is unavailable");
  }
  const auto pi = std::acos(svmp::FE::Real{-1.0});
  const auto physical_angle =
      coefficient->equilibrium_contact_angle_radians;
  if (!std::isfinite(physical_angle) ||
      !(physical_angle > svmp::FE::Real{0.0}) ||
      !(physical_angle < pi)) {
    return fail("the prescribed Young angle is invalid");
  }
  if (parameters.liquid_side ==
      svmp::FE::geometry::CutIntegrationSide::Negative) {
    constraint.target_angle_radians = physical_angle;
  } else if (parameters.liquid_side ==
             svmp::FE::geometry::CutIntegrationSide::Positive) {
    // The snapshot normal is grad(phi)/|grad(phi)|.  The positive-side
    // liquid normal reverses it, so the equivalent level-set angle is the
    // supplement of the through-liquid Young angle.
    constraint.target_angle_radians = pi - physical_angle;
  } else {
    return fail("the declaration has no physical liquid side");
  }

  if (record.physical_rule.geometric_dimension != dimension - 2 ||
      record.physical_rule.points.empty() ||
      record.physical_rule.points.size() !=
          record.reference_rule.points.size() ||
      !std::isfinite(record.physical_rule.physical_measure) ||
      !(record.physical_rule.physical_measure > svmp::FE::Real{0.0})) {
    return fail("the accepted codimension-two rule is unavailable");
  }
  const auto finite = [](const Point& value) noexcept {
    return std::all_of(value.begin(), value.end(), [](const auto component) {
      return std::isfinite(component);
    });
  };
  const auto dot = [](const Point& left, const Point& right) noexcept {
    return left[0] * right[0] + left[1] * right[1] +
           left[2] * right[2];
  };
  const auto norm = [&](const Point& value) noexcept {
    return std::sqrt(dot(value, value));
  };
  const auto cross = [](const Point& left, const Point& right) noexcept {
    return Point{{left[1] * right[2] - left[2] * right[1],
                  left[2] * right[0] - left[0] * right[2],
                  left[0] * right[1] - left[1] * right[0]}};
  };
  const auto normalized = [&](Point value) -> std::optional<Point> {
    const auto magnitude = norm(value);
    if (!finite(value) || !std::isfinite(magnitude) ||
        !(magnitude > svmp::FE::Real{1.0e-14})) {
      return std::nullopt;
    }
    for (auto& component : value) {
      component /= magnitude;
    }
    return value;
  };

  struct FrameSample {
    Point point{};
    Point wall_normal{};
    Point phi_wall_conormal{};
    Point line_tangent{};
  };
  std::vector<FrameSample> samples;
  samples.reserve(record.physical_rule.points.size());
  Point point_integral{};
  Point wall_normal_integral{};
  Point line_tangent_integral{};
  svmp::FE::Real measure = 0.0;
  for (const auto& point : record.physical_rule.points) {
    if (!finite(point.physical_point) ||
        !std::isfinite(point.physical_weight) ||
        !(point.physical_weight > svmp::FE::Real{0.0})) {
      return fail("a physical contact point or weight is invalid");
    }
    const auto phi_normal = normalized(point.normal);
    const auto wall_normal = normalized(point.boundary_normal);
    if (!phi_normal || !wall_normal) {
      return fail("an accepted interface or wall normal is unavailable");
    }
    Point phi_wall_conormal{};
    const auto normal_component = dot(*phi_normal, *wall_normal);
    for (std::size_t component = 0u; component < 3u; ++component) {
      phi_wall_conormal[component] =
          (*phi_normal)[component] -
          normal_component * (*wall_normal)[component];
    }
    const auto unit_phi_wall_conormal = normalized(phi_wall_conormal);
    if (!unit_phi_wall_conormal) {
      return fail("the accepted contact is not transverse to the wall");
    }
    const auto line_tangent =
        normalized(cross(*wall_normal, *unit_phi_wall_conormal));
    if (!line_tangent) {
      return fail("the oriented contact-line direction is unavailable");
    }
    if (dimension == 3) {
      const auto recorded_tangent = normalized(point.tangent);
      if (!recorded_tangent ||
          std::abs(dot(*recorded_tangent, *line_tangent)) <
              svmp::FE::Real{1.0} - svmp::FE::Real{1.0e-8}) {
        return fail("the accepted contact-line tangent is inconsistent");
      }
    }
    samples.push_back(FrameSample{
        .point = point.physical_point,
        .wall_normal = *wall_normal,
        .phi_wall_conormal = *unit_phi_wall_conormal,
        .line_tangent = *line_tangent,
    });
    measure += point.physical_weight;
    for (std::size_t component = 0u; component < 3u; ++component) {
      point_integral[component] +=
          point.physical_point[component] * point.physical_weight;
      wall_normal_integral[component] +=
          (*wall_normal)[component] * point.physical_weight;
      line_tangent_integral[component] +=
          (*line_tangent)[component] * point.physical_weight;
    }
  }
  if (!std::isfinite(measure) || !(measure > svmp::FE::Real{0.0})) {
    return fail("the accepted contact measure is invalid");
  }
  const auto measure_tolerance =
      svmp::FE::Real{4096.0} *
      std::numeric_limits<svmp::FE::Real>::epsilon() *
      std::max({svmp::FE::Real{1.0},
                measure,
                record.physical_rule.physical_measure});
  if (std::abs(measure - record.physical_rule.physical_measure) >
      measure_tolerance) {
    return fail("the accepted physical contact measure is inconsistent");
  }
  for (auto& component : point_integral) {
    component /= measure;
  }
  const auto wall_normal = normalized(wall_normal_integral);
  if (!wall_normal) {
    return fail("the accepted wall-normal frame cancels");
  }
  const auto tangent_wall_component =
      dot(line_tangent_integral, *wall_normal);
  for (std::size_t component = 0u; component < 3u; ++component) {
    line_tangent_integral[component] -=
        tangent_wall_component * (*wall_normal)[component];
  }
  const auto line_tangent = normalized(line_tangent_integral);
  if (!line_tangent) {
    return fail("the accepted contact-line frame cancels");
  }
  const auto phi_wall_conormal =
      normalized(cross(*line_tangent, *wall_normal));
  if (!phi_wall_conormal) {
    return fail("the accepted contact-line frame is degenerate");
  }

  const auto direction_tolerance = svmp::FE::Real{1.0e-8};
  const auto point_scale =
      std::max({svmp::FE::Real{1.0}, norm(point_integral), measure});
  const auto point_tolerance =
      svmp::FE::Real{1.0e-9} * point_scale;
  for (const auto& sample : samples) {
    if (dot(sample.wall_normal, *wall_normal) <
            svmp::FE::Real{1.0} - direction_tolerance ||
        dot(sample.line_tangent, *line_tangent) <
            svmp::FE::Real{1.0} - direction_tolerance ||
        dot(sample.phi_wall_conormal, *phi_wall_conormal) <
            svmp::FE::Real{1.0} - direction_tolerance) {
      return fail(
          "one affine prescribed frame cannot represent the accepted rule");
    }
    Point from_centroid{};
    for (std::size_t component = 0u; component < 3u; ++component) {
      from_centroid[component] =
          sample.point[component] - point_integral[component];
    }
    const auto along_line = dot(from_centroid, *line_tangent);
    for (std::size_t component = 0u; component < 3u; ++component) {
      from_centroid[component] -=
          along_line * (*line_tangent)[component];
    }
    if (norm(from_centroid) > point_tolerance) {
      return fail("the accepted contact points do not form one affine line");
    }
  }
  if (dimension == 2 &&
      (std::abs((*wall_normal)[2]) > direction_tolerance ||
       std::hypot((*line_tangent)[0], (*line_tangent)[1]) >
           direction_tolerance)) {
    return fail("the accepted two-dimensional frame is not planar");
  }

  constraint.physical_wall_normal = *wall_normal;
  constraint.accepted_contact_point = point_integral;
  constraint.accepted_contact_line_tangent = *line_tangent;
  return constraint;
}

[[nodiscard]] bool sameAcceptedWallConstraint(
    const svmp::FE::level_set::LevelSetWallContactConstraint& left,
    const svmp::FE::level_set::LevelSetWallContactConstraint& right) noexcept
{
  return left.kind == right.kind &&
         left.interface_marker == right.interface_marker &&
         left.boundary_marker == right.boundary_marker &&
         left.parent_cell_global_id == right.parent_cell_global_id &&
         left.geometry_revision == right.geometry_revision &&
         left.target_angle_radians == right.target_angle_radians &&
         left.physical_wall_normal == right.physical_wall_normal &&
         left.accepted_contact_point == right.accepted_contact_point &&
         left.accepted_contact_line_tangent ==
             right.accepted_contact_line_tangent;
}

[[nodiscard]] bool acceptedWallConstraintFrameIsComplete(
    const svmp::FE::level_set::LevelSetWallContactConstraint& constraint,
    const svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters&
        parameters,
    int dimension) noexcept
{
  using Kind =
      svmp::FE::level_set::LevelSetWallContactConstraintKind;
  if (constraint.kind != Kind::PrescribedAngle) {
    return true;
  }
  const auto coefficient = std::find_if(
      parameters.young_wall_coefficients.begin(),
      parameters.young_wall_coefficients.end(),
      [&](const auto& candidate) {
        return candidate.boundary_marker == constraint.boundary_marker;
      });
  if (coefficient == parameters.young_wall_coefficients.end() ||
      (dimension != 2 && dimension != 3)) {
    return false;
  }
  const auto pi = std::acos(svmp::FE::Real{-1.0});
  svmp::FE::Real expected_angle = 0.0;
  if (parameters.liquid_side ==
      svmp::FE::geometry::CutIntegrationSide::Negative) {
    expected_angle = coefficient->equilibrium_contact_angle_radians;
  } else if (parameters.liquid_side ==
             svmp::FE::geometry::CutIntegrationSide::Positive) {
    expected_angle =
        pi - coefficient->equilibrium_contact_angle_radians;
  } else {
    return false;
  }
  const auto finite = [](const auto& value) noexcept {
    return std::all_of(value.begin(), value.end(), [](const auto component) {
      return std::isfinite(component);
    });
  };
  const auto dot = [](const auto& left, const auto& right) noexcept {
    return left[0] * right[0] + left[1] * right[1] +
           left[2] * right[2];
  };
  const auto wall_norm =
      std::sqrt(dot(constraint.physical_wall_normal,
                    constraint.physical_wall_normal));
  const auto tangent_norm =
      std::sqrt(dot(constraint.accepted_contact_line_tangent,
                    constraint.accepted_contact_line_tangent));
  const auto relative_alignment =
      wall_norm > svmp::FE::Real{0.0} &&
              tangent_norm > svmp::FE::Real{0.0}
          ? std::abs(dot(constraint.physical_wall_normal,
                         constraint.accepted_contact_line_tangent)) /
                (wall_norm * tangent_norm)
          : std::numeric_limits<svmp::FE::Real>::infinity();
  const auto frame_tolerance = svmp::FE::Real{1.0e-10};
  return std::isfinite(expected_angle) &&
         constraint.target_angle_radians == expected_angle &&
         finite(constraint.physical_wall_normal) &&
         finite(constraint.accepted_contact_point) &&
         finite(constraint.accepted_contact_line_tangent) &&
         std::isfinite(wall_norm) &&
         wall_norm > svmp::FE::Real{0.0} &&
         std::isfinite(tangent_norm) &&
         tangent_norm > svmp::FE::Real{0.0} &&
         relative_alignment <= frame_tolerance &&
         (dimension != 2 ||
          (std::abs(constraint.physical_wall_normal[2]) <=
               frame_tolerance * wall_norm &&
           std::hypot(constraint.accepted_contact_line_tangent[0],
                      constraint.accepted_contact_line_tangent[1]) <=
               frame_tolerance * tangent_norm));
}

[[nodiscard]] std::vector<
    svmp::FE::level_set::LevelSetWallContactConstraint>
canonicalizeAcceptedWallConstraints(
    std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
        constraints,
    const svmp::MeshComm& comm,
    std::string_view source)
{
  const auto order = [](const auto& left, const auto& right) {
    return std::tie(left.parent_cell_global_id,
                    left.interface_marker,
                    left.boundary_marker,
                    left.kind,
                    left.geometry_revision,
                    left.target_angle_radians,
                    left.physical_wall_normal,
                    left.accepted_contact_point,
                    left.accepted_contact_line_tangent) <
           std::tie(right.parent_cell_global_id,
                    right.interface_marker,
                    right.boundary_marker,
                    right.kind,
                    right.geometry_revision,
                    right.target_angle_radians,
                    right.physical_wall_normal,
                    right.accepted_contact_point,
                    right.accepted_contact_line_tangent);
  };
  std::sort(constraints.begin(), constraints.end(), order);
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint> unique;
  unique.reserve(constraints.size());
  bool local_conflict = false;
  for (const auto& constraint : constraints) {
    if (!unique.empty() &&
        unique.back().parent_cell_global_id ==
            constraint.parent_cell_global_id) {
      if (sameAcceptedWallConstraint(unique.back(), constraint)) {
        continue;
      }
      local_conflict = true;
      continue;
    }
    unique.push_back(constraint);
  }
  if (globalMinDouble(local_conflict ? 0.0 : 1.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " + std::string(source) +
        " assigns distinct accepted wall-contact frames to one parent cell; "
        "the projection supports exactly one physical frame per parent.");
  }
  return unique;
}

std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
captureAcceptedContactStageWallConstraints(
    application::core::SimulationComponents& sim,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        accepted_contact_stages)
{
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint> captured;
  const auto declarations =
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations();
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (globalMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage constraint capture requires an authoritative geometry snapshot on every rank.");
  }
  bool local_snapshots_current = true;
  try {
    context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
        sim.fe_system->meshAccess());
  } catch (const std::exception&) {
    local_snapshots_current = false;
  }
  if (globalMinDouble(local_snapshots_current ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage constraint capture rejected a stale authoritative geometry snapshot.");
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
        "[svMultiPhysics::Application] Accepted contact-stage constraint capture requires unique nonnegative interface markers.");
  }

  for (const auto& declaration : declarations) {
    if (declaration.parameters.dynamic_contact_coefficients.empty()) {
      continue;
    }
    const auto stage_it = stage_by_interface.find(declaration.interface_marker);
    const bool local_stage_available = stage_it != stage_by_interface.end();
    if (globalMinDouble(local_stage_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture is missing interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& stage = *stage_it->second;
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto snapshot_it = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     stage.geometry_revision.snapshot_revision_key;
        });
    if (globalMinDouble(snapshot_it != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture cannot retain the stage geometry for interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshot = **snapshot_it;
    const bool local_revision_valid =
        stage.geometry_revision.complete() &&
        snapshot.revision().snapshot_revision_key ==
            stage.geometry_revision.snapshot_revision_key &&
        snapshot.revision().interface_marker == declaration.interface_marker;
    if (globalMinDouble(local_revision_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture found mismatched stage geometry provenance.");
    }

    std::map<int, svmp::FE::level_set::LevelSetWallContactConstraintKind>
        contact_law_by_wall;
    for (const auto& coefficient :
         declaration.parameters.young_wall_coefficients) {
      contact_law_by_wall.emplace(
          coefficient.boundary_marker,
          svmp::FE::level_set::LevelSetWallContactConstraintKind::
              PrescribedAngle);
    }
    for (const auto& coefficient :
         declaration.parameters.dynamic_contact_coefficients) {
      contact_law_by_wall[coefficient.boundary_marker] =
          svmp::FE::level_set::LevelSetWallContactConstraintKind::
              AcceptedDynamicAngle;
    }
    bool local_parent_ids_valid = true;
    bool local_frames_valid = true;
    std::string local_frame_diagnostic;
    std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
        declaration_constraints;
    for (const auto& record : snapshot.rules()) {
      if (record.role != svmp::FE::interfaces::
                             FreeSurfaceGeometryRuleRole::Contact ||
          record.retention != svmp::FE::interfaces::
                                  FreeSurfaceGeometryRetention::Retained ||
          !record.locally_owned) {
        continue;
      }
      const auto law =
          contact_law_by_wall.find(record.physical_boundary_marker);
      if (law == contact_law_by_wall.end()) {
        continue;
      }
      const auto parent =
          record.reference_rule.provenance.parent_entity_global_id;
      local_parent_ids_valid =
          parent != svmp::FE::INVALID_GLOBAL_INDEX &&
          local_parent_ids_valid;
      if (parent == svmp::FE::INVALID_GLOBAL_INDEX) {
        continue;
      }
      try {
        declaration_constraints.push_back(makeAcceptedSnapshotWallConstraint(
            record,
            law->second,
            declaration.interface_marker,
            stage.geometry_revision.snapshot_revision_key,
            declaration.parameters,
            sim.fe_system->meshAccess().dimension()));
      } catch (const std::exception& error) {
        local_frames_valid = false;
        local_frame_diagnostic = error.what();
      }
    }
    if (globalMinDouble(local_parent_ids_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture found a rule without a partition-independent parent identity.");
    }
    if (globalMinDouble(local_frames_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture rejected an unavailable, degenerate, or stale physical contact frame" +
          (local_frame_diagnostic.empty()
               ? std::string{"."}
               : std::string{": "} + local_frame_diagnostic));
    }
    declaration_constraints = canonicalizeAcceptedWallConstraints(
        std::move(declaration_constraints),
        comm,
        "Accepted contact-stage constraint capture");
    captured.insert(captured.end(),
                    declaration_constraints.begin(),
                    declaration_constraints.end());
  }
  return captured;
}

struct LevelSetWallAwareMaintenanceContext {
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
      local_constraints{};
  bool requires_accepted_dynamic_stage{false};
  bool has_global_contact_constraints{false};
  std::size_t global_prescribed_contact_rules{0u};
  std::size_t global_dynamic_contact_rules{0u};
  svmp::FE::Real stage_time{0.0};
  svmp::FE::Real stage_alpha_f{1.0};
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
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations();
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

  const auto* context = sim.fe_system->cutIntegrationContext();
  if (globalMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set maintenance requires an authoritative geometry snapshot on every rank.");
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

  bool dynamic_stage_initialized = false;
  for (const auto& declaration : declarations) {
    if (declaration.level_set_field != level_set_field) {
      continue;
    }
    std::map<int, svmp::FE::level_set::LevelSetWallContactConstraintKind>
        contact_law_by_wall;
    for (const auto& coefficient :
         declaration.parameters.young_wall_coefficients) {
      contact_law_by_wall.emplace(
          coefficient.boundary_marker,
          svmp::FE::level_set::LevelSetWallContactConstraintKind::
              PrescribedAngle);
    }
    for (const auto& coefficient :
         declaration.parameters.dynamic_contact_coefficients) {
      // A dynamic law supersedes the equilibrium Young datum on the same
      // wall.  The latter remains the momentum-side energy coefficient; the
      // accepted stage owns the redistancing geometry.
      contact_law_by_wall[coefficient.boundary_marker] =
          svmp::FE::level_set::LevelSetWallContactConstraintKind::
              AcceptedDynamicAngle;
    }
    if (contact_law_by_wall.empty()) {
      continue;
    }

    if (!declaration.parameters.dynamic_contact_coefficients.empty()) {
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
      if (minimum_stage_revision != maximum_stage_revision ||
          minimum_previous_revision != maximum_previous_revision ||
          minimum_endpoint_revision != maximum_endpoint_revision ||
          minimum_stage_time != maximum_stage_time ||
          minimum_stage_alpha != maximum_stage_alpha) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wall-aware level-set maintenance received different accepted dynamic stages across ranks for interface marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
      if (!dynamic_stage_initialized) {
        resolved.stage_time = accepted_dynamic_stage->stage_time;
        resolved.stage_alpha_f = accepted_dynamic_stage->stage_alpha_f;
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
      resolved.requires_accepted_dynamic_stage = true;
      std::map<int, std::size_t> local_rule_count_by_wall;
      std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
          declaration_constraints;
      bool local_constraints_valid = true;
      for (const auto& constraint : accepted_contact_stage_constraints) {
        if (constraint.interface_marker != declaration.interface_marker) {
          continue;
        }
        const auto law =
            contact_law_by_wall.find(constraint.boundary_marker);
        local_constraints_valid =
            law != contact_law_by_wall.end() &&
            law->second == constraint.kind &&
            constraint.parent_cell_global_id !=
                svmp::FE::INVALID_GLOBAL_INDEX &&
            constraint.geometry_revision ==
                accepted_dynamic_stage->geometry_revision
                    .snapshot_revision_key &&
            acceptedWallConstraintFrameIsComplete(
                constraint,
                declaration.parameters,
                sim.fe_system->meshAccess().dimension()) &&
            local_constraints_valid;
        if (law == contact_law_by_wall.end() ||
            law->second != constraint.kind ||
            constraint.parent_cell_global_id ==
                svmp::FE::INVALID_GLOBAL_INDEX ||
            constraint.geometry_revision !=
                accepted_dynamic_stage->geometry_revision
                    .snapshot_revision_key ||
            !acceptedWallConstraintFrameIsComplete(
                constraint,
                declaration.parameters,
                sim.fe_system->meshAccess().dimension())) {
          continue;
        }
        declaration_constraints.push_back(constraint);
      }
      if (globalMinDouble(local_constraints_valid ? 1.0 : 0.0, comm) !=
          1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wall-aware level-set maintenance rejected a captured contact constraint that does not match its accepted stage.");
      }
      declaration_constraints = canonicalizeAcceptedWallConstraints(
          std::move(declaration_constraints),
          comm,
          "Accepted contact-stage wall-aware maintenance");
      for (const auto& constraint : declaration_constraints) {
        resolved.local_constraints.push_back(constraint);
        ++local_rule_count_by_wall[constraint.boundary_marker];
      }
      for (const auto& [boundary_marker, kind] : contact_law_by_wall) {
        const auto global_rule_count = globalSumSize(
            local_rule_count_by_wall[boundary_marker], comm);
        resolved.has_global_contact_constraints =
            resolved.has_global_contact_constraints || global_rule_count > 0u;
        if (kind == svmp::FE::level_set::
                        LevelSetWallContactConstraintKind::PrescribedAngle) {
          resolved.global_prescribed_contact_rules += global_rule_count;
          continue;
        }
        resolved.global_dynamic_contact_rules += global_rule_count;
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
        const bool dynamic_state_has_contact =
            wall_it->owned_quadrature_point_count > 0u ||
            wall_it->owned_contact_measure > svmp::FE::Real{0.0};
        if (dynamic_state_has_contact != (global_rule_count > 0u)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Wall-aware level-set maintenance contact-rule coverage disagrees with the accepted dynamic wall state for boundary marker " +
              std::to_string(boundary_marker) + ".");
        }
      }
      continue;
    }

    const bool local_marker_available =
        context->hasFreeSurfaceGeometrySnapshotForMarker(
            declaration.interface_marker);
    if (globalMinDouble(local_marker_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance is missing the authoritative snapshot for interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto snapshot_revision =
        context->freeSurfaceGeometrySnapshotRevisionForMarker(
            declaration.interface_marker);
    const auto [minimum_revision, maximum_revision] =
        globalMinMaxUint64(snapshot_revision, comm);
    if (minimum_revision != maximum_revision) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance found different geometry revisions across ranks for interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto snapshot_it = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     snapshot_revision;
        });
    if (globalMinDouble(snapshot_it != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance found incomplete snapshot storage for interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    bool local_snapshot_current = true;
    try {
      context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
          sim.fe_system->meshAccess());
    } catch (const std::exception&) {
      local_snapshot_current = false;
    }
    if (globalMinDouble(local_snapshot_current ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance rejected a stale geometry snapshot for interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshot = **snapshot_it;
    const bool local_snapshot_matches_declaration =
        snapshot.revision().complete() &&
        snapshot.revision().interface_marker == declaration.interface_marker &&
        snapshot.revision().snapshot_revision_key == snapshot_revision;
    if (globalMinDouble(local_snapshot_matches_declaration ? 1.0 : 0.0,
                        comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance snapshot provenance does not match interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    for (const auto& [boundary_marker, kind] : contact_law_by_wall) {
      std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
          wall_constraints;
      bool local_parent_ids_valid = true;
      bool local_frames_valid = true;
      std::string local_frame_diagnostic;
      for (const auto& record : snapshot.rules()) {
        if (record.role != svmp::FE::interfaces::
                               FreeSurfaceGeometryRuleRole::Contact ||
            record.retention != svmp::FE::interfaces::
                                    FreeSurfaceGeometryRetention::Retained ||
            record.physical_boundary_marker != boundary_marker ||
            !record.locally_owned) {
          continue;
        }
        const auto parent =
            record.reference_rule.provenance.parent_entity_global_id;
        local_parent_ids_valid =
            parent != svmp::FE::INVALID_GLOBAL_INDEX &&
            local_parent_ids_valid;
        if (parent == svmp::FE::INVALID_GLOBAL_INDEX) {
          continue;
        }
        try {
          wall_constraints.push_back(makeAcceptedSnapshotWallConstraint(
              record,
              kind,
              declaration.interface_marker,
              snapshot_revision,
              declaration.parameters,
              sim.fe_system->meshAccess().dimension()));
        } catch (const std::exception& error) {
          local_frames_valid = false;
          local_frame_diagnostic = error.what();
        }
      }
      if (globalMinDouble(local_parent_ids_valid ? 1.0 : 0.0, comm) != 1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wall-aware level-set maintenance found a contact rule without a partition-independent parent identity.");
      }
      if (globalMinDouble(local_frames_valid ? 1.0 : 0.0, comm) != 1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wall-aware level-set maintenance rejected an unavailable, degenerate, or stale prescribed physical contact frame" +
            (local_frame_diagnostic.empty()
                 ? std::string{"."}
                 : std::string{": "} + local_frame_diagnostic));
      }
      wall_constraints = canonicalizeAcceptedWallConstraints(
          std::move(wall_constraints),
          comm,
          "Prescribed wall-aware level-set maintenance");
      const auto local_rule_count = wall_constraints.size();
      resolved.local_constraints.insert(
          resolved.local_constraints.end(),
          wall_constraints.begin(),
          wall_constraints.end());
      const auto global_rule_count = globalSumSize(local_rule_count, comm);
      resolved.has_global_contact_constraints =
          resolved.has_global_contact_constraints || global_rule_count > 0u;
      resolved.global_prescribed_contact_rules += global_rule_count;
    }
  }
  resolved.local_constraints = canonicalizeAcceptedWallConstraints(
      std::move(resolved.local_constraints),
      comm,
      "Wall-aware level-set maintenance");
  return resolved;
}

void assertCollectiveWallAwareRepairResult(
    const svmp::FE::level_set::LevelSetSignedDistanceRepairResult& result,
    const svmp::MeshComm& comm,
    std::string_view field_name)
{
  const auto require_consistent_count = [&](std::size_t value) {
    const auto [minimum, maximum] = globalMinMaxUint64(
        static_cast<std::uint64_t>(value), comm);
    return minimum == maximum;
  };
  bool metadata_consistent =
      require_consistent_count(result.success ? 1u : 0u) &&
      require_consistent_count(result.converged ? 1u : 0u) &&
      require_consistent_count(
          result.wall_contact_constraints_satisfied ? 1u : 0u) &&
      require_consistent_count(result.wall_contact_constraints) &&
      require_consistent_count(result.wall_contact_cells) &&
      require_consistent_count(result.wall_contact_dofs);
  const std::array<svmp::FE::Real, 8> metrics{
      result.max_wall_contact_scale_residual,
      result.max_prescribed_contact_value_residual,
      result.max_prescribed_contact_angle_error_radians,
      result.max_contact_line_displacement,
      result.max_contact_angle_change_radians,
      result.max_unconstrained_signed_distance_error,
      result.max_wall_constrained_signed_distance_error,
      result.max_iteration_residual};
  for (const auto metric : metrics) {
    const bool local_finite = std::isfinite(metric);
    if (globalMinDouble(local_finite ? 1.0 : 0.0, comm) != 1.0) {
      metadata_consistent = false;
      continue;
    }
    const auto minimum = globalMinDouble(static_cast<double>(metric), comm);
    const auto maximum = globalMaxDouble(static_cast<double>(metric), comm);
    const auto tolerance =
        4096.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, std::abs(minimum), std::abs(maximum)});
    metadata_consistent =
        std::abs(maximum - minimum) <= tolerance && metadata_consistent;
  }
  if (!metadata_consistent) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set reinitialization produced rank-dependent convergence or contact-geometry results for field '" +
        std::string(field_name) + "'.");
  }
}

struct LevelSetProjectionReinitializationCandidate {
  bool applied{false};
  svmp::FE::level_set::LevelSetSignedDistanceRepairResult repair{};
  LevelSetWallAwareMaintenanceContext wall_context{};
};

LevelSetProjectionReinitializationCandidate
stageLevelSetProjectionReinitialization(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    const LevelSetMaintenanceRequest& request,
    svmp::FE::FieldId field,
    svmp::FE::Real endpoint_time,
    std::vector<svmp::FE::Real>& candidate,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        accepted_contact_stages,
    std::span<const svmp::FE::level_set::LevelSetWallContactConstraint>
        accepted_contact_stage_constraints,
    std::span<const svmp::FE::Real> accepted_contact_stage_solution)
{
  if (request.reinitialization.method !=
      svmp::FE::level_set::LevelSetReinitializationMethod::Projection) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Runtime level-set reinitialization currently supports Projection only.");
  }

  LevelSetProjectionReinitializationCandidate staged;
  staged.wall_context = resolveLevelSetWallAwareMaintenanceContext(
      sim,
      history,
      field,
      endpoint_time,
      accepted_contact_stages,
      accepted_contact_stage_constraints);
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  std::span<const svmp::FE::Real> reinitialization_input(candidate);
  if (staged.wall_context.requires_accepted_dynamic_stage) {
    const bool local_stage_layout_valid =
        accepted_contact_stage_solution.size() == candidate.size();
    if (globalMinDouble(local_stage_layout_valid ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic wall-aware level-set maintenance requires the exact accepted contact-stage solution on every rank.");
    }
    const double local_stage_solution_size =
        static_cast<double>(accepted_contact_stage_solution.size());
    if (globalMinDouble(local_stage_solution_size, comm) !=
        globalMaxDouble(local_stage_solution_size, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic wall-aware level-set maintenance received inconsistent contact-stage solution layouts.");
    }
    reinitialization_input = accepted_contact_stage_solution;
  }

  std::vector<svmp::FE::Real> repaired;
  staged.repair =
      svmp::FE::level_set::repairLevelSetSignedDistanceByProjection(
          *sim.fe_system,
          field,
          request.reinitialization,
          reinitialization_input,
          repaired,
          staged.wall_context.local_constraints);
  assertCollectiveWallAwareRepairResult(
      staged.repair, comm, request.level_set_field_name);
  if ((staged.repair.wall_contact_constraints > 0u) !=
      staged.wall_context.has_global_contact_constraints) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set reinitialization did not retain the complete accepted contact constraint set for field '" +
        request.level_set_field_name + "'.");
  }
  if (!staged.repair.success) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set reinitialization failed for field '" +
        request.level_set_field_name + "': " + staged.repair.diagnostic);
  }
  if (!staged.repair.converged) {
    return staged;
  }

  const auto raw_field_offset = sim.fe_system->fieldDofOffset(field);
  const auto raw_field_dof_count =
      sim.fe_system->fieldDofHandler(field).getNumDofs();
  if (raw_field_offset < 0 || raw_field_dof_count < 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set reinitialization returned an invalid field layout.");
  }
  const auto field_offset = static_cast<std::size_t>(raw_field_offset);
  const auto field_dof_count =
      static_cast<std::size_t>(raw_field_dof_count);
  const auto slice_fits = [&](std::size_t extent) {
    return field_offset <= extent &&
           field_dof_count <= extent - field_offset;
  };
  if (!slice_fits(candidate.size()) || !slice_fits(repaired.size()) ||
      !slice_fits(reinitialization_input.size())) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set reinitialization returned an incompatible field slice.");
  }
  for (std::size_t i = 0; i < field_dof_count; ++i) {
    const auto index = field_offset + i;
    candidate[index] += repaired[index] - reinitialization_input[index];
  }
  staged.applied = true;
  return staged;
}

struct LevelSetVolumeCorrectionMaintenanceEvent {
  svmp::FE::FieldId level_set_field{svmp::FE::INVALID_FIELD_ID};
  std::string level_set_field_name{};
  int completed_step{0};
  bool includes_other_maintenance{false};
  svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult correction{};
};

using LevelSetMaintenanceCandidateValidator = std::function<void(
    std::span<const svmp::FE::Real>,
    std::span<const LevelSetVolumeCorrectionMaintenanceEvent>)>;

bool applyLevelSetMaintenance(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    std::vector<LevelSetMaintenanceRequest>& requests,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        accepted_contact_stages = {},
    std::span<const svmp::FE::level_set::LevelSetWallContactConstraint>
        accepted_contact_stage_constraints = {},
    std::span<const svmp::FE::Real> accepted_contact_stage_solution = {},
    std::vector<LevelSetVolumeCorrectionMaintenanceEvent>*
        applied_volume_corrections = nullptr,
    const LevelSetMaintenanceCandidateValidator& validate_candidate = {},
    const LevelSetMaintenanceStageObserver& observe_stage = {})
{
  if (applied_volume_corrections != nullptr) {
    applied_volume_corrections->clear();
  }
  if (!sim.fe_system || requests.empty()) {
    return false;
  }

  // Treat maintenance across every configured level-set field as one
  // transaction.  A later request can fail after an earlier correction has
  // been computed, but no accounting/target state may advance unless every
  // request succeeds and the common FE history update is complete.
  auto staged_requests = requests;
  std::vector<std::string> staged_commit_logs;
  std::vector<LevelSetVolumeCorrectionMaintenanceEvent>
      staged_volume_corrections;
  bool changed = false;
  auto fe_solution = gatherFeOrderedSolution(history.u());
  const auto accepted_solution_before_maintenance = fe_solution;
  std::set<svmp::FE::FieldId> modified_level_set_fields;
  for (auto& request : staged_requests) {
    // Conservative requests own reinitialization inside their prospective
    // step transaction.  Repeating it here after acceptance would expose an
    // intermediate geometry and apply a second representation change outside
    // the phase-mass reconciliation contract.
    if (request.conservative_phase.enabled) {
      continue;
    }
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
      const auto before_reinitialization = fe_solution;
      const auto staged_reinitialization =
          stageLevelSetProjectionReinitialization(
              sim,
              history,
              request,
              field,
              static_cast<svmp::FE::Real>(history.time()),
              fe_solution,
              accepted_contact_stages,
              accepted_contact_stage_constraints,
              accepted_contact_stage_solution);
      const auto& result = staged_reinitialization.repair;
      const auto& wall_context = staged_reinitialization.wall_context;
      if (!result.converged) {
        // A geometrically bounded partial candidate is useful diagnostics, but
        // applying it would silently turn max_iterations into an uncontrolled
        // change of the production interface.  Retain the accepted solution
        // and make the skipped repair explicit.
        std::ostringstream log;
        log
            << "[svMultiPhysics::Application] Level-set reinitialization skipped field='"
            << request.level_set_field_name << "' step=" << completed_step
            << " reason=nonconverged"
            << " iterations=" << result.iterations
            << " max_iteration_residual="
            << result.max_iteration_residual
            << " max_signed_distance_error="
            << result.max_signed_distance_error
            << " max_unconstrained_signed_distance_error="
            << result.max_unconstrained_signed_distance_error
            << " max_wall_constrained_signed_distance_error="
            << result.max_wall_constrained_signed_distance_error
            << " wall_contact_constraints="
            << result.wall_contact_constraints
            << " wall_contact_cells=" << result.wall_contact_cells
            << " wall_contact_dofs=" << result.wall_contact_dofs
            << " wall_contact_constraints_satisfied="
            << (result.wall_contact_constraints_satisfied ? "true" : "false")
            << " max_wall_contact_scale_residual="
            << result.max_wall_contact_scale_residual
            << " max_prescribed_contact_value_residual="
            << result.max_prescribed_contact_value_residual
            << " max_prescribed_contact_angle_error_radians="
            << result.max_prescribed_contact_angle_error_radians
            << " max_contact_line_displacement="
            << result.max_contact_line_displacement
            << " max_contact_angle_change_radians="
            << result.max_contact_angle_change_radians
            << " max_interface_displacement="
            << result.max_interface_displacement
            << " diagnostic='" << result.diagnostic << "'";
        staged_commit_logs.push_back(log.str());
      } else {
        // For a generalized-alpha contact law, repaired is the accepted stage
        // representation.  Apply only its field delta to the endpoint.  The
        // common history update below applies that same representation delta
        // to every older level, so the accepted stage becomes repaired while
        // all temporal increments remain exact.
        if (!staged_reinitialization.applied) {
          throw std::logic_error(
              "[svMultiPhysics::Application] A converged level-set reinitialization was not applied to its staged candidate.");
        }
        changed = true;
        modified_level_set_fields.insert(field);
        if (observe_stage) {
          observe_stage(
              application::core::LevelSetMaintenanceWorkSubstage::
                  Reinitialization,
              before_reinitialization,
              fe_solution);
        }
        for (auto& event : staged_volume_corrections) {
          if (event.level_set_field == field) {
            event.includes_other_maintenance = true;
          }
        }
        std::ostringstream log;
        log
          << "[svMultiPhysics::Application] Level-set reinitialized field='"
          << request.level_set_field_name << "' step=" << completed_step
          << " repaired_dofs=" << result.repaired_dofs
          << " preserved_dofs=" << result.preserved_dofs
          << " preserve_band_width=" << result.preserve_band_width
          << " interface_fragments=" << result.interface_fragments
          << " interface_displacement_samples="
          << result.interface_displacement_samples
          << " max_interface_displacement="
          << result.max_interface_displacement
          << " l2_interface_displacement="
          << result.l2_interface_displacement
          << " zero_set_bound_satisfied="
          << (result.zero_set_bound_satisfied ? "true" : "false")
          << " iterations=" << result.iterations
          << " converged=" << (result.converged ? "true" : "false")
          << " max_iteration_residual="
          << result.max_iteration_residual
          << " max_signed_distance_error="
          << result.max_signed_distance_error
          << " max_unconstrained_signed_distance_error="
          << result.max_unconstrained_signed_distance_error
          << " max_wall_constrained_signed_distance_error="
          << result.max_wall_constrained_signed_distance_error
          << " wall_contact_model="
          << (wall_context.requires_accepted_dynamic_stage
                  ? "accepted_dynamic_stage"
                  : (wall_context.has_global_contact_constraints
                         ? "prescribed_angle"
                         : "none"))
          << " accepted_contact_stage_alpha_f="
          << (wall_context.requires_accepted_dynamic_stage
                  ? wall_context.stage_alpha_f
                  : svmp::FE::Real{1.0})
          << " accepted_contact_stage_revision="
          << wall_context.stage_state_revision
          << " prescribed_contact_rules="
          << wall_context.global_prescribed_contact_rules
          << " dynamic_contact_rules="
          << wall_context.global_dynamic_contact_rules
          << " wall_contact_constraints="
          << result.wall_contact_constraints
          << " wall_contact_cells=" << result.wall_contact_cells
          << " wall_contact_dofs=" << result.wall_contact_dofs
          << " wall_contact_constraints_satisfied="
          << (result.wall_contact_constraints_satisfied ? "true" : "false")
          << " max_wall_contact_scale_residual="
          << result.max_wall_contact_scale_residual
          << " max_prescribed_contact_value_residual="
          << result.max_prescribed_contact_value_residual
          << " max_prescribed_contact_angle_error_radians="
          << result.max_prescribed_contact_angle_error_radians
          << " max_contact_line_displacement="
          << result.max_contact_line_displacement
          << " max_contact_angle_change_radians="
          << result.max_contact_angle_change_radians
          << " max_abs_update=" << result.max_abs_update;
        staged_commit_logs.push_back(log.str());
      }
    }

    if (do_volume) {
      const auto before_correction = fe_solution;
      const bool includes_other_maintenance =
          modified_level_set_fields.contains(field);
      if (!request.volume_target_initialized) {
        request.volume_target =
            request.volume_correction.target_negative_volume;
        request.volume_target_initialized = true;
      }
      const auto volume_options =
          levelSetVolumeOptionsForMaintenance(request);
      svmp::FE::level_set::LevelSetGlobalShiftCorrectionOptions correction_options{};
      correction_options.target_negative_volume = request.volume_target;
      correction_options.volume_tolerance =
          request.volume_correction.volume_tolerance;
      correction_options.max_iterations =
          request.volume_correction.max_iterations;
      correction_options.minimum_relative_volume_error =
          request.volume_correction.minimum_relative_volume_error;
      correction_options.maximum_interface_displacement_fraction =
          request.volume_correction.maximum_interface_displacement_fraction;

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
      if (result.correction_applied) {
        // The low-level correction has not mutated the accepted coefficients;
        // reject before applying its returned vector if repeated global shifts
        // would exceed the application-level cumulative path-length budget.
        accountAppliedLevelSetVolumeCorrection(request, result);
        if (std::any_of(
                staged_volume_corrections.begin(),
                staged_volume_corrections.end(),
                [field](const auto& event) {
                  return event.level_set_field == field;
                })) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] One maintenance transaction cannot apply multiple global volume corrections to the same level-set field.");
        }
        staged_volume_corrections.push_back(
            LevelSetVolumeCorrectionMaintenanceEvent{
                .level_set_field = field,
                .level_set_field_name = request.level_set_field_name,
                .completed_step = completed_step,
                .includes_other_maintenance =
                    includes_other_maintenance,
                .correction = result,
            });
        fe_solution = std::move(corrected);
        changed = true;
        modified_level_set_fields.insert(field);
      }
      if (observe_stage) {
        observe_stage(
            application::core::LevelSetMaintenanceWorkSubstage::
                GlobalCorrection,
            before_correction,
            fe_solution);
      }
      std::ostringstream log;
      log << std::setprecision(17)
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
          << " correction_triggered="
          << (result.correction_triggered ? "true" : "false")
          << " correction_applied="
          << (result.correction_applied ? "true" : "false")
          << " target_reached="
          << (result.target_reached ? "true" : "false")
          << " trigger_volume_error=" << result.trigger_volume_error
          << " limited_by_displacement_bound="
          << (result.limited_by_displacement_bound ? "true" : "false")
          << " max_interface_displacement="
          << result.max_interface_displacement
          << " max_contact_line_displacement="
          << result.max_contact_line_displacement
          << " contact_line_displacement_bound="
          << result.contact_line_displacement_bound
          << " max_contact_angle_change_radians="
          << result.max_contact_angle_change_radians
          << " negative_component_topology_preserved="
          << (result.negative_component_topology_preserved ? "true"
                                                          : "false")
          << " negative_component_count="
          << result.negative_component_volume_transfers.size()
          << " total_component_volume_transfer="
          << result.total_component_volume_transfer
          << " total_absolute_component_volume_transfer="
          << result.total_absolute_component_volume_transfer
          << " maximum_absolute_component_volume_transfer="
          << result.maximum_absolute_component_volume_transfer
          << " maximum_allowed_interface_displacement="
          << result.maximum_allowed_interface_displacement
          << " maximum_topology_stable_shift="
          << result.maximum_topology_stable_shift
          << " minimum_edge_length=" << result.minimum_edge_length
          << " cumulative_interface_displacement="
          << request.cumulative_volume_correction_interface_displacement
          << " cumulative_contact_line_displacement="
          << request.cumulative_volume_correction_contact_line_displacement
          << " cumulative_displacement_budget_consumption="
          << std::max(
                 request.cumulative_volume_correction_interface_displacement,
                 request.cumulative_volume_correction_contact_line_displacement)
          << " cumulative_displacement_limiting_path="
          << (request.cumulative_volume_correction_contact_line_displacement >
                      request.cumulative_volume_correction_interface_displacement
                  ? "contact_line"
                  : "interface")
          << " cumulative_displacement_reference_minimum_edge_length="
          << request.volume_correction_reference_minimum_edge_length
          << " maximum_cumulative_interface_displacement="
          << (request.volume_correction_reference_minimum_edge_length *
              request.volume_correction
                  .maximum_cumulative_interface_displacement_fraction)
          << " status='" << result.diagnostic << "'"
          << " iterations=" << result.iterations
          << " volume_measure_source="
          << (volume_options.use_generated_interface_quadrature
                  ? "generated_interface_quadrature"
                  : "corner_linearized");
      for (const auto& transfer :
           result.negative_component_volume_transfers) {
        log << " component_global_vertex_id="
            << transfer.component_global_vertex_id
            << " component_initial_negative_volume="
            << transfer.initial_negative_volume
            << " component_corrected_negative_volume="
            << transfer.corrected_negative_volume
            << " component_volume_transfer="
            << transfer.volume_transfer;
      }
      staged_commit_logs.push_back(log.str());
    }
  }

  if (changed && validate_candidate) {
    validate_candidate(fe_solution, staged_volume_corrections);
  }

  if (changed) {
    // Reinitialization and a global shift change the level-set
    // representation after the physical step.  Apply the identical FE
    // coefficient delta to every stored history level.  This preserves every
    // BDF/generalized-alpha temporal difference exactly and therefore keeps
    // the already stored rate state consistent.  Copying the repaired current
    // field into uPrev (the former behavior) erased the accepted increment,
    // left older BDF history in another representation, and disagreed with
    // uDot.
    const auto apply_representation_delta =
        [&](std::vector<svmp::FE::Real>& target) {
          if (target.size() != fe_solution.size() ||
              accepted_solution_before_maintenance.size() !=
                  fe_solution.size()) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Level-set maintenance history "
                "synchronization encountered incompatible FE vector sizes.");
          }
          std::size_t updated = 0u;
          for (const auto field : modified_level_set_fields) {
            const auto offset = static_cast<std::size_t>(
                sim.fe_system->fieldDofOffset(field));
            const auto count = static_cast<std::size_t>(
                sim.fe_system->fieldDofHandler(field).getNumDofs());
            if (offset + count > target.size()) {
              throw std::runtime_error(
                  "[svMultiPhysics::Application] Level-set maintenance field "
                  "slice exceeds the FE history vector.");
            }
            for (std::size_t i = 0; i < count; ++i) {
              const auto index = offset + i;
              target[index] +=
                  fe_solution[index] -
                  accepted_solution_before_maintenance[index];
            }
            updated += count;
          }
          return updated;
        };

    auto current_solution = gatherFeOrderedSolution(history.u());
    const auto synchronized_dofs =
        apply_representation_delta(current_solution);
    std::vector<std::vector<svmp::FE::Real>> synchronized_history;
    synchronized_history.reserve(
        static_cast<std::size_t>(history.historyDepth()));
    for (int k = 1; k <= history.historyDepth(); ++k) {
      auto previous_solution =
          gatherFeOrderedSolution(history.uPrevK(k));
      (void)apply_representation_delta(previous_solution);
      synchronized_history.push_back(std::move(previous_solution));
    }
    if (synchronized_history.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set maintenance requires at least one FE history level.");
    }
    const auto current_previous_delta = globalMaxAbsDifference(
        std::span<const svmp::FE::Real>(current_solution.data(),
                                       current_solution.size()),
        std::span<const svmp::FE::Real>(
            synchronized_history.front().data(),
            synchronized_history.front().size()),
        activeFESystemCommunicator(*sim.fe_system));
    std::ostringstream log;
    log
        << "[svMultiPhysics::Application] Level-set maintenance synchronized"
        << " step=" << history.stepIndex()
        << " representation_delta=all_history_levels"
        << " temporal_increments=preserved"
        << " rate_state=unchanged_consistently"
        << " synchronized_fields=" << modified_level_set_fields.size()
        << " synchronized_dofs=" << synchronized_dofs
        << " current_previous_max_abs_delta=" << current_previous_delta
        << " history_depth=" << history.historyDepth();
    staged_commit_logs.push_back(log.str());

    // Every size, field-slice, collective, and diagnostic invariant has now
    // passed.  Publish the staged coefficient arrays only after those checks
    // so a rejected candidate leaves every accepted history vector untouched.
    scatterFeOrderedSolution(history.u(), current_solution);
    for (int k = 1; k <= history.historyDepth(); ++k) {
      scatterFeOrderedSolution(
          history.uPrevK(k),
          synchronized_history[static_cast<std::size_t>(k - 1)]);
    }
    history.updateGhosts();
  }
  requests = std::move(staged_requests);
  for (const auto& log : staged_commit_logs) {
    application::core::oopCout() << log << std::endl;
  }
  if (applied_volume_corrections != nullptr) {
    *applied_volume_corrections = std::move(staged_volume_corrections);
  }
  return changed;
}

std::vector<std::string> buildLevelSetVolumeCorrectionFreeSurfaceWorkLogs(
    application::core::SimulationComponents& sim,
    std::span<const LevelSetVolumeCorrectionMaintenanceEvent> events,
    std::span<const svmp::FE::systems::
                        AcceptedFreeSurfaceDiscreteFunctionalState>
        before,
    std::span<const svmp::FE::systems::
                        AcceptedFreeSurfaceDiscreteFunctionalState>
        after)
{
  std::vector<std::string> logs;
  if (!sim.fe_system) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Volume-correction work accounting requires an FE system.");
  }
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const auto local_event_count = static_cast<double>(events.size());
  if (globalMinDouble(local_event_count, comm) !=
      globalMaxDouble(local_event_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Applied volume-correction event coverage differs across the FE communicator.");
  }
  if (events.empty()) {
    return logs;
  }
  const auto declarations =
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations();
  const auto local_declaration_count =
      static_cast<double>(declarations.size());
  const auto local_before_count = static_cast<double>(before.size());
  const auto local_after_count = static_cast<double>(after.size());
  if (globalMinDouble(local_declaration_count, comm) !=
          globalMaxDouble(local_declaration_count, comm) ||
      globalMinDouble(local_before_count, comm) !=
          globalMaxDouble(local_before_count, comm) ||
      globalMinDouble(local_after_count, comm) !=
          globalMaxDouble(local_after_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Volume-correction functional state coverage differs across the FE communicator.");
  }
  std::map<
      int,
      const svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState*>
      before_by_marker;
  std::map<
      int,
      const svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState*>
      after_by_marker;
  bool local_state_maps_valid =
      before.size() == after.size() && before.size() == declarations.size();
  for (const auto& state : before) {
    local_state_maps_valid =
        state.interface_marker >= 0 &&
        before_by_marker.emplace(state.interface_marker, &state).second &&
        local_state_maps_valid;
  }
  for (const auto& state : after) {
    local_state_maps_valid =
        state.interface_marker >= 0 &&
        after_by_marker.emplace(state.interface_marker, &state).second &&
        local_state_maps_valid;
  }
  if (local_state_maps_valid) {
    auto before_it = before_by_marker.begin();
    auto after_it = after_by_marker.begin();
    for (; before_it != before_by_marker.end(); ++before_it, ++after_it) {
      local_state_maps_valid =
          before_it->first == after_it->first && local_state_maps_valid;
    }
  }
  if (globalMinDouble(local_state_maps_valid ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Volume-correction work accounting requires unique, matching functional state sets.");
  }

  for (const auto& event : events) {
    const bool local_event_valid =
        event.level_set_field != svmp::FE::INVALID_FIELD_ID &&
        !event.level_set_field_name.empty() &&
        sim.fe_system->findFieldByName(event.level_set_field_name) ==
            event.level_set_field &&
        event.completed_step >= 0 && event.correction.correction_applied &&
        std::isfinite(event.correction.applied_shift);
    if (globalMinDouble(local_event_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Volume-correction work accounting received an invalid applied event.");
    }
    const std::array<double, 5> event_metadata{
        static_cast<double>(event.level_set_field),
        static_cast<double>(event.completed_step),
        event.includes_other_maintenance ? 1.0 : 0.0,
        static_cast<double>(
            event.correction.negative_component_volume_transfers.size()),
        static_cast<double>(event.correction.applied_shift)};
    bool event_metadata_consistent = true;
    for (const auto value : event_metadata) {
      const auto minimum = globalMinDouble(value, comm);
      const auto maximum = globalMaxDouble(value, comm);
      const auto tolerance =
          4096.0 * std::numeric_limits<double>::epsilon() *
          std::max({1.0, std::abs(minimum), std::abs(maximum)});
      event_metadata_consistent =
          std::isfinite(value) &&
          std::abs(maximum - minimum) <= tolerance &&
          event_metadata_consistent;
    }
    if (!event_metadata_consistent) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Applied volume-correction event metadata differs across the FE communicator.");
    }
    std::size_t functional_count = 0u;
    svmp::FE::Real liquid_volume_change = 0.0;
    svmp::FE::Real liquid_gas_area_change = 0.0;
    svmp::FE::Real wetted_wall_area_change = 0.0;
    svmp::FE::Real contact_measure_change = 0.0;
    svmp::FE::Real surface_energy_change = 0.0;
    svmp::FE::Real young_wall_energy_change = 0.0;
    svmp::FE::Real volume_constraint_potential_change = 0.0;
    svmp::FE::Real total_potential_change = 0.0;
    std::ostringstream functional_details;
    for (const auto& declaration : declarations) {
      if (declaration.level_set_field != event.level_set_field) {
        continue;
      }
      ++functional_count;
      const auto before_it =
          before_by_marker.find(declaration.interface_marker);
      const auto after_it =
          after_by_marker.find(declaration.interface_marker);
      const bool local_state_available =
          before_it != before_by_marker.end() &&
          after_it != after_by_marker.end();
      if (globalMinDouble(local_state_available ? 1.0 : 0.0, comm) != 1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Volume-correction work accounting is missing a functional state for interface marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
      const auto& initial = *before_it->second;
      const auto& corrected = *after_it->second;
      const bool local_revision_valid =
          initial.geometry_revision.complete() &&
          corrected.geometry_revision.complete() &&
          initial.geometry_revision.interface_marker ==
              declaration.interface_marker &&
          corrected.geometry_revision.interface_marker ==
              declaration.interface_marker &&
          initial.geometry_revision.source_id ==
              corrected.geometry_revision.source_id &&
          initial.geometry_revision.domain_id ==
              declaration.geometry_domain_id &&
          corrected.geometry_revision.domain_id ==
              declaration.geometry_domain_id &&
          initial.geometry_revision.isovalue ==
              corrected.geometry_revision.isovalue &&
          initial.geometry_revision.source_value_revision !=
              corrected.geometry_revision.source_value_revision &&
          initial.geometry_revision.snapshot_revision_key !=
              corrected.geometry_revision.snapshot_revision_key &&
          initial.state.snapshot_revision_key ==
              initial.geometry_revision.snapshot_revision_key &&
          corrected.state.snapshot_revision_key ==
              corrected.geometry_revision.snapshot_revision_key &&
          initial.state.liquid_side == corrected.state.liquid_side &&
          initial.state.liquid_side == declaration.parameters.liquid_side &&
          initial.state.surface_tension == corrected.state.surface_tension &&
          initial.state.surface_tension ==
              declaration.parameters.surface_tension &&
          initial.state.volume_multiplier ==
              corrected.state.volume_multiplier &&
          initial.state.volume_multiplier ==
              declaration.parameters.volume_multiplier;
      if (globalMinDouble(local_revision_valid ? 1.0 : 0.0, comm) != 1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Applied volume correction did not produce distinct, complete free-surface geometry revisions.");
      }
      const auto& initial_state = initial.state;
      const auto& corrected_state = corrected.state;
      const auto marker_liquid_volume_change =
          corrected_state.owned_liquid_volume -
          initial_state.owned_liquid_volume;
      const auto marker_liquid_gas_area_change =
          corrected_state.owned_liquid_gas_area -
          initial_state.owned_liquid_gas_area;
      const auto marker_wetted_wall_area_change =
          corrected_state.owned_wetted_wall_area -
          initial_state.owned_wetted_wall_area;
      const auto marker_contact_measure_change =
          corrected_state.owned_contact_measure -
          initial_state.owned_contact_measure;
      const auto marker_surface_energy_change =
          corrected_state.liquid_gas_surface_energy -
          initial_state.liquid_gas_surface_energy;
      const auto marker_young_wall_energy_change =
          corrected_state.young_wall_energy -
          initial_state.young_wall_energy;
      const auto marker_volume_potential_change =
          corrected_state.volume_constraint_potential -
          initial_state.volume_constraint_potential;
      const auto marker_total_potential_change =
          corrected_state.total_potential - initial_state.total_potential;
      liquid_volume_change += marker_liquid_volume_change;
      liquid_gas_area_change += marker_liquid_gas_area_change;
      wetted_wall_area_change += marker_wetted_wall_area_change;
      contact_measure_change += marker_contact_measure_change;
      surface_energy_change += marker_surface_energy_change;
      young_wall_energy_change += marker_young_wall_energy_change;
      volume_constraint_potential_change +=
          marker_volume_potential_change;
      total_potential_change += marker_total_potential_change;
      functional_details
          << " interface_marker=" << declaration.interface_marker
          << " geometry_domain_id='" << declaration.geometry_domain_id << "'"
          << " initial_source_value_revision="
          << initial.geometry_revision.source_value_revision
          << " corrected_source_value_revision="
          << corrected.geometry_revision.source_value_revision
          << " initial_snapshot_revision="
          << initial.geometry_revision.snapshot_revision_key
          << " corrected_snapshot_revision="
          << corrected.geometry_revision.snapshot_revision_key
          << " marker_liquid_volume_change="
          << marker_liquid_volume_change
          << " marker_liquid_gas_area_change="
          << marker_liquid_gas_area_change
          << " marker_wetted_wall_area_change="
          << marker_wetted_wall_area_change
          << " marker_contact_measure_change="
          << marker_contact_measure_change
          << " marker_surface_energy_change="
          << marker_surface_energy_change
          << " marker_young_wall_energy_change="
          << marker_young_wall_energy_change
          << " marker_volume_constraint_potential_change="
          << marker_volume_potential_change
          << " marker_total_potential_change="
          << marker_total_potential_change;
    }
    const auto local_functional_count =
        static_cast<double>(functional_count);
    if (globalMinDouble(local_functional_count, comm) !=
        globalMaxDouble(local_functional_count, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Volume-correction functional matching differs across the FE communicator.");
    }
    const std::array<svmp::FE::Real, 8> totals{
        liquid_volume_change,
        liquid_gas_area_change,
        wetted_wall_area_change,
        contact_measure_change,
        surface_energy_change,
        young_wall_energy_change,
        volume_constraint_potential_change,
        total_potential_change};
    bool totals_consistent = true;
    for (const auto value : totals) {
      const bool local_finite = std::isfinite(value);
      if (globalMinDouble(local_finite ? 1.0 : 0.0, comm) != 1.0) {
        totals_consistent = false;
        continue;
      }
      const auto minimum = globalMinDouble(value, comm);
      const auto maximum = globalMaxDouble(value, comm);
      const auto tolerance =
          4096.0 * std::numeric_limits<double>::epsilon() *
          std::max({1.0, std::abs(minimum), std::abs(maximum)});
      totals_consistent =
          std::abs(maximum - minimum) <= tolerance && totals_consistent;
    }
    if (!totals_consistent) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Volume-correction work accounting differs across the FE communicator.");
    }

    std::ostringstream log;
    log << std::setprecision(17)
        << "[svMultiPhysics::Application] Level-set volume correction work"
        << " diagnostic=level_set_volume_correction_work"
        << " field='" << event.level_set_field_name << "'"
        << " step=" << event.completed_step
        << " scope="
        << (event.includes_other_maintenance
                ? "maintenance_sequence_with_other_stage"
                : "global_shift_only")
        << " numerical_work_classification=maintenance"
        << " numerical_work_sign=energy_after_minus_before"
        << " free_surface_functional_count=" << functional_count
        << " applied_shift=" << event.correction.applied_shift
        << " negative_component_count="
        << event.correction.negative_component_volume_transfers.size()
        << " negative_component_volume_transfer="
        << event.correction.total_component_volume_transfer
        << " liquid_volume_change=" << liquid_volume_change
        << " liquid_gas_area_change=" << liquid_gas_area_change
        << " wetted_wall_area_change=" << wetted_wall_area_change
        << " contact_measure_change=" << contact_measure_change
        << " surface_energy_change=" << surface_energy_change
        << " young_wall_energy_change=" << young_wall_energy_change
        << " volume_constraint_potential_change="
        << volume_constraint_potential_change
        << " total_potential_change=" << total_potential_change
        << " numerical_free_surface_work=" << total_potential_change
        << " applicability="
        << (functional_count > 0u ? "declared_free_surface_functional"
                                  : "non_free_surface_level_set")
        << functional_details.str();
    logs.push_back(log.str());
  }
  return logs;
}

void logLevelSetVolumeCorrectionFreeSurfaceWork(
    application::core::SimulationComponents& sim,
    std::span<const LevelSetVolumeCorrectionMaintenanceEvent> events,
    std::span<const svmp::FE::systems::
                        AcceptedFreeSurfaceDiscreteFunctionalState>
        before,
    std::span<const svmp::FE::systems::
                        AcceptedFreeSurfaceDiscreteFunctionalState>
        after)
{
  for (const auto& log : buildLevelSetVolumeCorrectionFreeSurfaceWorkLogs(
           sim, events, before, after)) {
    application::core::oopCout() << log << std::endl;
  }
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

std::optional<std::array<svmp::FE::Real, 3>> physicalCellPointAtReference(
    const svmp::FE::geometry::GeometryMapping& mapping,
    const std::array<svmp::FE::Real, 3>& reference_point)
{
  svmp::FE::math::Vector<svmp::FE::Real, 3> xi{};
  xi[0] = reference_point[0];
  xi[1] = reference_point[1];
  xi[2] = reference_point[2];
  const auto physical = mapping.map_to_physical(xi);
  return std::array<svmp::FE::Real, 3>{
      physical[0], physical[1], physical[2]};
}

std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample>
collectLevelSetCurvatureSupplementalSamples(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId field,
    svmp::FE::Real isovalue,
    std::optional<int> interface_marker = std::nullopt,
    std::optional<svmp::FE::geometry::CutIntegrationSide> volume_side =
        std::nullopt,
    std::uint64_t evaluated_state_source_revision = 0u)
{
  std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample> samples;
  const auto& rec = system.fieldRecord(field);
  if (!rec.space || rec.components != 1) {
    return samples;
  }

  std::uint64_t authoritative_snapshot_revision_key = 0u;
  std::uint64_t authoritative_source_value_revision = 0u;
  if (interface_marker.has_value()) {
    if (const auto* cut_context = system.cutIntegrationContext();
        cut_context != nullptr &&
        cut_context->hasFreeSurfaceGeometrySnapshotForMarker(
            *interface_marker)) {
      cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
          system.meshAccess());
      authoritative_snapshot_revision_key =
          cut_context->freeSurfaceGeometrySnapshotRevisionForMarker(
              *interface_marker);
      const auto& snapshots = cut_context->freeSurfaceGeometrySnapshots();
      const auto found = std::find_if(
          snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
            return candidate &&
                   candidate->revision().snapshot_revision_key ==
                       authoritative_snapshot_revision_key;
          });
      if (found == snapshots.end()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Curvature sample collection could "
            "not resolve its authoritative geometry snapshot revision.");
      }
      authoritative_source_value_revision =
          (*found)->revision().source_value_revision;
      if (evaluated_state_source_revision == 0u ||
          evaluated_state_source_revision !=
              authoritative_source_value_revision) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Curvature supplemental sample "
            "state does not match the authoritative geometry source revision.");
      }
    }
  }

  const auto& mesh = system.meshAccess();
  auto append_sample =
      [&](svmp::FE::MeshIndex parent_cell,
          const std::array<svmp::FE::Real, 3>& coordinate,
          svmp::FE::Real value,
          std::uint64_t snapshot_revision_key = 0u,
          std::uint64_t source_value_revision = 0u,
          std::uint64_t cut_topology_revision = 0u) {
        if (authoritative_snapshot_revision_key != 0u) {
          if (snapshot_revision_key !=
                  authoritative_snapshot_revision_key ||
              source_value_revision !=
                  authoritative_source_value_revision ||
              cut_topology_revision == 0u) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Curvature supplemental sample "
                "has incomplete or stale authoritative revision provenance.");
          }
        } else if (snapshot_revision_key != 0u ||
                   source_value_revision != 0u ||
                   cut_topology_revision != 0u) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Revisioned curvature "
              "supplemental sample has no authoritative geometry snapshot.");
        }
        if (!std::isfinite(coordinate[0]) ||
            !std::isfinite(coordinate[1]) ||
            !std::isfinite(coordinate[2]) ||
            !std::isfinite(value)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set curvature projection "
              "received a non-finite supplemental sample.");
        }
        constexpr svmp::FE::Real duplicate_tol2 =
            svmp::FE::Real{1.0e-24};
        constexpr svmp::FE::Real duplicate_value_tol =
            svmp::FE::Real{1.0e-12};
        for (const auto& existing : samples) {
          if (existing.parent_cell != parent_cell) {
            continue;
          }
          const auto dx = existing.coordinate[0] - coordinate[0];
          const auto dy = existing.coordinate[1] - coordinate[1];
          const auto dz = existing.coordinate[2] - coordinate[2];
          const auto dist2 = dx * dx + dy * dy + dz * dz;
          if (dist2 <= duplicate_tol2 &&
              std::abs(existing.value - value) <= duplicate_value_tol) {
            if (existing.free_surface_snapshot_revision_key != 0u &&
                snapshot_revision_key != 0u &&
                (existing.free_surface_snapshot_revision_key !=
                     snapshot_revision_key ||
                 existing.source_value_revision != source_value_revision)) {
              throw std::runtime_error(
                  "[svMultiPhysics::Application] Level-set curvature projection "
                  "found duplicate samples from different geometry snapshots.");
            }
            return;
          }
        }
        samples.push_back(
            svmp::FE::level_set::LevelSetCurvatureProjectionSample{
                .parent_cell = parent_cell,
                .coordinate = coordinate,
                .value = value,
                .free_surface_snapshot_revision_key = snapshot_revision_key,
                .source_value_revision = source_value_revision,
                .cut_topology_revision = cut_topology_revision});
      };

  if (interface_marker.has_value()) {
    if (const auto* cut_context = system.cutIntegrationContext()) {
      std::map<svmp::FE::GlobalIndex,
               std::shared_ptr<svmp::FE::geometry::GeometryMapping>>
          mapping_cache;
      auto mapping_for_cell =
          [&](svmp::FE::GlobalIndex cell)
              -> std::shared_ptr<svmp::FE::geometry::GeometryMapping> {
        auto it = mapping_cache.find(cell);
        if (it != mapping_cache.end()) {
          return it->second;
        }
        auto mapping = createCellGeometryMapping(mesh, cell);
        mapping_cache.emplace(cell, mapping);
        return mapping;
      };
      for (const auto* rule :
           cut_context->interfaceRulesForMarker(*interface_marker)) {
        if (rule == nullptr ||
            rule->kind != svmp::FE::geometry::CutQuadratureKind::Interface) {
          continue;
        }
        const auto parent_cell = rule->provenance.parent_entity;
        if (parent_cell < 0 || parent_cell >= mesh.numCells()) {
          continue;
        }
        for (const auto& point : rule->points) {
          if (!std::isfinite(point.weight) ||
              !(std::abs(point.weight) > svmp::FE::Real{0.0})) {
            continue;
          }
          std::optional<std::array<svmp::FE::Real, 3>> physical_point;
          if (rule->frame == svmp::FE::geometry::CutGeometryFrame::Current) {
            physical_point = point.point;
          } else {
            auto mapping = mapping_for_cell(parent_cell);
            if (mapping != nullptr) {
              physical_point =
                  physicalCellPointAtReference(*mapping, point.point);
            }
          }
          if (!physical_point.has_value()) {
            continue;
          }
          append_sample(
              parent_cell,
              *physical_point,
              isovalue,
              rule->provenance.free_surface_snapshot_revision_key,
              rule->provenance.source_value_revision,
              rule->provenance.cut_topology_revision);
        }
      }
    }
  }

  if (interface_marker.has_value() && volume_side.has_value()) {
    const auto cut_volume_samples =
        application::core::collectLevelSetCurvatureCutVolumeSupplementalSamples(
            system,
            state,
            field,
            *interface_marker,
            *volume_side,
            evaluated_state_source_revision);
    for (const auto& sample : cut_volume_samples) {
      append_sample(sample.parent_cell,
                    sample.coordinate,
                    sample.value,
                    sample.free_surface_snapshot_revision_key,
                    sample.source_value_revision,
                    sample.cut_topology_revision);
    }
  }

  bool has_high_order_cells = false;
  mesh.forEachCell([&](svmp::FE::GlobalIndex cell) {
    has_high_order_cells =
        has_high_order_cells || rec.space->polynomial_order(cell) > 1;
  });
  if (has_high_order_cells) {
    if (!interface_marker.has_value()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] High-order curvature sampling "
          "requires an authoritative interface marker and geometry snapshot.");
    }
    const auto high_order_samples =
        application::core::
            collectLevelSetCurvatureHighOrderSupplementalSamples(
                system,
                state,
                field,
                *interface_marker,
                evaluated_state_source_revision);
    for (const auto& sample : high_order_samples) {
      append_sample(sample.parent_cell,
                    sample.coordinate,
                    sample.value,
                    sample.free_surface_snapshot_revision_key,
                    sample.source_value_revision,
                    sample.cut_topology_revision);
    }
  }

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

void mixCurvatureProjectionCutRuleSignature(
    std::uint64_t& seed,
    const svmp::FE::geometry::CutQuadratureRule& rule)
{
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(rule.kind));
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(rule.side));
  mixCurvatureSignature(seed, static_cast<std::uint64_t>(rule.points.size()));
  mixCurvatureSignature(seed, rule.exact_for_constants ? 1u : 0u);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0, rule.exact_polynomial_order)));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(rule.policy.kind));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0, rule.policy.polynomial_order)));
  mixCurvatureSignatureReal(seed, rule.policy.tolerance);
  mixCurvatureSignatureString(seed, rule.policy.name);
  mixCurvatureSignatureString(seed, rule.provenance.embedded_geometry_id);
  mixCurvatureSignatureString(seed, rule.provenance.cut_topology_id);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max<svmp::FE::MeshIndex>(
                                0, rule.provenance.parent_entity)));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0, rule.provenance.marker)));
  mixCurvatureSignature(seed, rule.provenance.predicate_policy_key);
  mixCurvatureSignature(seed, rule.provenance.cut_topology_revision);
  mixCurvatureSignature(seed, rule.provenance.source_value_revision);
  mixCurvatureSignature(
      seed, rule.provenance.free_surface_snapshot_revision_key);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            rule.provenance.construction));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(rule.provenance.frame));
  mixCurvatureSignatureString(seed, rule.provenance.implicit_geometry_mode);
  mixCurvatureSignatureString(seed, rule.provenance.implicit_quadrature_backend);
  mixCurvatureSignatureString(seed,
                              rule.provenance.selected_implicit_quadrature_backend);
  mixCurvatureSignatureString(seed, rule.provenance.implicit_fallback_policy);
  mixCurvatureSignatureString(seed, rule.provenance.implicit_fallback_status);
  mixCurvatureSignatureString(seed, rule.provenance.geometry_tangent_policy);
  mixCurvatureSignatureReal(seed, rule.provenance.implicit_cut_root_tolerance);
  mixCurvatureSignatureReal(seed,
                            rule.provenance.implicit_cut_root_coordinate_tolerance);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0,
                                     rule.provenance.implicit_cut_root_max_iterations)));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0, rule.provenance.requested_quadrature_order)));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0, rule.provenance.achieved_quadrature_order)));
  mixCurvatureSignature(seed, rule.curved_geometry ? 1u : 0u);
  mixCurvatureSignature(seed, rule.full_cell_equivalent ? 1u : 0u);
}

std::optional<int> generatedCutContextMarkerForMaintenance(
    const svmp::FE::systems::FESystem& system,
    const LevelSetMaintenanceRequest& request)
{
  if (!request.volume_cut_request.has_value()) {
    return std::nullopt;
  }
  const auto& cut_request = *request.volume_cut_request;
  return application::core::resolvedActiveCutVolumeInterfaceMarker(
      system, cut_request);
}

struct CurvatureProjectionSnapshotIdentity {
  std::uint64_t free_surface_snapshot_revision_key{0};
  std::uint64_t source_value_revision{0};
};

CurvatureProjectionSnapshotIdentity curvatureProjectionSnapshotIdentity(
    const svmp::FE::systems::FESystem& system,
    const LevelSetMaintenanceRequest& request)
{
  const auto marker = generatedCutContextMarkerForMaintenance(system, request);
  const auto* cut_context = system.cutIntegrationContext();
  if (!marker.has_value() || cut_context == nullptr ||
      !cut_context->hasFreeSurfaceGeometrySnapshotForMarker(*marker)) {
    return {};
  }

  cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
      system.meshAccess());
  const auto snapshot_revision_key =
      cut_context->freeSurfaceGeometrySnapshotRevisionForMarker(*marker);
  const auto& snapshots = cut_context->freeSurfaceGeometrySnapshots();
  const auto found = std::find_if(
      snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
        return candidate &&
               candidate->revision().snapshot_revision_key ==
                   snapshot_revision_key;
      });
  if (found == snapshots.end()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Curvature projection could not resolve "
        "its authoritative geometry snapshot revision.");
  }
  return CurvatureProjectionSnapshotIdentity{
      .free_surface_snapshot_revision_key = snapshot_revision_key,
      .source_value_revision = (*found)->revision().source_value_revision};
}

std::uint64_t curvatureProjectionCutRequestSignature(
    const LevelSetMaintenanceRequest& request)
{
  std::uint64_t seed = 0x9e3779b97f4a7c15ull;
  mixCurvatureSignature(seed, request.volume_cut_request.has_value() ? 1u : 0u);
  if (!request.volume_cut_request.has_value()) {
    return seed;
  }

  const auto& cut_request = *request.volume_cut_request;
  mixCurvatureSignatureString(seed, cut_request.level_set_field_name);
  mixCurvatureSignatureString(seed, cut_request.domain_id);
  mixCurvatureSignatureReal(
      seed, static_cast<svmp::FE::Real>(cut_request.isovalue));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0, cut_request.requested_interface_marker)));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(cut_request.active_side));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(cut_request.geometry_mode));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            cut_request.implicit_cut_backend));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            cut_request.implicit_cut_fallback_policy));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            cut_request.geometry_tangent_policy));
  mixCurvatureSignatureReal(
      seed, static_cast<svmp::FE::Real>(
                cut_request.implicit_cut_root_tolerance));
  mixCurvatureSignatureReal(
      seed, static_cast<svmp::FE::Real>(
                cut_request.implicit_cut_root_coordinate_tolerance));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0, cut_request.implicit_cut_root_max_iterations)));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0, cut_request.implicit_cut_max_subdivision_depth)));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            std::max(0, cut_request.affected_cell_neighborhood_layers)));
  return seed;
}

std::uint64_t curvatureProjectionCutContextSignature(
    const svmp::FE::systems::FESystem& system,
    const LevelSetMaintenanceRequest& request,
    CurvatureProjectionCacheEntry* cache_entry = nullptr,
    bool* cache_hit = nullptr)
{
  std::uint64_t seed = curvatureProjectionCutRequestSignature(request);

  const auto* cut_context = system.cutIntegrationContext();
  mixCurvatureSignature(seed, cut_context != nullptr ? 1u : 0u);
  if (cut_context == nullptr) {
    if (cache_hit != nullptr) {
      *cache_hit = false;
    }
    return seed;
  }

  const auto marker = generatedCutContextMarkerForMaintenance(system, request);
  if (!marker.has_value()) {
    mixCurvatureSignature(seed, 0u);
    if (cache_hit != nullptr) {
      *cache_hit = false;
    }
    return seed;
  }
  mixCurvatureSignature(seed, 1u);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(std::max(0, *marker)));
  const auto snapshot_identity =
      curvatureProjectionSnapshotIdentity(system, request);
  mixCurvatureSignature(
      seed, snapshot_identity.free_surface_snapshot_revision_key);
  mixCurvatureSignature(seed, snapshot_identity.source_value_revision);

  const auto request_key = curvatureProjectionCutRequestSignature(request);
  const auto context_revision = cut_context->contentRevision();
  if (cache_entry != nullptr &&
      cache_entry->cut_context_signature_valid &&
      cache_entry->cut_context_signature_context == cut_context &&
      cache_entry->cut_context_signature_context_revision == context_revision &&
      cache_entry->cut_context_signature_marker == marker &&
      cache_entry->cut_context_signature_request_key == request_key) {
    if (cache_hit != nullptr) {
      *cache_hit = true;
    }
    ++cache_entry->cut_context_signature_cache_hits;
    return cache_entry->cut_context_signature;
  }
  if (cache_hit != nullptr) {
    *cache_hit = false;
  }
  if (cache_entry != nullptr) {
    ++cache_entry->cut_context_signature_cache_misses;
  }

  const auto volume_rules = cut_context->generatedVolumeRulesForMarker(*marker);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(volume_rules.size()));
  for (const auto* rule : volume_rules) {
    if (rule != nullptr) {
      mixCurvatureProjectionCutRuleSignature(seed, *rule);
    }
  }

  const auto interface_rules = cut_context->interfaceRulesForMarker(*marker);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(interface_rules.size()));
  for (const auto* rule : interface_rules) {
    if (rule != nullptr) {
      mixCurvatureProjectionCutRuleSignature(seed, *rule);
    }
  }
  if (cache_entry != nullptr) {
    cache_entry->cut_context_signature_valid = true;
    cache_entry->cut_context_signature_context = cut_context;
    cache_entry->cut_context_signature_context_revision = context_revision;
    cache_entry->cut_context_signature_marker = marker;
    cache_entry->cut_context_signature_request_key = request_key;
    cache_entry->cut_context_signature = seed;
  }
  return seed;
}

std::uint64_t curvatureProjectionInputSignature(
    const LevelSetMaintenanceRequest& request,
    const svmp::FE::level_set::LevelSetCurvatureProjectionOptions& options,
    std::uint64_t cut_context_signature,
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
                            options.max_neighbor_fallback_vertices));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.max_zero_fallback_vertices));
  mixCurvatureSignatureReal(seed, options.supplemental_sample_weight);
  mixCurvatureSignatureReal(seed, options.narrow_band_width);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.smoothing_iterations));
  mixCurvatureSignatureReal(seed, options.smoothing_relaxation);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.smoothing_mode));
  mixCurvatureSignature(seed, cut_context_signature);
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
    mixCurvatureSignature(
        seed, sample.free_surface_snapshot_revision_key);
    mixCurvatureSignature(seed, sample.source_value_revision);
    mixCurvatureSignature(seed, sample.cut_topology_revision);
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
    std::uint64_t cut_context_signature,
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
    const auto field_offset = system.fieldDofOffset(phi_field);
    const auto n_field_dofs = system.fieldDofHandler(phi_field).getNumDofs();
    std::optional<std::uint64_t> field_value_hash;
    if (field_offset >= 0 && n_field_dofs >= 0 &&
        state.u_vector->size() >= 0) {
      try {
        const auto values = state.u_vector->localSpan();
        const auto begin = static_cast<std::size_t>(field_offset);
        const auto count = static_cast<std::size_t>(n_field_dofs);
        const auto vector_size = static_cast<std::size_t>(state.u_vector->size());
        if (values.size() == vector_size &&
            begin <= values.size() &&
            count <= values.size() - begin) {
          std::uint64_t h = 0x510e527fade682d1ull;
          mixCurvatureSignature(h, static_cast<std::uint64_t>(begin));
          mixCurvatureSignature(h, static_cast<std::uint64_t>(count));
          for (std::size_t i = 0; i < count; ++i) {
            mixCurvatureSignatureReal(h, values[begin + i]);
          }
          field_value_hash = h;
        }
      } catch (...) {
        field_value_hash.reset();
      }
    }

    if (field_value_hash.has_value()) {
      source_kind = 2u;
      source_revision = *field_value_hash;
    } else {
      source_kind = 3u;
      source_revision = state.u_vector->valueRevision();
    }
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
                            options.max_neighbor_fallback_vertices));
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.max_zero_fallback_vertices));
  mixCurvatureSignatureReal(seed, options.supplemental_sample_weight);
  mixCurvatureSignatureReal(seed, options.narrow_band_width);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.smoothing_iterations));
  mixCurvatureSignatureReal(seed, options.smoothing_relaxation);
  mixCurvatureSignature(seed,
                        static_cast<std::uint64_t>(
                            options.smoothing_mode));
  mixCurvatureSignature(seed, cut_context_signature);
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
    std::uint64_t signature,
    std::string_view cut_signature_cache_state = "unavailable",
    std::size_t cut_signature_cache_hits = 0u,
    std::size_t cut_signature_cache_misses = 0u)
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
      << " free_surface_snapshot_revision_key="
      << result.free_surface_snapshot_revision_key
      << " source_value_revision=" << result.source_value_revision
      << " cut_rule_signature=" << result.cut_rule_signature
      << " cut_signature_cache=" << cut_signature_cache_state
      << " cut_signature_cache_hits=" << cut_signature_cache_hits
      << " cut_signature_cache_misses=" << cut_signature_cache_misses
      << " fitted_vertices=" << result.fitted_vertices
      << " supplemental_samples=" << result.supplemental_samples
      << " supplemental_sample_rows=" << result.supplemental_sample_rows
      << " vertices_with_supplemental_samples="
      << result.vertices_with_supplemental_samples
      << " supplemental_sample_weight=" << result.supplemental_sample_weight
      << " narrow_band_width=" << result.narrow_band_width
      << " narrow_band_vertices=" << result.narrow_band_vertices
      << " skipped_far_vertices=" << result.skipped_far_vertices
      << " fallback_vertices=" << result.fallback_vertices
      << " max_neighbor_fallback_vertices="
      << request.curvature_projection.max_neighbor_fallback_vertices
      << " zero_fallback_vertices=" << result.zero_fallback_vertices
      << " max_zero_fallback_vertices="
      << request.curvature_projection.max_zero_fallback_vertices
      << " insufficient_stencil_vertices="
      << result.insufficient_stencil_vertices
      << " singular_stencil_vertices=" << result.singular_stencil_vertices
      << " small_gradient_vertices=" << result.small_gradient_vertices
      << " fit_residual_failure_vertices="
      << result.fit_residual_failure_vertices
      << " smoothing_mode="
      << svmp::FE::level_set::levelSetCurvatureSmoothingModeName(
             result.smoothing_mode)
      << " smoothing_iterations="
      << result.smoothing_iterations_applied
      << " smoothing_operator_edges="
      << result.smoothing_operator_edges
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
  const auto n_field_dofs =
      static_cast<std::size_t>(field_dofs.getNumDofs());
  std::vector<svmp::FE::Real> coefficients(n_field_dofs, 0.0);
  std::vector<std::uint8_t> assigned(n_field_dofs, 0u);
  const auto projection =
      system.projectMeshVertexValuesToFieldCoefficients(
          field,
          vertex_values,
          1u,
          std::span<svmp::FE::Real>(coefficients.data(),
                                    coefficients.size()),
          std::span<std::uint8_t>(assigned.data(), assigned.size()),
          context);
  if (projection.unassigned_dofs != 0u) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " + std::string(context) +
        " left " + std::to_string(projection.unassigned_dofs) +
        " target field coefficient(s) without a safe mesh-vertex projection.");
  }
  system.setPrescribedFieldCoefficients(
      field,
      std::span<const svmp::FE::Real>(coefficients.data(),
                                      coefficients.size()));
}

std::size_t projectLevelSetCurvatureFieldsFromState(
    application::core::SimulationComponents& sim,
    const svmp::FE::systems::SystemStateView& state,
    const std::map<int, std::uint64_t>& evaluated_state_source_revisions,
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
    const bool cadence_due =
        !honor_cadence || shouldProjectLevelSetCurvature(request, step);
    if (!honor_cadence && request.curvature_field_name.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set curvature projection requires Curvature_field_name.");
    }
    CurvatureProjectionCacheEntry* cache_entry = nullptr;
    if (curvature_cache != nullptr) {
      const auto key = curvatureProjectionCacheKey(request);
      auto [it, inserted] = curvature_cache->entries.try_emplace(key);
      (void)inserted;
      cache_entry = &it->second;
    }
    if (honor_cadence && !cadence_due &&
        (cache_entry == nullptr || !cache_entry->valid)) {
      continue;
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

    auto reapply_cached_curvature_if_needed =
        [&](CurvatureProjectionCacheEntry& entry) {
      if (!entry.valid || entry.last_curvature_vertex_values.empty()) {
        return;
      }
      if (entry.last_curvature_vertex_values.size() !=
          static_cast<std::size_t>(mesh.n_vertices())) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Cached level-set curvature projection "
            "has an incompatible vertex count for field '" +
            request.level_set_field_name + "'.");
      }
      const auto current_revision = system.prescribedFieldRevision(kappa_field);
      if (current_revision == entry.target_prescribed_revision) {
        return;
      }
      setScalarPrescribedVertexFieldFromValues(
          system,
          mesh,
          kappa_field,
          std::span<const svmp::FE::Real>(
              entry.last_curvature_vertex_values.data(),
              entry.last_curvature_vertex_values.size()),
          "Cached level-set curvature projection");
      entry.target_prescribed_revision =
          system.prescribedFieldRevision(kappa_field);
    };

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
    bool cut_context_signature_cache_hit = false;
    const auto cut_context_signature =
        curvatureProjectionCutContextSignature(
            system, request, cache_entry, &cut_context_signature_cache_hit);
    const auto snapshot_identity =
        curvatureProjectionSnapshotIdentity(system, request);
    const auto interface_marker =
        generatedCutContextMarkerForMaintenance(system, request);
    std::uint64_t evaluated_state_source_revision = 0u;
    if (snapshot_identity.free_surface_snapshot_revision_key != 0u) {
      if (!interface_marker.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Revision-bound curvature "
            "projection has no authoritative interface marker.");
      }
      const auto evaluated_revision =
          evaluated_state_source_revisions.find(*interface_marker);
      if (evaluated_revision == evaluated_state_source_revisions.end() ||
          evaluated_revision->second == 0u ||
          evaluated_revision->second !=
              snapshot_identity.source_value_revision) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Curvature projection state does "
            "not match the authoritative geometry source revision.");
      }
      evaluated_state_source_revision = evaluated_revision->second;
    }
    const auto cut_context_signature_cache_state =
        cache_entry == nullptr
            ? std::string_view{"disabled"}
            : (cut_context_signature_cache_hit ? std::string_view{"hit"}
                                               : std::string_view{"miss"});

    std::optional<std::uint64_t> fast_signature;
    if (cache_entry != nullptr) {
      fast_signature = curvatureProjectionFastInputSignature(
          system,
          state,
          phi_field,
          request,
          options,
          cut_context_signature,
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
        reapply_cached_curvature_if_needed(*cache_entry);
        logLevelSetCurvatureProjectionDiagnostic(
            request,
            step,
            reason,
            cache_entry->last_result,
            "fast_hit",
            /*projection_skipped=*/true,
            cache_entry->signature,
            cut_context_signature_cache_state,
            cache_entry->cut_context_signature_cache_hits,
            cache_entry->cut_context_signature_cache_misses);
        continue;
      }
    }

    const auto phi_values =
        evaluateVertexField(system,
                            mesh,
                            phi_field,
                            state,
                            1u,
                            "projecting level-set curvature");
    std::vector<svmp::FE::Real> phi(
        phi_values.begin(), phi_values.end());
    const auto supplemental_samples =
        collectLevelSetCurvatureSupplementalSamples(
            system,
            state,
            phi_field,
            options.isovalue,
            interface_marker,
            request.volume_cut_request.has_value()
                ? std::optional<svmp::FE::geometry::CutIntegrationSide>(
                      cutIntegrationSide(request.volume_cut_request->active_side))
                : std::nullopt,
            evaluated_state_source_revision);
    std::vector<svmp::FE::Real> curvature;
    const auto signature = curvatureProjectionInputSignature(
        request,
        options,
        cut_context_signature,
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
        reapply_cached_curvature_if_needed(*cache_entry);
        logLevelSetCurvatureProjectionDiagnostic(
            request,
            step,
            reason,
            cache_entry->last_result,
            "hit",
            /*projection_skipped=*/true,
            signature,
            cut_context_signature_cache_state,
            cache_entry->cut_context_signature_cache_hits,
            cache_entry->cut_context_signature_cache_misses);
        continue;
      }
    }

    if (honor_cadence && !cadence_due) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] Level-set curvature projection "
          << "overriding cadence because level-set/cut signature changed"
          << " field='" << request.level_set_field_name << "'"
          << " curvature_field='" << request.curvature_field_name << "'"
          << " step=" << step
          << " reason=" << reason
          << " diagnostic=curvature_projection_cadence_override_changed_signature"
          << std::endl;
    }

    auto result =
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
    if (snapshot_identity.free_surface_snapshot_revision_key != 0u) {
      if (result.free_surface_snapshot_revision_key == 0u ||
          result.free_surface_snapshot_revision_key !=
              snapshot_identity.free_surface_snapshot_revision_key ||
          result.source_value_revision !=
              snapshot_identity.source_value_revision) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Curvature projection samples do "
            "not carry the authoritative geometry snapshot revision.");
      }
    }
    result.cut_rule_signature = cut_context_signature;
    if (cache_entry != nullptr) {
      cache_entry->workspace.free_surface_snapshot_revision_key =
          result.free_surface_snapshot_revision_key;
      cache_entry->workspace.source_value_revision =
          result.source_value_revision;
      cache_entry->workspace.cut_rule_signature = cut_context_signature;
    }
    if (!result.success) {
      if (reuse_cached_on_projection_failure &&
          cache_entry != nullptr &&
          cache_entry->valid &&
          cache_entry->last_result.free_surface_snapshot_revision_key ==
              result.free_surface_snapshot_revision_key &&
          cache_entry->last_result.source_value_revision ==
              result.source_value_revision &&
          cache_entry->last_result.cut_rule_signature ==
              result.cut_rule_signature) {
        reapply_cached_curvature_if_needed(*cache_entry);
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
            cache_entry->signature,
            cut_context_signature_cache_state,
            cache_entry->cut_context_signature_cache_hits,
            cache_entry->cut_context_signature_cache_misses);
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
      cache_entry->last_curvature_vertex_values = curvature;
      cache_entry->target_prescribed_revision =
          system.prescribedFieldRevision(kappa_field);
    }
    ++updated_fields;

    logLevelSetCurvatureProjectionDiagnostic(
        request,
        step,
        reason,
        result,
        cache_entry != nullptr ? "miss" : "disabled",
        /*projection_skipped=*/false,
        signature,
        cut_context_signature_cache_state,
        cache_entry != nullptr ? cache_entry->cut_context_signature_cache_hits : 0u,
        cache_entry != nullptr ? cache_entry->cut_context_signature_cache_misses : 0u);
  }

  return updated_fields;
}

struct ActiveLevelSetMeshFieldCheckpoint {
  struct Field {
    std::string name{};
    std::size_t components{0};
    std::size_t entity_count{0};
    std::vector<double> values{};
  };

  bool valid{false};
  svmp::MeshRevisionState revisions{};
  std::vector<Field> fields{};
};

ActiveLevelSetMeshFieldCheckpoint captureActiveLevelSetMeshFields(
    application::core::SimulationComponents& sim,
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
    application::core::SimulationComponents& sim,
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

bool updateLevelSetAdvectionVelocitiesFromState(
    application::core::SimulationComponents& sim,
    const svmp::FE::systems::SystemStateView& state,
    const std::vector<LevelSetAdvectionVelocityRequest>& requests,
    bool refresh_frozen_algebraic_map = true,
    std::vector<AcceptedVelocityExtensionMapRecord>* refreshed_maps = nullptr)
{
  if (!sim.fe_system || !sim.primary_mesh || requests.empty()) {
    return false;
  }

  auto& system = *sim.fe_system;
  const auto& mesh = *sim.primary_mesh;
  const auto extension_comm = activeFESystemCommunicator(system);
  const auto n_vertices = mesh.n_vertices();
  const int mesh_dim = mesh.dim();
  const auto& coords = mesh.X_ref();
  const bool trace_updates =
      parseBoolEnv("SVMP_TRACE_LEVEL_SET_ADVECTION", false) ||
      application::core::oopTraceEnabled();

  if (refreshed_maps != nullptr) {
    refreshed_maps->clear();
  }

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

    const auto& phi_rec = system.fieldRecord(phi_field);
    const auto& target_rec = system.fieldRecord(target_field);
    auto extension_constraint =
        svmp::FE::level_set::
            findLevelSetVelocityExtensionConstraintKernel(
                system, request.operator_tag, target_field);
    const bool algebraic_extension =
        target_rec.source_kind ==
            svmp::FE::systems::FieldSourceKind::Unknown &&
        static_cast<bool>(extension_constraint);
    if (target_rec.source_kind !=
            svmp::FE::systems::FieldSourceKind::PrescribedData &&
        !algebraic_extension) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity field '" +
          target_rec.name +
          "' must be either PrescribedData or an algebraic extension unknown with a registered exact constraint kernel.");
    }
    if (algebraic_extension &&
        extension_constraint->sourceVelocityField() != source_field) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Algebraic level-set velocity extension source does not match the registered physical velocity field.");
    }
    if (algebraic_extension && !refresh_frozen_algebraic_map) {
      // The map contains nonsmooth active-set/BFS choices and phi-dependent
      // regression weights.  Freeze all of them throughout one nonlinear
      // solve; only E and the physical source velocity remain monolithic.
      continue;
    }
    if (algebraic_extension) {
      // Rebuilding the cut/graph map is fail-closed.  Any exception below,
      // including disappearance of the interface, must not leave the previous
      // accepted interface map available to a later assembly.
      extension_constraint->invalidateFrozenMap();
    }

    const auto& source_rec = system.fieldRecord(source_field);
    if (source_rec.components <= 0 || target_rec.components <= 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity update found an empty velocity field.");
    }
    const auto source_components =
        static_cast<std::size_t>(source_rec.components);
    const auto target_components =
        static_cast<std::size_t>(target_rec.components);
    const auto copy_components =
        std::min(source_components, target_components);
    if (request.extension_method == "wall_compatible_normal" ||
        request.extension_method == "nearest_interface_point") {
      const auto require_p1_field = [](const auto& record,
                                       std::string_view role) {
        if (!record.space || record.space->is_variable_order() ||
            record.space->polynomial_order() != 1) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] wall_compatible_normal "
              "wet-extension requires a fixed P1 " +
              std::string(role) + " field; field '" + record.name +
              "' is higher-order, variable-order, or has no FE space.");
        }
      };
      if (phi_rec.components != 1) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] wall_compatible_normal "
            "wet-extension requires a scalar P1 level-set field.");
      }
      require_p1_field(phi_rec, "level-set");
      require_p1_field(source_rec, "source-velocity");
      require_p1_field(target_rec, "target-velocity");
      if (source_components != target_components ||
          source_rec.space->value_dimension() !=
              target_rec.space->value_dimension()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] wall_compatible_normal wet-extension requires source and target P1 fields with identical component layouts.");
      }
      const auto& phi_dofs = system.fieldDofHandler(phi_field);
      const auto& source_dofs = system.fieldDofHandler(source_field);
      const auto& target_dofs = system.fieldDofHandler(target_field);
      const auto* phi_entity_map = phi_dofs.getEntityDofMap();
      const auto* source_entity_map = source_dofs.getEntityDofMap();
      const auto* target_entity_map = target_dofs.getEntityDofMap();
      if (phi_entity_map == nullptr || source_entity_map == nullptr ||
          target_entity_map == nullptr) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] wall_compatible_normal wet-extension requires matching vertex-nodal level-set, source, and target layouts.");
      }
      for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        const auto phi_vertex_dofs = phi_entity_map->getVertexDofs(
            static_cast<svmp::FE::GlobalIndex>(vertex));
        const auto source_vertex_dofs = source_entity_map->getVertexDofs(
            static_cast<svmp::FE::GlobalIndex>(vertex));
        const auto target_vertex_dofs = target_entity_map->getVertexDofs(
            static_cast<svmp::FE::GlobalIndex>(vertex));
        if (phi_vertex_dofs.size() != 1u ||
            source_vertex_dofs.size() != source_components ||
            target_vertex_dofs.size() != target_components) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] wall_compatible_normal wet-extension requires exactly one level-set, source, and target P1 vertex DOF per component.");
        }
      }
      const auto& local_mesh = mesh.local_mesh();
      for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
        if (local_mesh.cell_shape(cell).order != 1) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] wall_compatible_normal "
              "wet-extension requires linear mesh topology; high-order "
              "geometry is not qualified for the vertex graph algorithm.");
        }
        auto [cell_vertices, cell_vertex_count] =
            local_mesh.cell_vertices_span(cell);
        const auto expected_cell_dofs =
            cell_vertex_count * source_components;
        if (cell_vertices == nullptr || cell_vertex_count == 0u ||
            phi_rec.space->dofs_per_element(cell) != cell_vertex_count ||
            source_rec.space->dofs_per_element(cell) != expected_cell_dofs ||
            target_rec.space->dofs_per_element(cell) != expected_cell_dofs) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] wall_compatible_normal wet-extension requires identical nodal P1 source/target cell layouts on every retained mesh cell.");
        }
      }
    }

    const auto phi_values =
        evaluateVertexField(system,
                            mesh,
                            phi_field,
                            state,
                            1u,
                            "updating level-set advection velocity");
    const auto source_values =
        evaluateVertexField(system,
                            mesh,
                            source_field,
                            state,
                            source_components,
                            "updating level-set advection velocity");

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
    bool interface_sample_used_cut_context = false;
    std::size_t interface_sample_candidate_cell_count = 0u;
    std::size_t interface_sample_count = 0u;
    std::size_t interface_sample_context_rule_count = 0u;
    std::size_t trace_support_vertex_count = 0u;
    std::size_t dry_trace_support_vertex_count = 0u;
    std::size_t trace_seed_vertex_count = 0u;
    std::size_t wall_boundary_label_count = 0u;
    WallCompatibleVelocityExtensionResult wall_extension_report{};
    std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>
        algebraic_rows;
    std::shared_ptr<const application::core::VelocityExtensionMapSnapshot>
        algebraic_map_snapshot;

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

    if (request.extension_method == "wall_compatible_normal" ||
        request.extension_method == "nearest_interface_point") {
      if (mesh_dim < 1 || mesh_dim > 3) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] nearest_interface_point level-set "
            "advection velocity extension requires mesh dimension in [1, 3].");
      }

      // For fixed nodal P1/Q1 fields, equality at every vertex supporting a
      // retained interface cell is both necessary and sufficient for the
      // finite-element traces to agree pointwise on that cell:
      //
      //     E_h|_K = u_h|_K  =>  E_h|_(Gamma cap K) = u_h|_(Gamma cap K).
      //
      // The older sign-only seed marked wet vertices but graph-extrapolated
      // the dry vertices of cut cells, so E_h and u_h differed on Gamma even
      // though all wet nodal values agreed.  Prefer the generated cut context
      // (the geometry used by assembly), with a nodal P1 crossing search only
      // when no authoritative generated-interface context exists anywhere on
      // the active communicator.  A locally empty retained-rule set is
      // authoritative and must not resurrect an omitted cell through a nodal
      // fallback on that rank.
      const bool authoritative_interface_context = globalAnyBool(
          hasAuthoritativeInterfaceVelocityContext(system, request),
          extension_comm);
      auto candidate_cells = authoritative_interface_context
                                 ? interfaceVelocitySampleCandidateCells(
                                       system, request)
                                 : std::vector<svmp::FE::MeshIndex>{};
      if (!authoritative_interface_context) {
        candidate_cells = nodalVelocityExtensionInterfaceCells(
            mesh,
            std::span<const double>(phi_values.data(), phi_values.size()),
            request.isovalue);
      }
      interface_sample_used_cut_context = authoritative_interface_context;
      const auto local_owned_candidate_cells =
          static_cast<std::size_t>(std::count_if(
              candidate_cells.begin(),
              candidate_cells.end(),
              [&](svmp::FE::MeshIndex cell) {
                return cell >= 0 && mesh.is_owned_cell(
                                        static_cast<svmp::index_t>(cell));
              }));
      interface_sample_candidate_cell_count = globalSumSize(
          local_owned_candidate_cells, extension_comm);

      std::vector<std::uint8_t> trace_support(n_vertices, 0u);
      markVelocityExtensionTraceSupportCells(
          mesh,
          std::span<const svmp::FE::MeshIndex>(candidate_cells.data(),
                                               candidate_cells.size()),
          trace_support);
      trace_support_vertex_count =
          synchronizeVelocityExtensionTraceSupportMask(
              mesh, extension_comm, trace_support);
      std::vector<std::uint8_t> trace_seed(active);
      std::vector<std::uint8_t> dry_trace_support(n_vertices, 0u);
      for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        if (trace_support[vertex] != 0u) {
          trace_seed[vertex] = 1u;
          if (active[vertex] == 0u) {
            dry_trace_support[vertex] = 1u;
          }
        }
      }
      dry_trace_support_vertex_count =
          globalOwnedVelocityExtensionMaskCount(
              mesh, extension_comm, dry_trace_support);
      trace_seed_vertex_count = globalOwnedVelocityExtensionMaskCount(
          mesh, extension_comm, trace_seed);

      std::size_t local_interface_geometry_sample_count = 0u;

      auto add_interface_sample = [&](std::size_t va,
                                      std::size_t vb,
                                      double edge_t) {
        if (va >= n_vertices || vb >= n_vertices ||
            !std::isfinite(edge_t)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] wall_compatible_normal wet-extension found an invalid nodal interface geometry sample.");
        }
        edge_t = std::clamp(edge_t, 0.0, 1.0);
        (void)edge_t;
        ++local_interface_geometry_sample_count;
      };

      auto add_cut_context_interface_samples = [&]() -> bool {
        const auto* cut_context = system.cutIntegrationContext();
        const auto marker = interfaceVelocitySampleMarker(system, request);
        if (cut_context == nullptr || !marker.has_value() ||
            !cut_context->hasGeneratedInterfaceMarker(*marker)) {
          return false;
        }
        bool added = false;
        for (const auto* rule :
             cut_context->interfaceRulesForMarker(*marker)) {
          if (rule == nullptr ||
              rule->kind != svmp::FE::geometry::CutQuadratureKind::Interface ||
              rule->provenance.parent_entity < 0) {
            continue;
          }
          ++interface_sample_context_rule_count;
          for (const auto& qp : rule->points) {
            if (!std::isfinite(qp.weight) ||
                !(std::abs(qp.weight) > 0.0)) {
              continue;
            }
            const auto& point = qp.point;
            if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
                !std::isfinite(point[2])) {
              throw std::runtime_error(
                "[svMultiPhysics::Application] nearest_interface_point level-set "
                "advection velocity extension received a non-finite cut-interface sample.");
            }
            ++local_interface_geometry_sample_count;
            added = true;
          }
        }
        return added;
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
      const bool used_cut_context_samples =
          add_cut_context_interface_samples();
      interface_sample_context_rule_count = globalSumSize(
          interface_sample_context_rule_count, extension_comm);

      auto visit_cell_edges = [&](svmp::index_t cell) {
        if (cell < 0 || cell >= local_mesh.n_cells()) {
          return;
        }
        auto [cell_vertices, n_cell_vertices] =
            local_mesh.cell_vertices_span(cell);
        if (cell_vertices == nullptr || n_cell_vertices < 2u) {
          return;
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
            return;
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
          return;
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
      };
      if (!used_cut_context_samples && mesh_dim == 2) {
        if (authoritative_interface_context) {
          for (const auto cell : candidate_cells) {
            visit_cell_edges(static_cast<svmp::index_t>(cell));
          }
        } else {
          for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
            visit_cell_edges(cell);
          }
        }
      }

      interface_sample_count = globalVelocityExtensionGeometrySampleCount(
          local_interface_geometry_sample_count, extension_comm);
      if (interface_sample_count == 0u) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] wall_compatible_normal wet-extension found no resolved interface geometry samples on any communicator rank; refusing to retain or install a stale extension map.");
      }
      if (trace_support_vertex_count == 0u) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] wall_compatible_normal wet-extension resolved interface geometry but no P1 trace-support vertices.");
      }

      std::vector<WallVelocityExtensionConstraint> wall_constraints;
      wall_constraints.reserve(request.wall_constraints.size());
      if (request.enforce_wall_impermeability) {
        if (request.wall_constraints.empty()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Wet-extension wall "
              "impermeability is enabled, but no strong homogeneous "
              "velocity Dirichlet component masks were resolved. Specify "
              "compatible wall BCs or disable "
              "Wet_extension_enforce_wall_impermeability.");
        }
        for (const auto& wall : request.wall_constraints) {
          const auto label = mesh.label_from_name(wall.face_name);
          if (label == svmp::INVALID_LABEL) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Wet-extension wall face '" +
                wall.face_name + "' is not registered on the active mesh.");
          }
          wall_constraints.push_back(WallVelocityExtensionConstraint{
              .boundary_label = label,
              .constrained_components = strongZeroVelocityComponentMask(
                  wall.effective_direction, mesh_dim)});
        }
      }
      wall_boundary_label_count = wall_constraints.size();

      const auto oriented_level_set = orientedLevelSetForVelocityExtension(
          phi_values, request.isovalue, request.active_side);

      if (algebraic_extension) {
        std::uint64_t free_surface_geometry_revision = 0u;
        if (const auto* cut_context = system.cutIntegrationContext()) {
          free_surface_geometry_revision = cut_context->contentRevision();
          if (const auto marker = interfaceVelocitySampleMarker(system, request);
              marker.has_value() &&
              cut_context->hasFreeSurfaceGeometrySnapshotForMarker(*marker)) {
            free_surface_geometry_revision =
                cut_context->freeSurfaceGeometrySnapshotRevisionForMarker(
                    *marker);
          }
        }
        const auto& mesh_access = system.meshAccess();
        const auto map_revision =
            application::core::velocityExtensionMapRevision(
                mesh_access.geometryRevision(),
                mesh_access.topologyRevision(),
                mesh_access.ownershipRevision(),
                mesh_access.numberingRevision(),
                free_surface_geometry_revision,
                std::span<const double>(oriented_level_set.data(),
                                        oriented_level_set.size()),
                std::span<const std::uint8_t>(trace_seed.data(),
                                              trace_seed.size()));
        algebraic_map_snapshot =
            application::core::buildVelocityExtensionMapSnapshot(
                mesh,
                extension_comm,
                map_revision,
                std::span<const double>(oriented_level_set.data(),
                                        oriented_level_set.size()),
                std::span<const double>(source_values.data(),
                                        source_values.size()),
                source_components,
                std::span<const std::uint8_t>(trace_seed.data(),
                                              trace_seed.size()),
                target_components,
                copy_components,
                request.extension_band_layers,
                request.enforce_wall_impermeability,
                std::span<const WallVelocityExtensionConstraint>(
                    wall_constraints));
        wall_extension_report = algebraic_map_snapshot->report();
        extended.assign(algebraic_map_snapshot->preview().begin(),
                        algebraic_map_snapshot->preview().end());
        algebraic_rows = algebraic_map_snapshot->copyRows();
      } else {
        wall_extension_report = extendVelocityInLevelSetNormalBand(
            mesh,
            extension_comm,
            std::span<const double>(oriented_level_set.data(),
                                    oriented_level_set.size()),
            std::span<const double>(source_values.data(), source_values.size()),
            source_components,
            std::span<const std::uint8_t>(trace_seed.data(), trace_seed.size()),
            target_components,
            copy_components,
            request.extension_band_layers,
            request.enforce_wall_impermeability,
            std::span<const WallVelocityExtensionConstraint>(wall_constraints),
            extended);
      }
      constexpr double trace_equality_factor = 64.0;
      for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        if (trace_support[vertex] == 0u) {
          continue;
        }
        for (std::size_t component = 0; component < copy_components;
             ++component) {
          const double source =
              source_values[vertex * source_components + component];
          const double extension =
              extended[vertex * target_components + component];
          const double tolerance =
              trace_equality_factor * std::numeric_limits<double>::epsilon() *
              std::max({1.0, std::abs(source), std::abs(extension)});
          if (!std::isfinite(source) || !std::isfinite(extension) ||
              std::abs(extension - source) > tolerance) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] wall_compatible_normal wet-extension failed the exact P1 trace-support equality invariant.");
          }
        }
      }
      for (std::size_t v = 0; v < n_vertices; ++v) {
        record_speed(v);
      }
    } else {
      if (algebraic_extension) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Algebraic wet-extension tangents are implemented only for wall_compatible_normal; a state-dependent prescribed fallback is not permitted.");
      }
      std::vector<VelocityExtensionSample> active_samples;
      active_samples.reserve(active_vertices.size());
      for (const auto candidate : active_vertices) {
        VelocityExtensionSample sample;
        sample.point = meshVertexPoint(coords, mesh_dim, candidate);
        sample.value.assign(target_components, 0.0);
        for (std::size_t c = 0; c < copy_components; ++c) {
          const auto value =
              source_values[candidate * source_components + c];
          sample.value[c] = std::isfinite(value) ? value : 0.0;
        }
        active_samples.push_back(std::move(sample));
      }
      active_samples =
          gatherVelocityExtensionSamples(std::move(active_samples),
                                         target_components,
                                         extension_comm);
      if (active_samples.empty()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set advection velocity "
            "extension found no active velocity samples on any MPI rank.");
      }

      std::vector<NearestPointRecord> active_records;
      active_records.reserve(active_samples.size());
      for (std::size_t s = 0; s < active_samples.size(); ++s) {
        active_records.push_back(
            NearestPointRecord{active_samples[s].point, s});
      }
      const NearestPointIndex active_index(mesh_dim, std::move(active_records));

      for (std::size_t v = 0; v < n_vertices; ++v) {
        const auto nearest =
            active_index.nearest(meshVertexPoint(coords, mesh_dim, v));
        if (!nearest.found || nearest.payload >= active_samples.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set advection velocity extension failed to find a nearest active sample.");
        }

        const auto source_sample = nearest.payload;
        for (std::size_t c = 0; c < copy_components; ++c) {
          extended[v * target_components + c] =
              active_samples[source_sample].value[c];
        }
        record_speed(v);
      }
    }

    max_active_speed = globalMaxDouble(max_active_speed, extension_comm);
    max_dry_extended_speed =
        globalMaxDouble(max_dry_extended_speed, extension_comm);
    const double velocity_scale_floor = 1.0e-12;
    const double wet_to_dry_amplification =
        max_dry_extended_speed /
        std::max(max_active_speed, velocity_scale_floor);
    if (algebraic_extension &&
        wet_to_dry_amplification >
            kVelocityExtensionMaxWetToDryAmplification) {
      std::ostringstream message;
      message
          << "[svMultiPhysics::Application] Algebraic wet-extension refused "
             "to install a frozen map because the dry-band preview "
             "amplification exceeded its fixed guard: max_active_speed="
          << max_active_speed
          << " max_dry_extended_speed=" << max_dry_extended_speed
          << " amplification=" << wet_to_dry_amplification
          << " guard=" << kVelocityExtensionMaxWetToDryAmplification;
      throw std::runtime_error(message.str());
    }

    const auto& target_dofs = system.fieldDofHandler(target_field);
    const auto* entity_map = target_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity update requires vertex DOFs.");
    }
    std::size_t unassigned = 0u;
    if (algebraic_extension) {
      if (!algebraic_map_snapshot || algebraic_rows.empty()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Algebraic wet-extension refresh produced no immutable map snapshot or owned constraint rows.");
      }

      std::vector<svmp::FE::Real> projected_coefficients(
          static_cast<std::size_t>(target_dofs.getNumDofs()),
          svmp::FE::Real{0.0});
      std::vector<std::uint8_t> projected_assignment(
          projected_coefficients.size(), 0u);
      const auto projection =
          system.projectMeshVertexValuesToFieldCoefficients(
              target_field,
              std::span<const svmp::FE::Real>(extended.data(),
                                              extended.size()),
              target_components,
              std::span<svmp::FE::Real>(projected_coefficients.data(),
                                        projected_coefficients.size()),
              std::span<std::uint8_t>(projected_assignment.data(),
                                      projected_assignment.size()),
              "ApplicationDriver::refreshAlgebraicVelocityExtension");
      if (projection.unassigned_dofs != 0u || state.u_vector == nullptr) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Algebraic wet-extension map refresh requires a mutable state vector and a complete P1 preview projection.");
      }
      const auto owned_field_dofs =
          target_dofs.getPartition().locallyOwned().toVector();
      std::vector<svmp::FE::GlobalIndex> state_dofs;
      std::vector<svmp::FE::Real> state_values;
      state_dofs.reserve(owned_field_dofs.size());
      state_values.reserve(owned_field_dofs.size());
      const auto target_offset = system.fieldDofOffset(target_field);
      for (const auto local_dof : owned_field_dofs) {
        if (local_dof < 0 ||
            static_cast<std::size_t>(local_dof) >=
                projected_coefficients.size() ||
            projected_assignment[static_cast<std::size_t>(local_dof)] == 0u) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Algebraic wet-extension preview omitted an owned extension DOF.");
        }
        state_dofs.push_back(target_offset + local_dof);
        state_values.push_back(
            projected_coefficients[static_cast<std::size_t>(local_dof)]);
      }
      auto* mutable_state =
          const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
      auto state_view = mutable_state->createAssemblyView();
      if (!state_view) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Algebraic wet-extension map refresh could not create a mutable state view.");
      }
      const bool map_changed =
          !extension_constraint->hasFrozenMap() ||
          extension_constraint->frozenMapRevision() !=
              algebraic_map_snapshot->revision().key();
      extension_constraint->setFrozenRows(
          std::move(algebraic_rows),
          algebraic_map_snapshot->revision().key());
      state_view->beginAssemblyPhase();
      state_view->setVectorEntries(
          std::span<const svmp::FE::GlobalIndex>(state_dofs.data(),
                                                 state_dofs.size()),
          std::span<const svmp::FE::Real>(state_values.data(),
                                          state_values.size()));
      state_view->endAssemblyPhase();
      state_view->finalizeAssembly();
      mutable_state->markModified();
      mutable_state->updateGhosts();
      if (trace_updates) {
        application::core::oopCout()
            << "[svMultiPhysics::Application] Accepted algebraic wet-extension map refresh"
            << " map_revision="
            << algebraic_map_snapshot->revision().key()
            << " map_changed=" << (map_changed ? 1 : 0)
            << " reprojected_owned_dofs=" << state_dofs.size()
            << std::endl;
      }
    } else {
      std::vector<svmp::FE::Real> coefficients(
          static_cast<std::size_t>(target_dofs.getNumDofs()),
          svmp::FE::Real{0.0});
      std::vector<std::uint8_t> assigned(coefficients.size(), 0u);

      const auto projection =
          system.projectMeshVertexValuesToFieldCoefficients(
              target_field,
              std::span<const svmp::FE::Real>(extended.data(),
                                              extended.size()),
              target_components,
              std::span<svmp::FE::Real>(coefficients.data(),
                                        coefficients.size()),
              std::span<std::uint8_t>(assigned.data(), assigned.size()),
              "ApplicationDriver::updateLevelSetAdvectionVelocities");
      if (projection.unassigned_dofs != 0u) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set advection velocity update "
            "could not safely project " +
            std::to_string(projection.unassigned_dofs) +
            " target coefficient(s) from mesh vertices.");
      }
      for (const auto flag : assigned) {
        unassigned += flag ? 0u : 1u;
      }
      system.setPrescribedFieldCoefficients(
          target_field,
          std::span<const svmp::FE::Real>(coefficients.data(),
                                          coefficients.size()));
    }
    if (refreshed_maps != nullptr && algebraic_map_snapshot) {
      refreshed_maps->push_back(AcceptedVelocityExtensionMapRecord{
          .level_set_field_name = request.level_set_field_name,
          .source_velocity_field_name = request.source_velocity_field_name,
          .target_velocity_field_name = request.target_velocity_field_name,
          .geometry_domain_id = request.domain_id,
          .operator_tag = request.operator_tag,
          .extension_method = request.extension_method,
          .isovalue = request.isovalue,
          .extension_band_layers = request.extension_band_layers,
          .enforce_wall_impermeability =
              request.enforce_wall_impermeability,
          .retained_side = request.active_side,
          .snapshot = algebraic_map_snapshot,
      });
    }
    updated = true;
    if (trace_updates) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] Updated level-set advection velocity field='"
          << request.target_velocity_field_name << "' from source='"
          << request.source_velocity_field_name << "' extension_method="
          << request.extension_method << " linearization="
          << (algebraic_extension ? "frozen_sparse_algebraic" :
                                    "prescribed_legacy")
          << " active_vertices=" << active_vertices.size() << " dry_vertices="
          << (n_vertices - active_vertices.size())
          << " interface_sample_candidates="
          << interface_sample_candidate_cell_count
          << " interface_sample_context_rules="
          << interface_sample_context_rule_count
          << " interface_samples=" << interface_sample_count
          << " interface_geometry_samples=" << interface_sample_count
          << " interface_geometry_sample_scope=summed_rank_local"
          << " interface_geometry_sample_exchange=checked_scalar_allreduce"
          << " interface_samples_used_for_constraints=0"
          << " interface_sample_source="
          << (interface_sample_used_cut_context ? "cut_context" : "all_cells")
          << " trace_constraint_source=cut_cell_p1_support"
          << " trace_support_vertices=" << trace_support_vertex_count
          << " dry_trace_support_vertices="
          << dry_trace_support_vertex_count
          << " trace_seed_vertices=" << trace_seed_vertex_count
          << " extension_band_layers=" << request.extension_band_layers
          << " extended_vertices="
          << wall_extension_report.extended_vertices
          << " vertices_outside_extension_band="
          << wall_extension_report.vertices_outside_band
          << " wall_projected_vertices="
          << wall_extension_report.wall_projected_vertices
          << " component_collision_vertices="
          << wall_extension_report.component_collision_vertices
          << " regression_candidate_rows="
          << wall_extension_report.regression_candidate_rows
          << " regression_accepted_rows="
          << wall_extension_report.regression_accepted_rows
          << " bounded_fallback_rows="
          << wall_extension_report.bounded_fallback_rows
          << " condition_rejected_rows="
          << wall_extension_report.condition_rejected_rows
          << " coefficient_rejected_rows="
          << wall_extension_report.coefficient_rejected_rows
          << " map_revision="
          << (algebraic_map_snapshot
                  ? algebraic_map_snapshot->revision().key()
                  : 0u)
          << " regression_condition_guard="
          << kVelocityExtensionMaxRegressionCondition
          << " max_regression_condition="
          << wall_extension_report.max_regression_condition
          << " graph_coefficient_guard="
          << (1.0 + kVelocityExtensionRowTolerance)
          << " max_abs_graph_coefficient="
          << wall_extension_report.max_abs_graph_coefficient
          << " graph_row_l1_guard="
          << (1.0 + kVelocityExtensionRowTolerance)
          << " max_graph_row_l1="
          << wall_extension_report.max_graph_row_l1
          << " graph_row_sum_tolerance="
          << kVelocityExtensionRowTolerance
          << " max_graph_row_sum_error="
          << wall_extension_report.max_graph_row_sum_error
          << " max_negative_graph_coefficient="
          << wall_extension_report.max_negative_graph_coefficient
          << " max_constant_reproduction_error="
          << wall_extension_report.max_constant_reproduction_error
          << " max_tangential_linear_reproduction_error="
          << wall_extension_report.max_linear_reproduction_error
          << " max_extrapolation_distance="
          << wall_extension_report.max_extrapolation_distance
          << " map_max_seed_speed="
          << wall_extension_report.max_seed_speed
          << " map_max_extended_speed="
          << wall_extension_report.max_extended_speed
          << " wall_boundary_labels=" << wall_boundary_label_count
          << " max_wall_normal_velocity="
          << wall_extension_report.max_wall_normal_velocity
          << " wall_normal_scope=graph_extended_vertices_outside_trace_seed"
          << " max_active_speed=" << max_active_speed
          << " max_dry_extended_speed=" << max_dry_extended_speed
          << " wet_to_dry_amplification=" << wet_to_dry_amplification
          << " wet_to_dry_amplification_guard="
          << kVelocityExtensionMaxWetToDryAmplification
          << " target_value_semantics="
          << (algebraic_extension ? "generalized_alpha_stage_unknown" :
                                    "prescribed_current_coefficients")
          << " computed_extension_semantics=current_state_constraint_preview"
          << " unassigned_dofs=" << unassigned << std::endl;
    }
  }

  return updated;
}

bool updateLevelSetAdvectionVelocities(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    const std::vector<LevelSetAdvectionVelocityRequest>& requests,
    std::vector<AcceptedVelocityExtensionMapRecord>* refreshed_maps = nullptr)
{
  const auto state = stateViewForHistory(history);
  return updateLevelSetAdvectionVelocitiesFromState(
      sim, state, requests, true, refreshed_maps);
}

using AcceptedVelocityExtensionMapRegistry =
    std::map<std::string,
             std::shared_ptr<const
                 application::core::VelocityExtensionMapSnapshot>>;

std::string acceptedVelocityExtensionMapKey(
    const AcceptedVelocityExtensionMapRecord& record)
{
  std::ostringstream key;
  key << std::setprecision(std::numeric_limits<double>::max_digits10);
  const auto append = [&](std::string_view value) {
    key << value.size() << ':' << value << ';';
  };
  append(record.level_set_field_name);
  append(record.source_velocity_field_name);
  append(record.target_velocity_field_name);
  append(record.geometry_domain_id);
  append(record.operator_tag);
  append(record.extension_method);
  key << record.isovalue << ';' << record.extension_band_layers << ';'
      << (record.enforce_wall_impermeability ? 1 : 0) << ';'
      << static_cast<int>(record.retained_side);
  return key.str();
}

std::uint64_t acceptedVelocityExtensionMapKeyFingerprint(
    std::string_view key) noexcept
{
  std::uint64_t fingerprint = 1469598103934665603ull;
  for (const char character : key) {
    fingerprint ^=
        static_cast<std::uint64_t>(static_cast<unsigned char>(character));
    fingerprint *= 1099511628211ull;
  }
  return fingerprint == 0u ? 1u : fingerprint;
}

std::optional<std::uint64_t> acceptedVelocityExtensionMapRegistryRevision(
    const AcceptedVelocityExtensionMapRegistry& accepted_maps) noexcept
{
  if (accepted_maps.empty()) {
    return std::nullopt;
  }
  std::uint64_t revision = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto mix = [&](std::uint64_t value) {
    for (std::size_t byte = 0u; byte < sizeof(value); ++byte) {
      revision ^= (value >> (byte * 8u)) & 0xffu;
      revision *= prime;
    }
  };
  for (const auto& [key, snapshot] : accepted_maps) {
    mix(acceptedVelocityExtensionMapKeyFingerprint(key));
    mix(snapshot ? snapshot->revision().key() : 0u);
  }
  return revision == 0u ? std::optional<std::uint64_t>{1u}
                        : std::optional<std::uint64_t>{revision};
}

std::optional<std::uint64_t> currentLevelSetVelocityExtensionRevision(
    const svmp::FE::systems::FESystem& system,
    const std::vector<LevelSetAdvectionVelocityRequest>& requests,
    const AcceptedVelocityExtensionMapRegistry& accepted_maps)
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
    mix(acceptedVelocityExtensionMapKeyFingerprint(
        request.level_set_field_name));
    mix(acceptedVelocityExtensionMapKeyFingerprint(
        request.target_velocity_field_name));
    mix(acceptedVelocityExtensionMapKeyFingerprint(
        request.operator_tag));
    mix(static_cast<std::uint64_t>(target));
    mix(kernel->frozenMapRevision());
  }
  const auto resolved =
      !found_current
          ? acceptedVelocityExtensionMapRegistryRevision(accepted_maps)
          : (revision == 0u
                 ? std::optional<std::uint64_t>{1u}
                 : std::optional<std::uint64_t>{revision});
  const auto [minimum, maximum] = globalMinMaxUint64(
      resolved.value_or(0u), activeFESystemCommunicator(system));
  if (minimum != maximum) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance extension-map revision differs across the FE communicator.");
  }
  return resolved;
}

void writeAcceptedVelocityExtensionMapArtifacts(
    const Parameters& params,
    const std::vector<AcceptedVelocityExtensionMapRecord>& records,
    std::uint64_t accepted_step,
    double accepted_time,
    double time_step,
    std::uint64_t state_revision,
    const svmp::MeshComm& comm,
    AcceptedVelocityExtensionMapRegistry& accepted_maps)
{
  const double local_record_count = static_cast<double>(records.size());
  if (globalMinDouble(local_record_count, comm) !=
      globalMaxDouble(local_record_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted velocity-extension map count differs across ranks.");
  }
  if (records.empty()) {
    return;
  }

  std::filesystem::path output_directory = ".";
  if (params.general_simulation_parameters.save_results_in_folder.defined() &&
      !params.general_simulation_parameters.save_results_in_folder
           .value()
           .empty()) {
    output_directory =
        params.general_simulation_parameters.save_results_in_folder.value();
  }
  output_directory /= "velocity_extension_maps";

  bool local_preflight_failure = accepted_step == 0u ||
                                 !std::isfinite(accepted_time) ||
                                 !std::isfinite(time_step) || time_step < 0.0;
  std::vector<std::string> record_keys;
  record_keys.reserve(records.size());
  std::set<std::string> unique_record_keys;
  for (const auto& record : records) {
    const auto key = acceptedVelocityExtensionMapKey(record);
    record_keys.push_back(key);
    local_preflight_failure =
        local_preflight_failure || !record.snapshot ||
        !unique_record_keys.insert(key).second;
    const auto fingerprint =
        acceptedVelocityExtensionMapKeyFingerprint(key);
    const auto [minimum_fingerprint, maximum_fingerprint] =
        globalMinMaxUint64(fingerprint, comm);
    local_preflight_failure = local_preflight_failure ||
                              minimum_fingerprint != maximum_fingerprint;
  }
  if (globalAnyBool(local_preflight_failure, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted velocity-extension artifact preflight failed on at least one rank.");
  }

  std::vector<application::core::VelocityExtensionMapArtifactResult>
      artifacts;
  artifacts.reserve(records.size());
  for (std::size_t index = 0u; index < records.size(); ++index) {
    const auto& record = records[index];
    const auto previous = accepted_maps.find(record_keys[index]);
    const auto* previous_snapshot =
        previous == accepted_maps.end() ? nullptr : previous->second.get();
    const application::core::VelocityExtensionMapArtifactContext context{
        .level_set_field_name = record.level_set_field_name,
        .source_velocity_field_name = record.source_velocity_field_name,
        .target_velocity_field_name = record.target_velocity_field_name,
        .geometry_domain_id = record.geometry_domain_id,
        .operator_tag = record.operator_tag,
        .extension_method = record.extension_method,
        .retained_side = activeSideName(record.retained_side),
        .accepted_step = accepted_step,
        .accepted_time = accepted_time,
        .time_step = time_step,
        .state_revision = state_revision,
        .isovalue = record.isovalue,
        .extension_band_layers = record.extension_band_layers,
        .enforce_wall_impermeability =
            record.enforce_wall_impermeability,
        .rank = comm.rank(),
        .ranks = comm.size(),
    };
    artifacts.push_back(
        application::core::writeVelocityExtensionMapArtifact(
            output_directory,
            context,
            *record.snapshot,
            previous_snapshot));
  }

  const bool local_publication_failure = std::any_of(
      artifacts.begin(), artifacts.end(), [](const auto& artifact) {
        return !artifact.success;
      });
  if (globalAnyBool(local_publication_failure, comm)) {
    bool local_rollback_failure = false;
    for (const auto& artifact : artifacts) {
      if (!artifact.success || artifact.path.empty()) {
        continue;
      }
      std::error_code removal_error;
      if (!std::filesystem::remove(artifact.path, removal_error) ||
          removal_error) {
        local_rollback_failure = true;
      }
    }
    if (globalAnyBool(local_rollback_failure, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted velocity-extension artifact publication and collective rollback both failed.");
    }
    std::string local_diagnostic;
    for (const auto& artifact : artifacts) {
      if (!artifact.success) {
        local_diagnostic = artifact.diagnostic;
        break;
      }
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted velocity-extension artifact publication failed: " +
        local_diagnostic);
  }

  for (std::size_t index = 0u; index < records.size(); ++index) {
    accepted_maps[record_keys[index]] = records[index].snapshot;
    if (comm.rank() == 0) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] Accepted velocity-extension map artifact"
          << " diagnostic=velocity_extension_map_artifact"
          << " field='" << records[index].target_velocity_field_name << "'"
          << " step=" << accepted_step
          << " map_revision=" << records[index].snapshot->revision().key()
          << " ranks=" << comm.size()
          << " owner_rows_rank0=" << artifacts[index].owner_rows
          << " constraint_rows_rank0=" << artifacts[index].constraint_rows
          << " bytes_rank0=" << artifacts[index].bytes
          << " path_rank0='" << artifacts[index].path.string() << "'"
          << " outcome=written" << std::endl;
    }
  }
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
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
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
    options.affected_cell_neighborhood_layers =
        request.affected_cell_neighborhood_layers;
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
    application::core::validateEquationLevelCutVolumeConsumer(
        *sim.fe_system, request, result.interface_marker);

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
                application::core::ActiveCutVolumeRetention::ActiveAndInactive
            ? std::optional<svmp::FE::geometry::CutIntegrationSide>{}
            : std::optional<svmp::FE::geometry::CutIntegrationSide>{
                  active_volume_side};
    const auto domain_volume_rules = result.domain.volumeQuadratureRules();
    const auto available_volume_rule_count =
        [&domain_volume_rules](svmp::FE::geometry::CutIntegrationSide side) {
          return static_cast<std::size_t>(std::count_if(
              domain_volume_rules.begin(),
              domain_volume_rules.end(),
              [side](const auto& rule) {
                return rule.kind ==
                           svmp::FE::geometry::CutQuadratureKind::Volume &&
                       rule.side == side;
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
      application::core::oopCout()
          << "[svMultiPhysics::Application] Generated interface-boundary "
             "intersection"
          << " diagnostic=generated_interface_boundary_intersection_marker"
          << " marker=" << intersection_domain.marker()
          << " interface_marker=" << result.interface_marker
          << " boundary_marker=" << boundary_marker
          << " field='" << request.level_set_field_name << "'"
          << " domain_id='" << request.domain_id << "'"
          << " source='" << domain_request.source.identifier() << "'"
          << " fragments=" << global_marker_fragments
          << " active_fragments=" << global_marker_active_fragments
          << " skipped_fragments=" << global_marker_skipped_fragments
          << " qpoints=" << global_marker_qpoints
          << " measure=" << global_marker_measure
          << " source_surface_fragments="
          << intersection_provenance.source_surface_fragment_count
          << " referenced_source_surface_fragments="
          << intersection_provenance
                 .referenced_source_surface_fragment_count
          << " orphan_contact_fragments="
          << intersection_provenance.orphan_contact_fragment_count
          << " stale_revision_count="
          << intersection_provenance.stale_revision_count
          << " max_level_set_residual="
          << intersection_provenance.max_level_set_residual
          << std::endl;
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
      mixCutContextHash(report.topology_key,
                        static_cast<std::uint64_t>(
                            std::max(0, intersection_domain.marker())));
      mixCutContextHash(report.topology_key,
                        static_cast<std::uint64_t>(
                            intersection_summary.active_fragment_count));
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
        auto negative_boundary =
            svmp::FE::interfaces::buildGeneratedActiveBoundaryDomain(
                std::move(negative_boundary_request),
                result.domain,
                intersection_domain,
                mesh_access,
                scalar_field);
        auto positive_boundary =
            svmp::FE::interfaces::buildGeneratedActiveBoundaryDomain(
                std::move(positive_boundary_request),
                result.domain,
                intersection_domain,
                mesh_access,
                scalar_field);
        const auto partition =
            svmp::FE::interfaces::validateGeneratedActiveBoundaryPartition(
                negative_boundary,
                positive_boundary,
                result.domain,
                intersection_domain,
                mesh_access);
        application::core::oopCout()
            << "[svMultiPhysics::Application] Generated sharp active boundary"
            << " diagnostic=generated_active_boundary_partition"
            << " interface_marker=" << result.interface_marker
            << " boundary_marker=" << boundary_marker
            << " negative_marker=" << negative_boundary.marker()
            << " positive_marker=" << positive_boundary.marker()
            << " boundary_faces=" << partition.boundary_face_count
            << " cut_boundary_faces=" << partition.cut_boundary_face_count
            << " contact_fragments="
            << partition.source_contact_fragment_count
            << " referenced_contact_fragments="
            << partition.referenced_contact_fragment_count
            << " orphan_source_references="
            << partition.orphan_source_reference_count
            << " stale_revision_count=" << partition.stale_revision_count
            << " reference_parent_measure="
            << partition.total_boundary_measure
            << " reference_negative_measure="
            << partition.negative_boundary_measure
            << " reference_positive_measure="
            << partition.positive_boundary_measure
            << " max_reference_partition_error="
            << partition.max_partition_error
            << std::endl;
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
          const auto evaluation = linear_corner
              ? snapshot_cell_evaluator->evaluateLinearCorner(
                    cell, parent_coordinate)
              : snapshot_cell_evaluator->evaluate(cell, parent_coordinate);
          return evaluation.value;
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
    svmp::FE::interfaces::FreeSurfaceGeometrySnapshotPolicy snapshot_policy;
    snapshot_policy.tolerance = domain_request.tolerance;
    snapshot_policy.minimum_retained_volume_fraction =
        svmp::FE::assembly::CutIntegrationContext::
            minGeneratedCutVolumeFraction();
    snapshot_policy.minimum_achieved_quadrature_order = 0;
    snapshot_policy.require_complete_exterior_boundary_partition = true;
    auto geometry_snapshot =
        svmp::FE::interfaces::buildFreeSurfaceGeometrySnapshot(
            result.domain,
            std::move(snapshot_contact_domains),
            std::move(snapshot_active_boundary_domains),
            mesh_access,
            snapshot_policy,
            std::move(snapshot_scalar_evaluator),
            request.domain_id,
            snapshotOwnershipCollective(comm));
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
    const auto geometry_cache_statistics =
        sim.free_surface_geometry_snapshot_cache->statistics();
    const auto& geometry_ledger = geometry_snapshot->ledger();
    application::core::oopCout()
        << "[svMultiPhysics::Application] Authoritative free-surface geometry"
        << " diagnostic=free_surface_geometry_snapshot"
        << " domain_id='" << request.domain_id << "'"
        << " interface_marker=" << result.interface_marker
        << " snapshot_revision="
        << geometry_snapshot->revision().snapshot_revision_key
        << " snapshot_resident_bytes="
        << geometry_snapshot->residentBytes()
        << " cache_live_snapshots="
        << geometry_cache_statistics.live_snapshot_count
        << " cache_live_resident_bytes="
        << geometry_cache_statistics.live_resident_bytes
        << " cache_peak_live_snapshots="
        << geometry_cache_statistics.peak_live_snapshot_count
        << " cache_peak_live_resident_bytes="
        << geometry_cache_statistics.peak_live_resident_bytes
        << " cache_hits=" << geometry_cache_statistics.hit_count
        << " cache_misses=" << geometry_cache_statistics.miss_count
        << " cache_expired_evictions="
        << geometry_cache_statistics.expired_eviction_count
        << " source_value_revision="
        << geometry_snapshot->revision().source_value_revision
        << " rules=" << geometry_ledger.rule_count
        << " retained_rules=" << geometry_ledger.retained_rule_count
        << " pruned_rules=" << geometry_ledger.pruned_rule_count
        << " qpoints=" << geometry_ledger.quadrature_point_count
        << " owned_rules=" << geometry_ledger.owned_rule_count
        << " global_owned_rules="
        << geometry_ledger.global_owned_rule_count
        << " invalid_global_identities="
        << geometry_ledger.invalid_global_identity_count
        << " duplicate_rule_identities="
        << geometry_ledger.duplicate_rule_identity_count
        << " contact_fragments=" << geometry_ledger.contact_fragment_count
        << " orphan_contact_fragments="
        << geometry_ledger.orphan_contact_fragment_count
        << " stale_revisions=" << geometry_ledger.stale_revision_count
        << " invalid_phase_points="
        << geometry_ledger.invalid_phase_point_count
        << " represented_phase_points="
        << geometry_ledger.represented_phase_point_count
        << " represented_phase_disagreements="
        << geometry_ledger.represented_phase_disagreement_count
        << " max_root_residual=" << geometry_ledger.maximum_root_residual
        << " max_normal_angle_error="
        << geometry_ledger.maximum_normal_angular_error
        << " max_represented_phase_disagreement="
        << geometry_ledger.maximum_represented_phase_disagreement
        << " max_constant_moment_error="
        << geometry_ledger.maximum_constant_moment_error
        << " certified_rules="
        << geometry_ledger.certified_rule_count
        << " parent_cell_moment_certificates="
        << geometry_ledger.parent_cell_moment_certificate_count
        << " centroid_moment_certificates="
        << geometry_ledger.centroid_moment_certificate_count
        << " piecewise_affine_moment_certificates="
        << geometry_ledger.piecewise_affine_moment_certificate_count
        << " backend_reference_moment_certificates="
        << geometry_ledger.backend_reference_moment_certificate_count
        << " stored_generated_moment_certificates="
        << geometry_ledger.stored_generated_moment_certificate_count
        << " validated_rule_polynomial_moments="
        << geometry_ledger.validated_rule_polynomial_moment_count
        << " validated_polynomial_moments="
        << geometry_ledger.validated_polynomial_moment_count
        << " max_polynomial_moment_error="
        << geometry_ledger.maximum_polynomial_moment_error
        << " max_polynomial_moment_scaled_error="
        << geometry_ledger.maximum_polynomial_moment_scaled_error
        << " max_volume_partition_error="
        << geometry_ledger.maximum_volume_partition_error
        << " max_boundary_partition_error="
        << geometry_ledger.maximum_boundary_partition_error
        << " negative_reference_volume="
        << geometry_ledger.retained_negative_reference_volume
        << " positive_reference_volume="
        << geometry_ledger.retained_positive_reference_volume
        << " negative_physical_volume="
        << geometry_ledger.retained_negative_physical_volume
        << " positive_physical_volume="
        << geometry_ledger.retained_positive_physical_volume
        << " resident_bytes=" << geometry_snapshot->residentBytes()
        << std::endl;
    context->addFreeSurfaceGeometrySnapshot(
        geometry_snapshot, retained_volume_sides);
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
        application::core::collectCutVolumeMeasures(
            mesh_access,
            context->generatedVolumeRulesForMarkerAndSide(
                result.interface_marker,
                active_volume_side));
    const auto inactive_measure_summary =
        application::core::collectCutVolumeMeasures(
            mesh_access,
            context->generatedVolumeRulesForMarkerAndSide(
                result.interface_marker,
                inactive_volume_side));
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
        << " retained_volume_sides="
        << retainedVolumeSidesName(request.volume_retention)
        << " interface_contract=one_sided_embedded"
        << " generated_interface_boundary_intersection_fragments="
        << global_boundary_intersection_fragments
        << " generated_interface_boundary_intersection_active_fragments="
        << global_active_boundary_intersection_fragments
        << " generated_interface_boundary_intersection_skipped_fragments="
        << global_skipped_boundary_intersection_fragments
        << " generated_interface_boundary_intersection_qpoints="
        << global_boundary_intersection_qpoints
        << " generated_interface_boundary_intersection_measure="
        << global_boundary_intersection_measure
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
        << " generated_cell_work_scope=summed_rank_local_including_ghost_cells"
        << " generated_cell_cache_hits="
        << global_generated_cell_cache_hits
        << " generated_cell_cache_misses="
        << global_generated_cell_cache_misses
        << " generated_cell_cache_unchanged_dof_hits="
        << global_generated_cell_cache_unchanged_dof_hits
        << " generated_cell_refresh_candidates="
        << global_generated_cell_refresh_candidates
        << " generated_cell_directly_affected="
        << global_generated_cell_directly_affected
        << " generated_cell_affected_neighborhood="
        << global_generated_cell_affected_neighborhood
        << " generated_cell_affected_neighborhood_layers="
        << result.affected_cell_neighborhood_layers
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
        << " affected_cell_neighborhood_layers="
        << options.affected_cell_neighborhood_layers
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
        << " snapshot_active_side_physical_volume="
        << global_snapshot_active_physical_volume
        << " active_constant_one_measure_error="
        << active_constant_one_error
        << " constant_one_measure_tolerance="
        << constant_one_tolerance
        << " active_side_physical_rule_count="
        << global_active_physical_volume_rules
        << " active_side_skipped_physical_rule_count="
        << global_active_skipped_physical_volume_rules
        << " active_side_available_cut_volume_rule_count="
        << global_active_available_cut_volume_rules
        << " active_side_retained_cut_volume_rule_count="
        << global_active_retained_cut_volume_rules
        << " active_side_cut_volume_consumer_count="
        << active_cut_volume_consumer_count
        << " inactive_side_physical_volume="
        << global_inactive_physical_volume
        << " snapshot_inactive_side_physical_volume="
        << global_snapshot_inactive_physical_volume
        << " inactive_constant_one_measure_error="
        << inactive_constant_one_error
        << " inactive_side_physical_rule_count="
        << global_inactive_physical_volume_rules
        << " inactive_side_skipped_physical_rule_count="
        << global_inactive_skipped_physical_volume_rules
        << " inactive_side_available_cut_volume_rule_count="
        << global_inactive_available_cut_volume_rules
        << " inactive_side_retained_cut_volume_rule_count="
        << global_inactive_retained_cut_volume_rules
        << " inactive_side_cut_volume_consumer_count="
        << inactive_cut_volume_consumer_count
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
        << " generated_volume_prune_min_fraction="
        << svmp::FE::assembly::CutIntegrationContext::
               minGeneratedCutVolumeFraction()
        << " generated_pruned_volume_scope=summed_rank_local_including_ghost_cells"
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
        << " cut_adjacent_scope=summed_rank_local_including_ghost_cells"
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
  std::uint64_t mesh_numbering_revision{0};
  std::uint64_t mesh_field_layout_revision{0};
  std::uint64_t mesh_label_revision{0};
  std::uint64_t mesh_active_configuration_epoch{0};
  std::uint64_t mesh_coordinate_configuration_key{0};
  std::uint64_t system_space_revision{0};
  std::uint64_t system_dof_layout_revision{0};
  std::uint64_t system_block_layout_revision{0};
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
           mesh_numbering_revision == other.mesh_numbering_revision &&
           mesh_field_layout_revision == other.mesh_field_layout_revision &&
           mesh_label_revision == other.mesh_label_revision &&
           mesh_active_configuration_epoch ==
               other.mesh_active_configuration_epoch &&
           mesh_coordinate_configuration_key ==
               other.mesh_coordinate_configuration_key &&
           system_space_revision == other.system_space_revision &&
           system_dof_layout_revision == other.system_dof_layout_revision &&
           system_block_layout_revision == other.system_block_layout_revision &&
           solution_signature_kind == other.solution_signature_kind &&
           solution_hash == other.solution_hash &&
           solution_size == other.solution_size;
  }
};

struct ActiveCutContextRefreshCache {
  std::optional<ActiveCutContextRefreshSignature> last_signature{};
  std::optional<ActiveCutContextRefreshSignature> last_vector_signature{};
  std::map<int, std::uint64_t> evaluated_state_source_revisions{};
};

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

void populateActiveCutContextRefreshDependencies(
    ActiveCutContextRefreshSignature& signature,
    const svmp::FE::systems::FESystem& system)
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
}

void logActiveCutContextRefreshSkipped(
    const ActiveCutContextRefreshSignature& signature,
    const char* provenance,
    const char* solution_source,
    const char* skip_reason)
{
  application::core::oopCout()
      << "[svMultiPhysics::Application] Cut-context refresh skipped"
      << " diagnostic=cut_context_refresh_skip"
      << " provenance=" << (provenance != nullptr ? provenance : "unknown")
      << " solution_source="
      << (solution_source != nullptr ? solution_source : "unknown")
      << " skip_reason=" << (skip_reason != nullptr ? skip_reason : "unknown")
      << " active_cut_request_policy_key=" << signature.request_policy_key
      << " signature_kind="
      << activeCutRefreshSignatureKindName(signature.solution_signature_kind)
      << " solution_hash=" << signature.solution_hash
      << " solution_size=" << signature.solution_size
      << " mesh_geometry_revision=" << signature.mesh_geometry_revision
      << " mesh_topology_revision=" << signature.mesh_topology_revision
      << " ownership_revision=" << signature.mesh_ownership_revision
      << " mesh_numbering_revision=" << signature.mesh_numbering_revision
      << " mesh_field_layout_revision="
      << signature.mesh_field_layout_revision
      << " mesh_label_revision=" << signature.mesh_label_revision
      << " mesh_active_configuration_epoch="
      << signature.mesh_active_configuration_epoch
      << " mesh_coordinate_configuration_key="
      << signature.mesh_coordinate_configuration_key
      << " system_space_revision=" << signature.system_space_revision
      << " system_dof_layout_revision="
      << signature.system_dof_layout_revision
      << " system_block_layout_revision="
      << signature.system_block_layout_revision << std::endl;
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
  ActiveCutContextRefreshSignature signature;
  signature.request_policy_key = activeCutVolumeRequestPolicyKey(requests);
  populateActiveCutContextRefreshDependencies(signature, *sim.fe_system);
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
        populateActiveCutContextRefreshDependencies(signature, *sim.fe_system);
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
  populateActiveCutContextRefreshDependencies(signature, *sim.fe_system);
  signature.solution_signature_kind =
      ActiveCutContextRefreshSignature::SolutionSignatureKind::
          VectorValueRevision;
  signature.solution_size = solution_size;
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
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const auto signature =
      activeCutContextRefreshSignature(sim, requests, fe_solution);
  const bool local_can_skip =
      !parseBoolEnv("SVMP_DISABLE_ACTIVE_CUT_REFRESH_CACHE", false) &&
      signature.has_value() &&
      cache.last_signature.has_value() &&
      *signature == *cache.last_signature &&
      sim.fe_system->cutIntegrationContext() != nullptr;
  if (!globalAnyBool(!local_can_skip, comm)) {
    skipped_report.evaluated_state_source_revisions =
        cache.evaluated_state_source_revisions;
    logActiveCutContextRefreshSkipped(
        *signature,
        provenance,
        solution_source,
        "level_set_field_hash_unchanged");
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
    cache.evaluated_state_source_revisions =
        report.evaluated_state_source_revisions;
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
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const auto vector_signature =
      activeCutContextRefreshSignature(sim, requests, solution);
  const bool disable_refresh_cache =
      parseBoolEnv("SVMP_DISABLE_ACTIVE_CUT_REFRESH_CACHE", false);
  const bool local_vector_can_skip =
      !disable_refresh_cache &&
      vector_signature.has_value() &&
      cache.last_vector_signature.has_value() &&
      *vector_signature == *cache.last_vector_signature &&
      sim.fe_system->cutIntegrationContext() != nullptr;
  if (!globalAnyBool(!local_vector_can_skip, comm)) {
    skipped_report.evaluated_state_source_revisions =
        cache.evaluated_state_source_revisions;
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
    return skipped_report;
  }

  const auto fe_solution = gatherFeOrderedSolution(solution);
  const auto signature = activeCutContextRefreshSignature(
      sim,
      requests,
      std::span<const svmp::FE::Real>(fe_solution.data(), fe_solution.size()));
  const bool local_fe_can_skip =
      !disable_refresh_cache &&
      signature.has_value() &&
      cache.last_signature.has_value() &&
      *signature == *cache.last_signature &&
      sim.fe_system->cutIntegrationContext() != nullptr;
  if (!globalAnyBool(!local_fe_can_skip, comm)) {
    skipped_report.evaluated_state_source_revisions =
        cache.evaluated_state_source_revisions;
    if (vector_signature.has_value()) {
      cache.last_vector_signature = *vector_signature;
    }
    logActiveCutContextRefreshSkipped(
        *signature,
        provenance,
        solution_source,
        "level_set_field_hash_unchanged");
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
    cache.evaluated_state_source_revisions =
        report.evaluated_state_source_revisions;
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

class LevelSetMaintenanceGeometryTransaction {
public:
  LevelSetMaintenanceGeometryTransaction(
      application::core::SimulationComponents& sim,
      svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
      ActiveCutContextRefreshCache& refresh_cache,
      const std::vector<ActiveCutVolumeRequest>& requests)
      : sim_(sim),
        lifecycle_(lifecycle),
        refresh_cache_(refresh_cache),
        refresh_cache_backup_(refresh_cache),
        mesh_field_checkpoint_(
            captureActiveLevelSetMeshFields(sim, requests))
  {
    if (!sim_.fe_system) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set maintenance geometry transaction requires an FE system.");
    }
    lifecycle_.beginTransaction();
    lifecycle_transaction_active_ = true;
    try {
      sim_.fe_system->beginCutIntegrationContextTransaction();
      system_transaction_active_ = true;
      active_ = true;
    } catch (...) {
      lifecycle_.rollbackTransaction();
      lifecycle_transaction_active_ = false;
      throw;
    }
  }

  LevelSetMaintenanceGeometryTransaction(
      const LevelSetMaintenanceGeometryTransaction&) = delete;
  LevelSetMaintenanceGeometryTransaction& operator=(
      const LevelSetMaintenanceGeometryTransaction&) = delete;

  ~LevelSetMaintenanceGeometryTransaction()
  {
    if (!active_) {
      return;
    }
    try {
      rollback();
    } catch (...) {
    }
  }

  [[nodiscard]] ActiveCutContextRefreshReport refresh(
      const Parameters& params,
      std::span<const svmp::FE::Real> candidate)
  {
    if (!active_) {
      throw std::logic_error(
          "level-set maintenance geometry transaction is not active");
    }
    return refreshActiveCutIntegrationContextFromSolutionCached(
        sim_,
        params,
        candidate,
        lifecycle_,
        refresh_cache_,
        "accepted_step_maintenance_candidate",
        "staged_fe_solution");
  }

  void commit()
  {
    if (!active_ || !system_transaction_active_ ||
        !lifecycle_transaction_active_) {
      throw std::logic_error(
          "level-set maintenance geometry transaction is not active");
    }
    sim_.fe_system->commitCutIntegrationContextTransaction();
    system_transaction_active_ = false;
    lifecycle_.commitTransaction();
    lifecycle_transaction_active_ = false;
    active_ = false;
  }

  void rollback()
  {
    if (!active_) {
      return;
    }
    std::exception_ptr first_failure;
    const auto attempt = [&](auto&& action) {
      try {
        action();
      } catch (...) {
        if (!first_failure) {
          first_failure = std::current_exception();
        }
      }
    };
    attempt([&] {
      restoreActiveLevelSetMeshFields(sim_, mesh_field_checkpoint_);
    });
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
    attempt([&] { refresh_cache_ = refresh_cache_backup_; });
    active_ = false;
    if (first_failure) {
      std::rethrow_exception(first_failure);
    }
  }

private:
  application::core::SimulationComponents& sim_;
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle_;
  ActiveCutContextRefreshCache& refresh_cache_;
  ActiveCutContextRefreshCache refresh_cache_backup_{};
  ActiveLevelSetMeshFieldCheckpoint mesh_field_checkpoint_{};
  bool lifecycle_transaction_active_{false};
  bool system_transaction_active_{false};
  bool active_{false};
};

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
    const LevelSetMaintenanceRequest& request)
{
  if (!request.volume_cut_request.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport requires one matching authoritative active cut-volume request for level-set field '" +
        request.level_set_field_name + "'.");
  }
  const auto requested_side = conservativePhaseSide(
      request.conservative_phase);
  const auto active_side = cutIntegrationSide(
      request.volume_cut_request->active_side);
  if (requested_side != active_side) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase liquid side disagrees with the active cut-volume side for level-set field '" +
        request.level_set_field_name + "'.");
  }
  const auto marker = generatedCutContextMarkerForMaintenance(
      system, request);
  if (!marker.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport could not resolve an authoritative generated interface marker for level-set field '" +
        request.level_set_field_name + "'.");
  }
}

svmp::FE::level_set::LevelSetP1PhaseTransportGraph&
requireCurrentConservativePhaseGraph(
    svmp::FE::systems::FESystem& system,
    LevelSetMaintenanceRequest& request)
{
  const auto phase_field = system.findFieldByName(
      request.conservative_phase.liquid_indicator.field_name);
  if (phase_field == svmp::FE::INVALID_FIELD_ID) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport could not find field '" +
        request.conservative_phase.liquid_indicator.field_name + "'.");
  }
  const auto& mesh = system.meshAccess();
  const auto& dofs = system.fieldDofHandler(phase_field);
  const auto graph_is_current = [&] {
    if (!request.conservative_phase_graph.has_value()) {
      return false;
    }
    const auto& graph = *request.conservative_phase_graph;
    return graph.success &&
           graph.geometry_revision == mesh.geometryRevision() &&
           graph.topology_revision == mesh.topologyRevision() &&
           graph.ownership_revision == mesh.ownershipRevision() &&
           graph.numbering_revision == mesh.numberingRevision() &&
           graph.dof_layout_revision == dofs.dofLayoutRevision() &&
           graph.nodes == static_cast<std::size_t>(dofs.getNumDofs());
  };
  if (!graph_is_current()) {
    svmp::FE::level_set::LevelSetP1PhaseGraphOptions graph_options;
    graph_options.invariant_tolerance =
        request.conservative_phase.invariant_tolerance;
    auto graph =
        svmp::FE::level_set::buildLevelSetP1PhaseTransportGraph(
            system, phase_field, graph_options);
    if (!graph.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase graph assembly failed for field '" +
          request.conservative_phase.liquid_indicator.field_name +
          "': " + graph.diagnostic);
    }
    request.conservative_phase_graph = std::move(graph);
  }
  return *request.conservative_phase_graph;
}

svmp::FE::level_set::LevelSetP1PhaseProjectionOptions
conservativePhaseProjectionOptions(
    const svmp::FE::systems::FESystem& system,
    const LevelSetMaintenanceRequest& request)
{
  const auto marker = generatedCutContextMarkerForMaintenance(
      system, request);
  if (!marker.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase projection could not resolve its generated interface marker.");
  }
  svmp::FE::level_set::LevelSetP1PhaseProjectionOptions options;
  options.interface_marker = *marker;
  options.liquid_side = conservativePhaseSide(
      request.conservative_phase);
  options.invariant_tolerance =
      request.conservative_phase.invariant_tolerance;
  return options;
}

svmp::FE::level_set::LevelSetP1PhaseProjectionResult
projectCurrentConservativePhaseGeometry(
    svmp::FE::systems::FESystem& system,
    LevelSetMaintenanceRequest& request)
{
  requireConservativePhaseGeometryBinding(system, request);
  auto& graph = requireCurrentConservativePhaseGraph(system, request);
  const auto phase_field = system.findFieldByName(
      request.conservative_phase.liquid_indicator.field_name);
  const auto projection_options = conservativePhaseProjectionOptions(
      system, request);
  return svmp::FE::level_set::
      projectLevelSetP1PhaseIndicatorFromCutContext(
          system, phase_field, graph, projection_options);
}

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
  auto values = gatherFeOrderedSolution(rate);
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

void initializeConservativePhaseStates(
    application::core::SimulationComponents& sim,
    std::vector<LevelSetMaintenanceRequest>& requests)
{
  if (!sim.fe_system || !sim.time_history) {
    return;
  }
  auto& system = *sim.fe_system;
  auto& history = *sim.time_history;
  for (auto& request : requests) {
    if (!request.conservative_phase.enabled ||
        request.conservative_phase_initialized) {
      continue;
    }
    const auto projection = projectCurrentConservativePhaseGeometry(
        system, request);
    if (!projection.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Initial conservative phase projection failed for field '" +
          request.conservative_phase.liquid_indicator.field_name +
          "': " + projection.diagnostic);
    }
    const auto phase_field = system.findFieldByName(
        request.conservative_phase.liquid_indicator.field_name);
    auto current = gatherFeOrderedSolution(history.u());
    assignConservativePhaseSlice(
        system, phase_field, projection.liquid_indicator, current);
    std::vector<std::vector<svmp::FE::Real>> initialized_history;
    initialized_history.reserve(
        static_cast<std::size_t>(history.historyDepth()));
    for (int k = 1; k <= history.historyDepth(); ++k) {
      auto previous = gatherFeOrderedSolution(history.uPrevK(k));
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
    request.conservative_phase_initialized = true;
    application::core::oopCout()
        << "[svMultiPhysics::Application] Conservative phase initialized"
        << " field='"
        << request.conservative_phase.liquid_indicator.field_name << "'"
        << " level_set_field='" << request.level_set_field_name << "'"
        << " liquid_side="
        << (request.conservative_phase.liquid_side ==
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

void allReduceConservativePhaseBuffer(
    std::vector<svmp::FE::Real>& values,
    const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
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
    std::vector<svmp::FE::Real> reduced(values.size(),
                                         svmp::FE::Real{0.0});
    std::size_t offset = 0u;
    while (offset < values.size()) {
      const auto remaining = values.size() - offset;
      const int count = static_cast<int>(std::min<std::size_t>(
          remaining,
          static_cast<std::size_t>(std::numeric_limits<int>::max())));
      MPI_Allreduce(values.data() + offset, reduced.data() + offset,
                    count, datatype, MPI_SUM, comm.native());
      offset += static_cast<std::size_t>(count);
    }
    values = std::move(reduced);
  }
#else
  (void)comm;
#endif
}

std::vector<std::array<svmp::FE::Real, 3>>
sampleConservativePhaseVelocity(
    application::core::SimulationComponents& sim,
    const LevelSetMaintenanceRequest& request,
    const svmp::FE::level_set::LevelSetP1PhaseTransportGraph& graph,
    const svmp::FE::systems::SystemStateView& state)
{
  std::vector<std::array<svmp::FE::Real, 3>> velocity(graph.nodes);
  if (request.velocity.source ==
      svmp::FE::level_set::LevelSetVelocitySource::ConstantVector) {
    std::fill(velocity.begin(), velocity.end(),
              request.velocity.constant_value);
    return velocity;
  }
  if (!sim.fe_system || !sim.primary_mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase velocity sampling requires an FE system and primary mesh.");
  }
  auto& system = *sim.fe_system;
  const auto phase_field = system.findFieldByName(
      request.conservative_phase.liquid_indicator.field_name);
  const auto velocity_field = system.findFieldByName(
      request.velocity.field_name);
  if (phase_field == svmp::FE::INVALID_FIELD_ID ||
      velocity_field == svmp::FE::INVALID_FIELD_ID) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase velocity sampling could not resolve its phase or velocity field.");
  }
  const auto& velocity_record = system.fieldRecord(velocity_field);
  if (velocity_record.components != graph.dimension) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase velocity dimension does not match the phase graph.");
  }
  const auto* entity_map =
      system.fieldDofHandler(phase_field).getEntityDofMap();
  if (entity_map == nullptr) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase velocity sampling requires a vertex-nodal phase field.");
  }
  const auto vertex_velocity = evaluateVertexField(
      system,
      *sim.primary_mesh,
      velocity_field,
      state,
      static_cast<std::size_t>(graph.dimension),
      "sampling conservative phase velocity");
  std::vector<svmp::FE::Real> accumulated(graph.nodes * 4u,
                                           svmp::FE::Real{0.0});
  for (std::size_t vertex = 0;
       vertex < sim.primary_mesh->n_vertices(); ++vertex) {
    const auto dofs = entity_map->getVertexDofs(
        static_cast<svmp::FE::GlobalIndex>(vertex));
    if (dofs.size() != 1u || dofs.front() < 0 ||
        static_cast<std::size_t>(dofs.front()) >= graph.nodes) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase field does not have exactly one in-range DOF per mesh vertex.");
    }
    const auto node = static_cast<std::size_t>(dofs.front());
    for (int d = 0; d < graph.dimension; ++d) {
      accumulated[4u * node + static_cast<std::size_t>(d)] +=
          static_cast<svmp::FE::Real>(
              vertex_velocity[
                  vertex * static_cast<std::size_t>(graph.dimension) +
                  static_cast<std::size_t>(d)]);
    }
    accumulated[4u * node + 3u] += svmp::FE::Real{1.0};
  }
  allReduceConservativePhaseBuffer(
      accumulated, activeFESystemCommunicator(system));
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
  return velocity;
}

std::vector<std::array<svmp::FE::Real, 3>>
conservativePhaseNodeCoordinates(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId phase_field,
    const svmp::FE::level_set::LevelSetP1PhaseTransportGraph& graph)
{
  const auto* entity_map =
      system.fieldDofHandler(phase_field).getEntityDofMap();
  if (entity_map == nullptr) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase fixed regions require a vertex-nodal phase field.");
  }
  const auto& mesh = system.meshAccess();
  std::vector<svmp::FE::Real> accumulated(
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
  allReduceConservativePhaseBuffer(
      accumulated, activeFESystemCommunicator(system));
  std::vector<std::array<svmp::FE::Real, 3>> coordinates(graph.nodes);
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

std::vector<std::uint8_t> conservativePhaseContactProtectedNodes(
    const svmp::FE::systems::FESystem& system,
    const LevelSetMaintenanceRequest& request,
    const svmp::FE::level_set::LevelSetP1PhaseTransportGraph& graph,
    std::span<const svmp::FE::level_set::LevelSetWallContactConstraint>
        constraints)
{
  std::vector<std::uint8_t> protected_nodes(graph.nodes, 0u);
  const auto comm = activeFESystemCommunicator(system);
  if (globalSumSize(constraints.size(), comm) == 0u) {
    return protected_nodes;
  }
  const auto phase_field = system.findFieldByName(
      request.conservative_phase.liquid_indicator.field_name);
  if (phase_field == svmp::FE::INVALID_FIELD_ID) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase contact protection could not resolve the phase field.");
  }
  const auto& phase_dofs = system.fieldDofHandler(phase_field);
  const auto& mesh = system.meshAccess();
  const bool local_identity_available =
      mesh.parallelSize() == 1 || mesh.globalEntityIdsAvailable();
  if (globalMinDouble(local_identity_available ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase contact protection requires globally unique parent-cell identities in a distributed run.");
  }

  std::set<svmp::FE::GlobalIndex> local_parent_ids;
  bool local_constraints_valid = true;
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
  std::vector<svmp::FE::Real> global_mask(graph.nodes,
                                           svmp::FE::Real{0.0});
  std::set<svmp::FE::GlobalIndex> matched_parent_ids;
  try {
    mesh.forEachOwnedCell([&](svmp::FE::GlobalIndex cell) {
      const auto global_cell = mesh.globalEntityIdsAvailable()
          ? mesh.getCellGlobalId(cell)
          : cell;
      if (!local_parent_ids.contains(global_cell)) {
        return;
      }
      const auto cell_dofs = phase_dofs.getCellDofs(cell);
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
  } catch (const std::exception&) {
    local_constraints_valid = false;
  }
  if (globalMinDouble(local_constraints_valid ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase contact protection could not map every accepted contact parent to its phase DOFs.");
  }
  allReduceConservativePhaseBuffer(global_mask, comm);
  for (std::size_t node = 0u; node < graph.nodes; ++node) {
    if (!std::isfinite(global_mask[node]) ||
        global_mask[node] < svmp::FE::Real{0.0}) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase contact protection produced an invalid distributed node mask.");
    }
    protected_nodes[node] = global_mask[node] > svmp::FE::Real{0.0}
        ? 1u
        : 0u;
  }
  return protected_nodes;
}

struct ConservativePhaseContactStageCandidate {
  std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
      stages{};
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
      constraints{};
  std::vector<svmp::FE::Real> stage_solution{};
};

using ConservativePhaseContactStageBuilder = std::function<
    ConservativePhaseContactStageCandidate(
        std::span<const svmp::FE::Real>,
        LevelSetMaintenanceGeometryTransaction*)>;

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
    LevelSetMaintenanceGeometryTransaction* active_transaction)
{
  if (endpoint_solution.size() != previous_solution.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Contact-stage reconstruction requires equal endpoint and previous solution layouts.");
  }

  ConservativePhaseContactStageCandidate contact_stage;
  contact_stage.stage_solution.assign(
      endpoint_solution.size(), svmp::FE::Real{0.0});
  for (std::size_t i = 0; i < contact_stage.stage_solution.size(); ++i) {
    contact_stage.stage_solution[i] =
        (svmp::FE::Real{1.0} - stage_alpha_f) * previous_solution[i] +
        stage_alpha_f * endpoint_solution[i];
  }

  std::unique_ptr<LevelSetMaintenanceGeometryTransaction>
      owned_transaction;
  if (active_transaction == nullptr) {
    owned_transaction =
        std::make_unique<LevelSetMaintenanceGeometryTransaction>(
            sim, lifecycle, refresh_cache, active_cut_requests);
    active_transaction = owned_transaction.get();
  }

  try {
    (void)active_transaction->refresh(
        params,
        std::span<const svmp::FE::Real>(
            contact_stage.stage_solution.data(),
            contact_stage.stage_solution.size()));
    contact_stage.stages = evaluateAcceptedFreeSurfaceContactStages(
        sim,
        stage_time,
        stage_alpha_f,
        previous_state_revision,
        endpoint_state_revision,
        contact_stage.stage_solution);
    contact_stage.constraints =
        captureAcceptedContactStageWallConstraints(
            sim, contact_stage.stages);
  } catch (...) {
    const auto stage_failure = std::current_exception();
    (void)active_transaction->refresh(params, endpoint_solution);
    std::rethrow_exception(stage_failure);
  }

  (void)active_transaction->refresh(params, endpoint_solution);
  if (owned_transaction) {
    owned_transaction->commit();
  }
  return contact_stage;
}

struct ConservativePhaseMomentMismatch {
  svmp::FE::Real maximum_nodal_residual{0.0};
  svmp::FE::Real residual_norm{0.0};
  svmp::FE::Real total_residual{0.0};
};

ConservativePhaseMomentMismatch conservativePhaseMomentMismatch(
    std::span<const svmp::FE::Real> current,
    std::span<const svmp::FE::Real> target)
{
  if (current.size() != target.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase moment vectors have incompatible layouts.");
  }
  ConservativePhaseMomentMismatch mismatch;
  svmp::FE::Real squared_norm{0.0};
  for (std::size_t node = 0u; node < current.size(); ++node) {
    const auto residual = target[node] - current[node];
    if (!std::isfinite(residual)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase geometry reconciliation found a non-finite local moment residual.");
    }
    mismatch.maximum_nodal_residual = std::max(
        mismatch.maximum_nodal_residual, std::abs(residual));
    squared_norm += residual * residual;
    mismatch.total_residual += residual;
  }
  mismatch.residual_norm = std::sqrt(squared_norm);
  return mismatch;
}

struct ConservativePhaseGeometryReconciliationResult {
  bool success{false};
  bool target_reached{false};
  bool limited_by_displacement{false};
  bool limited_by_topology{false};
  int iterations{0};
  int line_search_evaluations{0};
  int geometry_refresh_requests{0};
  int geometry_rebuilds{0};
  int rejected_geometry_trials{0};
  std::size_t contact_protected_nodes{0u};
  svmp::FE::Real allowed_interface_displacement{0.0};
  svmp::FE::Real accumulated_interface_displacement_bound{0.0};
  svmp::FE::Real initial_residual_norm{0.0};
  svmp::FE::Real final_residual_norm{0.0};
  svmp::FE::Real maximum_final_nodal_residual{0.0};
  svmp::FE::Real final_total_residual{0.0};
  svmp::FE::Real maximum_removed_contact_increment{0.0};
  std::string last_rejected_trial_diagnostic{};
  std::string diagnostic{};
};

struct ConservativePhaseMaintenanceStageLedger {
  bool reinitialization_due{false};
  bool reinitialization_applied{false};
  svmp::FE::Real raw_post_transport_phase_measure{0.0};
  svmp::FE::Real post_limit_phase_measure{0.0};
  svmp::FE::Real raw_post_transport_geometry_measure{0.0};
  svmp::FE::Real post_reinitialization_geometry_measure{0.0};
  svmp::FE::Real post_correction_geometry_measure{0.0};
  ConservativePhaseMomentMismatch post_reinitialization_mismatch{};
  ConservativePhaseMomentMismatch post_correction_mismatch{};
  svmp::FE::level_set::LevelSetP1PhaseTransportStageResult
      transport_stage{};
  svmp::FE::level_set::LevelSetPhaseRegionLedgerResult region_ledger{};
  svmp::FE::level_set::LevelSetSignedDistanceRepairResult
      reinitialization{};
  ConservativePhaseGeometryReconciliationResult reconciliation{};
};

struct ConservativePhaseCandidateResult {
  bool accept_step{true};
  bool changed{false};
  std::vector<svmp::FE::Real> original_solution{};
  ConservativePhaseContactStageCandidate contact_stage{};
  std::vector<ConservativePhaseMaintenanceStageLedger>
      maintenance_ledgers{};
  std::unique_ptr<LevelSetMaintenanceGeometryTransaction>
      geometry_transaction{};
};

template <typename Value>
void mixConservativePhaseArtifactFingerprint(
    std::uint64_t& fingerprint,
    const Value& value) noexcept
{
  static_assert(std::is_trivially_copyable_v<Value>);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  for (std::size_t index = 0u; index < sizeof(Value); ++index) {
    fingerprint ^= static_cast<std::uint64_t>(bytes[index]);
    fingerprint *= 1099511628211ull;
  }
}

void mixConservativePhaseArtifactFingerprint(
    std::uint64_t& fingerprint,
    std::string_view value) noexcept
{
  mixConservativePhaseArtifactFingerprint(fingerprint, value.size());
  for (const char character : value) {
    fingerprint ^=
        static_cast<std::uint64_t>(static_cast<unsigned char>(character));
    fingerprint *= 1099511628211ull;
  }
}

void mixConservativePhaseArtifactFingerprint(
    std::uint64_t& fingerprint,
    const std::string& value) noexcept
{
  mixConservativePhaseArtifactFingerprint(
      fingerprint, std::string_view(value));
}

std::uint64_t conservativePhaseArtifactFingerprint(
    const LevelSetMaintenanceRequest& request,
    const ConservativePhaseMaintenanceStageLedger& ledger)
{
  std::uint64_t fingerprint = 14695981039346656037ull;
  mixConservativePhaseArtifactFingerprint(
      fingerprint, request.level_set_field_name);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      request.conservative_phase.liquid_indicator.field_name);
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      request.volume_cut_request.has_value()
          ? std::string_view(request.volume_cut_request->domain_id)
          : std::string_view{});
  mixConservativePhaseArtifactFingerprint(
      fingerprint,
      request.conservative_phase
          .classify_nonprimary_components_as_satellites);
  mixConservativePhaseArtifactFingerprint(
      fingerprint, request.conservative_phase.fixed_flux_regions.size());
  for (const auto& box : request.conservative_phase.fixed_flux_regions) {
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
      fingerprint, ledger.region_ledger.regions.size());
  for (const auto& region : ledger.region_ledger.regions) {
    mixConservativePhaseArtifactFingerprint(fingerprint, region.name);
    mixConservativePhaseArtifactFingerprint(fingerprint, region.kind);
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.member_nodes.size());
    for (const auto node : region.member_nodes) {
      mixConservativePhaseArtifactFingerprint(fingerprint, node);
    }
    mixConservativePhaseArtifactFingerprint(
        fingerprint, region.crossing_edges.size());
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
  }
  return fingerprint;
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
  const bool locally_any_artifact_configured =
      std::any_of(
          requests.begin(), requests.end(), [](const auto& request) {
            return request.conservative_phase.enabled &&
                   request.conservative_phase.write_flux_artifacts;
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

  bool any_artifact_due = false;
  bool local_preflight_failure = false;
  std::vector<bool> artifact_due(requests.size(), false);
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const auto& request = requests[index];
    const int cadence =
        request.conservative_phase.flux_artifact_cadence_steps;
    const bool locally_configured =
        request.conservative_phase.enabled &&
        request.conservative_phase.write_flux_artifacts;
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
      const bool local_ledger_ready =
          candidate.maintenance_ledgers.size() == requests.size() &&
          request.conservative_phase_graph.has_value() &&
          request.volume_cut_request.has_value() &&
          candidate.maintenance_ledgers.size() > index &&
          candidate.maintenance_ledgers[index].transport_stage.success &&
          candidate.maintenance_ledgers[index].region_ledger.success;
      local_preflight_failure =
          !local_ledger_ready || local_preflight_failure;
      const auto fingerprint = local_ledger_ready
          ? conservativePhaseArtifactFingerprint(
                request, candidate.maintenance_ledgers[index])
          : 0u;
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

  std::filesystem::path output_directory = ".";
  if (params.general_simulation_parameters.save_results_in_folder.defined() &&
      !params.general_simulation_parameters.save_results_in_folder
           .value()
           .empty()) {
    output_directory =
        params.general_simulation_parameters.save_results_in_folder.value();
  }
  output_directory /= "conservative_phase_flux";

  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const auto& request = requests[index];
    if (!artifact_due[index]) {
      continue;
    }
    const auto& ledger = candidate.maintenance_ledgers[index];
    const auto& graph = *request.conservative_phase_graph;
    const auto& repair = ledger.reinitialization;
    const auto& reconciliation = ledger.reconciliation;
    svmp::FE::level_set::LevelSetConservativePhaseArtifactResult artifact;
    if (comm.rank() == 0) {
      svmp::FE::level_set::LevelSetConservativePhaseArtifactContext context;
      context.phase_field_name =
          request.conservative_phase.liquid_indicator.field_name;
      context.level_set_field_name = request.level_set_field_name;
      context.geometry_domain_id = request.volume_cut_request->domain_id;
      context.accepted_step = accepted_step;
      context.accepted_time = accepted_time;
      context.time_step = time_step;
      context.state_revision = state_revision;
      context.graph_geometry_revision = graph.geometry_revision;
      context.graph_topology_revision = graph.topology_revision;
      context.graph_ownership_revision = graph.ownership_revision;
      context.graph_numbering_revision = graph.numbering_revision;
      context.graph_dof_layout_revision = graph.dof_layout_revision;
      context.geometry_validated_before_commit = true;
      context.reinitialization_due = ledger.reinitialization_due;
      context.reinitialization_applied = ledger.reinitialization_applied;
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
          ledger.post_limit_phase_measure;
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
      if (!artifact.success) {
        application::core::oopCout()
            << "[svMultiPhysics::Application] Conservative phase artifact"
            << " diagnostic=conservative_phase_flux_artifact"
            << " field='"
            << request.conservative_phase.liquid_indicator.field_name
            << "' step=" << accepted_step
            << " outcome=failed"
            << " reason='" << artifact.diagnostic << "'" << std::endl;
      }
    }
    const bool publication_failed = globalAnyBool(
        comm.rank() == 0 && !artifact.success, comm);
    if (publication_failed) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase artifact publication failed on the output rank.");
    }
    if (comm.rank() == 0) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] Conservative phase artifact"
          << " diagnostic=conservative_phase_flux_artifact"
          << " field='"
          << request.conservative_phase.liquid_indicator.field_name << "'"
          << " step=" << accepted_step
          << " outcome=written"
          << " path='" << artifact.path.string() << "'"
          << " bytes=" << artifact.bytes
          << " nodes=" << artifact.nodes
          << " edges=" << artifact.edges
          << " resolved_components=" << artifact.resolved_components
          << " tracked_regions=" << artifact.tracked_regions
          << " subthreshold_component_present="
          << (artifact.subthreshold_component_present ? "true" : "false")
          << std::endl;
    }
  }
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
  ConservativePhaseGeometryReconciliationResult result;
  if (!sim.fe_system) {
    result.diagnostic =
        "Conservative phase geometry reconciliation requires an FE system";
    return result;
  }
  auto& system = *sim.fe_system;
  auto& graph = requireCurrentConservativePhaseGraph(system, request);
  if (target_liquid_phase_mass.size() != graph.nodes) {
    result.diagnostic =
        "Conservative phase geometry reconciliation target does not match the phase graph";
    return result;
  }
  if (!contact_protected_nodes.empty() &&
      contact_protected_nodes.size() != graph.nodes) {
    result.diagnostic =
        "Conservative phase geometry reconciliation contact mask does not match the phase graph";
    return result;
  }
  result.contact_protected_nodes = static_cast<std::size_t>(std::count_if(
      contact_protected_nodes.begin(),
      contact_protected_nodes.end(),
      [](std::uint8_t value) { return value != 0u; }));
  const auto phase_field = system.findFieldByName(
      request.conservative_phase.liquid_indicator.field_name);
  const auto level_set_field = system.findFieldByName(
      request.level_set_field_name);
  if (phase_field == svmp::FE::INVALID_FIELD_ID ||
      level_set_field == svmp::FE::INVALID_FIELD_ID) {
    result.diagnostic =
        "Conservative phase geometry reconciliation could not resolve its phase or level-set field";
    return result;
  }
  const auto raw_level_set_offset =
      system.fieldDofOffset(level_set_field);
  const auto raw_level_set_count =
      system.fieldDofHandler(level_set_field).getNumDofs();
  if (raw_level_set_offset < 0 || raw_level_set_count < 0) {
    result.diagnostic =
        "Conservative phase geometry reconciliation found an invalid level-set layout";
    return result;
  }
  const auto level_set_offset =
      static_cast<std::size_t>(raw_level_set_offset);
  const auto level_set_count =
      static_cast<std::size_t>(raw_level_set_count);
  if (level_set_count != graph.nodes ||
      level_set_offset > candidate.size() ||
      level_set_count > candidate.size() - level_set_offset) {
    result.diagnostic =
        "Conservative phase geometry reconciliation requires identical P1 phase and level-set layouts";
    return result;
  }

  const auto projection_options = conservativePhaseProjectionOptions(
      system, request);
  const auto volume_options = levelSetVolumeOptionsForMaintenance(request);
  const auto moment_tolerance =
      request.conservative_phase.geometry_measure_tolerance *
      std::max({svmp::FE::Real{1.0}, graph.physical_measure,
                std::accumulate(target_liquid_phase_mass.begin(),
                                target_liquid_phase_mass.end(),
                                svmp::FE::Real{0.0})});
  if (!std::isfinite(moment_tolerance) ||
      !(moment_tolerance > svmp::FE::Real{0.0})) {
    result.diagnostic =
        "Conservative phase geometry reconciliation has an invalid local moment tolerance";
    return result;
  }

  ++result.geometry_refresh_requests;
  result.geometry_rebuilds +=
      transaction.refresh(params, candidate).refreshed ? 1 : 0;
  for (int iteration = 0;
       iteration <
       request.conservative_phase.geometry_correction_max_iterations;
       ++iteration) {
    const auto projection = projectCurrentConservativePhaseGeometry(
        system, request);
    if (!projection.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase local geometry projection failed: " +
          projection.diagnostic);
    }
    const auto mismatch = conservativePhaseMomentMismatch(
        projection.liquid_phase_mass, target_liquid_phase_mass);
    if (iteration == 0) {
      result.initial_residual_norm = mismatch.residual_norm;
    }
    result.final_residual_norm = mismatch.residual_norm;
    result.maximum_final_nodal_residual =
        mismatch.maximum_nodal_residual;
    result.final_total_residual = mismatch.total_residual;
    if (mismatch.maximum_nodal_residual <= moment_tolerance &&
        std::abs(mismatch.total_residual) <= moment_tolerance) {
      result.success = true;
      result.target_reached = true;
      result.diagnostic = "ok";
      return result;
    }

    const auto sensitivity =
        svmp::FE::level_set::buildLevelSetP1PhaseGeometrySensitivity(
            system,
            level_set_field,
            phase_field,
            graph,
            projection_options,
            candidate);
    if (!sensitivity.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase local geometry sensitivity failed: " +
          sensitivity.diagnostic);
    }
    if (iteration == 0) {
      result.allowed_interface_displacement =
          request.conservative_phase
              .maximum_geometry_displacement_fraction *
          sensitivity.minimum_cell_node_distance;
    }
    const auto current_level_set = std::span<const svmp::FE::Real>(
        candidate.data() + level_set_offset, level_set_count);
    svmp::FE::level_set::LevelSetP1PhaseGeometryCorrectionOptions
        correction_options;
    correction_options.invariant_tolerance =
        request.conservative_phase.invariant_tolerance;
    correction_options.relative_linear_tolerance = std::max(
        request.conservative_phase.invariant_tolerance,
        std::min(svmp::FE::Real{1.0e-8},
                 request.conservative_phase.geometry_measure_tolerance));
    auto correction =
        svmp::FE::level_set::solveLevelSetP1PhaseGeometryCorrection(
            sensitivity,
            projection_options.liquid_side,
            current_level_set,
            projection.liquid_phase_mass,
            target_liquid_phase_mass,
            correction_options);
    if (!correction.success) {
      result.diagnostic =
          "Conservative phase local geometry correction has no admissible shape update: " +
          correction.diagnostic;
      return result;
    }

    for (std::size_t node = 0u;
         node < correction.level_set_increment.size(); ++node) {
      if (!contact_protected_nodes.empty() &&
          contact_protected_nodes[node] != 0u) {
        result.maximum_removed_contact_increment = std::max(
            result.maximum_removed_contact_increment,
            std::abs(correction.level_set_increment[node]));
        correction.level_set_increment[node] = svmp::FE::Real{0.0};
      }
    }

    svmp::FE::Real maximum_increment{0.0};
    for (const auto increment : correction.level_set_increment) {
      maximum_increment = std::max(
          maximum_increment, std::abs(increment));
    }
    if (!(maximum_increment > svmp::FE::Real{0.0}) ||
        !std::isfinite(maximum_increment)) {
      result.diagnostic = result.contact_protected_nodes > 0u
          ? "Conservative phase local geometry correction cannot close the nodal moments without changing accepted wall-contact parent cells"
          : "Conservative phase local geometry correction produced a zero or non-finite update";
      return result;
    }
    const auto full_step_displacement =
        maximum_increment / sensitivity.minimum_level_set_gradient;
    const auto remaining_displacement =
        result.allowed_interface_displacement -
        result.accumulated_interface_displacement_bound;
    if (!(remaining_displacement > svmp::FE::Real{0.0})) {
      result.limited_by_displacement = true;
      result.diagnostic =
          "Conservative phase local geometry correction exhausted its cumulative displacement contract";
      return result;
    }
    svmp::FE::Real step_scale = std::min(
        svmp::FE::Real{1.0},
        remaining_displacement / full_step_displacement);
    if (step_scale < svmp::FE::Real{1.0}) {
      result.limited_by_displacement = true;
    }

    const auto sign_margin =
        request.conservative_phase.invariant_tolerance *
        std::max(svmp::FE::Real{1.0},
                 std::abs(*std::max_element(
                     current_level_set.begin(), current_level_set.end(),
                     [](svmp::FE::Real first, svmp::FE::Real second) {
                       return std::abs(first) < std::abs(second);
                     })));
    for (std::size_t node = 0u; node < level_set_count; ++node) {
      const auto signed_value =
          current_level_set[node] - volume_options.isovalue;
      const auto increment = correction.level_set_increment[node];
      if (signed_value * increment < svmp::FE::Real{0.0}) {
        const auto safe_magnitude =
            std::abs(signed_value) - sign_margin;
        if (!(safe_magnitude > svmp::FE::Real{0.0})) {
          result.limited_by_topology = true;
          result.diagnostic =
              "Conservative phase local geometry correction encountered a level-set node inside its topology margin";
          return result;
        }
        step_scale = std::min(
            step_scale,
            svmp::FE::Real{0.9} * safe_magnitude /
                std::abs(increment));
      }
    }
    if (!(step_scale > svmp::FE::Real{0.0}) ||
        !std::isfinite(step_scale)) {
      result.limited_by_topology = true;
      result.diagnostic =
          "Conservative phase local geometry correction has no topology-stable step";
      return result;
    }

    bool accepted_line_search = false;
    constexpr int maximum_line_search_evaluations = 16;
    for (int line_search = 0;
         line_search < maximum_line_search_evaluations;
         ++line_search) {
      ++result.line_search_evaluations;
      auto trial = candidate;
      for (std::size_t node = 0u; node < level_set_count; ++node) {
        trial[level_set_offset + node] =
            current_level_set[node] +
            step_scale * correction.level_set_increment[node];
      }
      svmp::FE::level_set::LevelSetP1PhaseProjectionResult
          trial_projection;
      try {
        ++result.geometry_refresh_requests;
        result.geometry_rebuilds +=
            transaction.refresh(params, trial).refreshed ? 1 : 0;
        trial_projection = projectCurrentConservativePhaseGeometry(
            system, request);
      } catch (const std::exception& exception) {
        ++result.rejected_geometry_trials;
        result.last_rejected_trial_diagnostic = exception.what();
        step_scale *= svmp::FE::Real{0.5};
        continue;
      }
      if (!trial_projection.success) {
        ++result.rejected_geometry_trials;
        result.last_rejected_trial_diagnostic =
            trial_projection.diagnostic;
        step_scale *= svmp::FE::Real{0.5};
        continue;
      }
      const auto trial_mismatch = conservativePhaseMomentMismatch(
          trial_projection.liquid_phase_mass,
          target_liquid_phase_mass);
      const bool trial_reached =
          trial_mismatch.maximum_nodal_residual <= moment_tolerance &&
          std::abs(trial_mismatch.total_residual) <= moment_tolerance;
      const bool sufficient_decrease =
          trial_mismatch.residual_norm <
          mismatch.residual_norm *
              (svmp::FE::Real{1.0} -
               svmp::FE::Real{1.0e-4} * step_scale);
      if (trial_reached || sufficient_decrease) {
        candidate = std::move(trial);
        result.accumulated_interface_displacement_bound +=
            step_scale * full_step_displacement;
        result.final_residual_norm = trial_mismatch.residual_norm;
        result.maximum_final_nodal_residual =
            trial_mismatch.maximum_nodal_residual;
        result.final_total_residual = trial_mismatch.total_residual;
        result.iterations = iteration + 1;
        accepted_line_search = true;
        break;
      }
      step_scale *= svmp::FE::Real{0.5};
    }
    if (!accepted_line_search) {
      ++result.geometry_refresh_requests;
      result.geometry_rebuilds +=
          transaction.refresh(params, candidate).refreshed ? 1 : 0;
      result.diagnostic =
          "Conservative phase local geometry correction line search did not reduce the nodal moment residual";
      if (!result.last_rejected_trial_diagnostic.empty()) {
        result.diagnostic += ": " +
                             result.last_rejected_trial_diagnostic;
      }
      return result;
    }
  }

  result.diagnostic =
      "Conservative phase local geometry correction did not reach its nodal moment tolerance within the bounded iteration count";
  return result;
}

ConservativePhaseCandidateResult applyConservativePhaseCandidates(
    application::core::SimulationComponents& sim,
    svmp::FE::timestepping::TimeHistory& history,
    std::vector<LevelSetMaintenanceRequest>& requests,
    const Parameters& params,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& refresh_cache,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    const ConservativePhaseContactStageBuilder& contact_stage_builder = {},
    const LevelSetMaintenanceStageObserver& observe_stage = {})
{
  ConservativePhaseCandidateResult result;
  if (!sim.fe_system || requests.empty()) {
    return result;
  }
  const bool any_enabled = std::any_of(
      requests.begin(), requests.end(), [](const auto& request) {
        return request.conservative_phase.enabled;
      });
  if (!any_enabled) {
    return result;
  }
  if (!(history.dt() > 0.0) || !std::isfinite(history.dt())) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport requires a positive finite candidate time step.");
  }

  auto& system = *sim.fe_system;
  result.original_solution = gatherFeOrderedSolution(history.u());
  const auto previous_solution = gatherFeOrderedSolution(history.uPrev());
  if (result.original_solution.size() != previous_solution.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase transport requires equal endpoint and previous FE solution layouts.");
  }
  auto candidate = result.original_solution;
  auto raw_transport_candidate = result.original_solution;
  std::vector<svmp::FE::Real> accepted_phase_measures(
      requests.size(), std::numeric_limits<svmp::FE::Real>::quiet_NaN());
  std::vector<std::vector<svmp::FE::Real>> accepted_phase_masses(
      requests.size());
  std::vector<ConservativePhaseGeometryReconciliationResult>
      reconciliation_reports(requests.size());
  std::vector<std::vector<std::uint8_t>> contact_protected_nodes(
      requests.size());
  result.maintenance_ledgers.resize(requests.size());

  for (std::size_t request_index = 0u;
       request_index < requests.size(); ++request_index) {
    auto& request = requests[request_index];
    if (!request.conservative_phase.enabled) {
      continue;
    }
    if (!request.conservative_phase_initialized) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase transport reached a candidate before its geometry projection was initialized.");
    }
    requireConservativePhaseGeometryBinding(system, request);
    auto& graph = requireCurrentConservativePhaseGraph(system, request);
    const auto phase_field = system.findFieldByName(
        request.conservative_phase.liquid_indicator.field_name);
    const auto raw_phase_offset = system.fieldDofOffset(phase_field);
    const auto raw_phase_count =
        system.fieldDofHandler(phase_field).getNumDofs();
    if (raw_phase_offset < 0 || raw_phase_count < 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase graph found an invalid FE solution layout.");
    }
    const auto phase_offset = static_cast<std::size_t>(raw_phase_offset);
    const auto phase_count = static_cast<std::size_t>(raw_phase_count);
    if (phase_count != graph.nodes ||
        phase_offset > previous_solution.size() ||
        phase_count > previous_solution.size() - phase_offset ||
        phase_offset > candidate.size() ||
        phase_count > candidate.size() - phase_offset) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase graph does not match the FE solution layout.");
    }
    const std::span<const svmp::FE::Real> previous_phase(
        previous_solution.data() + phase_offset, phase_count);
    const auto [lower, upper] = conservativePhaseOneRingBounds(
        graph, previous_phase);

    svmp::FE::systems::SystemStateView state;
    state.time = history.time() + history.dt();
    state.dt = history.dt();
    state.dt_prev = history.dtPrev();
    state.u = std::span<const svmp::FE::Real>(candidate);
    state.u_prev = std::span<const svmp::FE::Real>(previous_solution);

    svmp::FE::level_set::LevelSetBoundPreservingOptions safety_options;
    safety_options.enabled = true;
    safety_options.bound_tolerance =
        request.conservative_phase.invariant_tolerance;
    safety_options.sign_tolerance =
        request.conservative_phase.invariant_tolerance;
    safety_options.enforce_courant_limit = false;
    safety_options.enforce_impermeable_boundaries = true;
    safety_options.impermeable_normal_velocity_tolerance =
        request.conservative_phase.impermeable_normal_velocity_tolerance;
    const auto safety =
        svmp::FE::level_set::evaluateLevelSetTransportSafety(
            system,
            request.velocity,
            svmp::FE::level_set::LevelSetBoundaryOptions{},
            safety_options,
            state,
            static_cast<svmp::FE::Real>(history.dt()));
    if (!safety.impermeable_boundaries_satisfied) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase transport rejected a nonzero wall-normal velocity for field '" +
          request.conservative_phase.liquid_indicator.field_name +
          "': marker=" + std::to_string(safety.worst_boundary_marker) +
          " normalized_flux=" +
          std::to_string(safety.maximum_boundary_normal_velocity_ratio) +
          ". " + safety.diagnostic);
    }
    if (!safety.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase transport safety validation failed for field '" +
          request.conservative_phase.liquid_indicator.field_name +
          "': " + safety.diagnostic);
    }

    const auto velocity = sampleConservativePhaseVelocity(
        sim, request, graph, state);
    svmp::FE::level_set::LevelSetP1PhaseStageOptions stage_options;
    stage_options.invariant_tolerance =
        request.conservative_phase.invariant_tolerance;
    stage_options.component_activity_tolerance =
        request.conservative_phase.component_activity_tolerance;
    stage_options.maximum_courant =
        request.conservative_phase.maximum_courant;
    stage_options.enforce_courant_limit =
        request.conservative_phase.enforce_courant_limit;
    stage_options.require_constant_preservation =
        request.conservative_phase.require_constant_preservation;
    auto stage =
        svmp::FE::level_set::advanceLevelSetP1ConservativePhaseStage(
            graph,
            previous_phase,
            lower,
            upper,
            velocity,
            static_cast<svmp::FE::Real>(history.dt()),
            stage_options);
    if (!stage.success) {
      if (!stage.courant_satisfied &&
          request.conservative_phase.enforce_courant_limit) {
        application::core::oopCout()
            << "[svMultiPhysics::Application] Conservative phase candidate rejected"
            << " field='"
            << request.conservative_phase.liquid_indicator.field_name
            << "' reason=courant_contract"
            << " courant=" << stage.maximum_courant
            << " maximum_courant="
            << request.conservative_phase.maximum_courant
            << " dt=" << history.dt() << std::endl;
        result.accept_step = false;
        return result;
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase graph stage failed for field '" +
          request.conservative_phase.liquid_indicator.field_name +
          "': " + stage.diagnostic);
    }
    std::vector<svmp::FE::Real> limited_phase(graph.nodes,
                                                svmp::FE::Real{0.0});
    std::vector<svmp::FE::Real> raw_phase(
        graph.nodes, svmp::FE::Real{0.0});
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
    const auto phase_measure =
        stage.correction.total_limited_liquid_measure;
    const auto measure_tolerance =
        request.conservative_phase.geometry_measure_tolerance *
        std::max({svmp::FE::Real{1.0}, graph.physical_measure,
                  std::abs(phase_measure)});
    if (!std::isfinite(phase_measure) ||
        phase_measure < -measure_tolerance ||
        phase_measure > graph.physical_measure + measure_tolerance) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase stage produced an invalid liquid measure.");
    }
    accepted_phase_measures[request_index] = phase_measure;
    auto& maintenance_ledger =
        result.maintenance_ledgers[request_index];
    maintenance_ledger.raw_post_transport_phase_measure =
        stage.correction.total_raw_target_liquid_measure;
    maintenance_ledger.post_limit_phase_measure = phase_measure;
    accepted_phase_masses[request_index].resize(
        graph.nodes, svmp::FE::Real{0.0});
    for (std::size_t node = 0u; node < graph.nodes; ++node) {
      accepted_phase_masses[request_index][node] =
          graph.lumped_control_volume[node] * limited_phase[node];
    }
    std::vector<svmp::FE::level_set::LevelSetPhaseRegionDefinition>
        region_definitions;
    if (!request.conservative_phase.fixed_flux_regions.empty()) {
      const auto coordinates = conservativePhaseNodeCoordinates(
          system, phase_field, graph);
      region_definitions =
          svmp::FE::level_set::makeAxisAlignedBoxPhaseRegions(
              request.conservative_phase.fixed_flux_regions,
              coordinates);
    }
    if (request.conservative_phase
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
            request.conservative_phase.invariant_tolerance);
    if (!maintenance_ledger.region_ledger.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase fixed region ledger failed for field '" +
          request.conservative_phase.liquid_indicator.field_name +
          "': " + maintenance_ledger.region_ledger.diagnostic);
    }

    application::core::oopCout()
        << "[svMultiPhysics::Application] Conservative phase staged"
        << " field='"
        << request.conservative_phase.liquid_indicator.field_name << "'"
        << " step=" << history.stepIndex() + 1
        << " previous_measure="
        << stage.correction.total_previous_liquid_measure
        << " accepted_measure=" << phase_measure
        << " boundary_transfer="
        << stage.correction.total_physical_boundary_mass_transfer
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
        << " courant=" << stage.maximum_courant << std::endl;
    maintenance_ledger.transport_stage = std::move(stage);
  }

  auto transaction =
      std::make_unique<LevelSetMaintenanceGeometryTransaction>(
          sim, lifecycle, refresh_cache, active_cut_requests);
  try {
    const auto raw_transport_refresh_report =
        transaction->refresh(params, raw_transport_candidate);
    if (observe_stage) {
      observe_stage(
          application::core::LevelSetMaintenanceWorkSubstage::Transport,
          result.original_solution,
          raw_transport_candidate);
    }
    const auto post_limit_refresh_report =
        transaction->refresh(params, candidate);
    if (observe_stage) {
      observe_stage(
          application::core::LevelSetMaintenanceWorkSubstage::
              Limiting,
          raw_transport_candidate,
          candidate);
    }
    for (std::size_t request_index = 0u;
         request_index < requests.size(); ++request_index) {
      auto& request = requests[request_index];
      if (!request.conservative_phase.enabled) {
        continue;
      }
      const auto projection = projectCurrentConservativePhaseGeometry(
          system, request);
      if (!projection.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase raw post-transport geometry projection failed for field '" +
            request.conservative_phase.liquid_indicator.field_name +
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
      if (!request.conservative_phase.enabled) {
        continue;
      }
      auto& maintenance_ledger =
          result.maintenance_ledgers[request_index];
      maintenance_ledger.reinitialization_due =
          svmp::FE::level_set::shouldReinitializeLevelSet(
              request.reinitialization,
              history.stepIndex() + 1);
      const auto field =
          system.findFieldByName(request.level_set_field_name);
      if (field == svmp::FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase maintenance could not find level-set field '" +
            request.level_set_field_name + "'.");
      }
      const auto wall_context = resolveLevelSetWallAwareMaintenanceContext(
          sim,
          history,
          field,
          static_cast<svmp::FE::Real>(history.time() + history.dt()),
          result.contact_stage.stages,
          result.contact_stage.constraints);
      auto& graph = requireCurrentConservativePhaseGraph(system, request);
      contact_protected_nodes[request_index] =
          conservativePhaseContactProtectedNodes(
              system,
              request,
              graph,
              wall_context.local_constraints);
      if (!maintenance_ledger.reinitialization_due) {
        continue;
      }
      const auto staged_reinitialization =
          stageLevelSetProjectionReinitialization(
              sim,
              history,
              request,
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
        application::core::oopCout()
            << "[svMultiPhysics::Application] Conservative phase candidate rejected"
            << " field='"
            << request.conservative_phase.liquid_indicator.field_name
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
        transaction->refresh(params, candidate);
    if (observe_stage && any_reinitialization_applied) {
      observe_stage(
          application::core::LevelSetMaintenanceWorkSubstage::
              Reinitialization,
          before_reinitialization_candidate,
          candidate);
    }
    const auto before_reconciliation_candidate = candidate;
    bool any_reconciliation = false;
    for (std::size_t request_index = 0u;
         request_index < requests.size(); ++request_index) {
      auto& request = requests[request_index];
      if (!request.conservative_phase.enabled) {
        continue;
      }
      const auto projection = projectCurrentConservativePhaseGeometry(
          system, request);
      if (!projection.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase post-reinitialization geometry projection failed for field '" +
            request.conservative_phase.liquid_indicator.field_name +
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
      if (!request.conservative_phase.enabled ||
          !request.conservative_phase.reconcile_geometry) {
        continue;
      }
      any_reconciliation = true;
      auto reconciliation = reconcileConservativePhaseGeometry(
          sim,
          request,
          params,
          accepted_phase_masses[request_index],
          candidate,
          *transaction,
          contact_protected_nodes[request_index]);
      reconciliation_reports[request_index] = reconciliation;
      result.maintenance_ledgers[request_index].reconciliation =
          reconciliation;
      if (!reconciliation.success || !reconciliation.target_reached) {
        application::core::oopCout()
            << "[svMultiPhysics::Application] Conservative phase candidate rejected"
            << " field='"
            << request.conservative_phase.liquid_indicator.field_name
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
          application::core::LevelSetMaintenanceWorkSubstage::
              GeometryReconciliation,
          before_reconciliation_candidate,
          candidate);
    }
    for (std::size_t request_index = 0u;
         request_index < requests.size(); ++request_index) {
      auto& request = requests[request_index];
      if (!request.conservative_phase.enabled) {
        continue;
      }
      const auto projection = projectCurrentConservativePhaseGeometry(
          system, request);
      if (!projection.success) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase authoritative geometry projection failed for field '" +
            request.conservative_phase.liquid_indicator.field_name +
            "': " + projection.diagnostic);
      }
      const auto phase_measure = accepted_phase_measures[request_index];
      const auto mismatch = std::abs(
          projection.retained_liquid_measure - phase_measure);
      const auto tolerance =
          request.conservative_phase.geometry_measure_tolerance *
          std::max({svmp::FE::Real{1.0},
                    std::abs(projection.retained_liquid_measure),
                    std::abs(phase_measure)});
      const auto local_mismatch = conservativePhaseMomentMismatch(
          projection.liquid_phase_mass,
          accepted_phase_masses[request_index]);
      auto& maintenance_ledger =
          result.maintenance_ledgers[request_index];
      maintenance_ledger.post_correction_geometry_measure =
          projection.retained_liquid_measure;
      maintenance_ledger.post_correction_mismatch = local_mismatch;
      if (request.conservative_phase.reconcile_geometry &&
          (mismatch > tolerance ||
           local_mismatch.maximum_nodal_residual > tolerance ||
           std::abs(local_mismatch.total_residual) > tolerance)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Conservative phase local geometry reconciliation did not match the transported nodal moments for field '" +
            request.conservative_phase.liquid_indicator.field_name +
            "'.");
      }
      const auto& reconciliation =
          reconciliation_reports[request_index];
      application::core::oopCout()
          << "[svMultiPhysics::Application] Conservative phase geometry validated"
          << " field='"
          << request.conservative_phase.liquid_indicator.field_name << "'"
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
          << " interface_displacement_bound="
          << reconciliation.accumulated_interface_displacement_bound
          << " interface_displacement_limit="
          << reconciliation.allowed_interface_displacement
          << " tolerance=" << tolerance
          << " reconciliation_enabled="
          << (request.conservative_phase.reconcile_geometry ? "true"
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
      application::core::oopCout()
          << std::setprecision(17)
          << "[svMultiPhysics::Application] Conservative phase maintenance ledger"
          << " diagnostic=conservative_phase_maintenance_ledger"
          << " field='"
          << request.conservative_phase.liquid_indicator.field_name << "'"
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
          << maintenance_ledger.post_limit_phase_measure
          << " post_correction_geometry_measure="
          << maintenance_ledger.post_correction_geometry_measure
          << " post_correction_max_nodal_mismatch="
          << maintenance_ledger.post_correction_mismatch
                 .maximum_nodal_residual
          << " retained_assembly_measure="
          << projection.retained_liquid_measure
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
          << std::endl;
      const auto emit_component_ledger =
          [&](const auto& component, std::string_view classification) {
        application::core::oopCout()
            << std::setprecision(17)
            << "[svMultiPhysics::Application] Conservative phase component ledger"
            << " diagnostic=conservative_phase_component_ledger"
            << " field='"
            << request.conservative_phase.liquid_indicator.field_name
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
        application::core::oopCout()
            << std::setprecision(17)
            << "[svMultiPhysics::Application] Conservative phase region ledger"
            << " diagnostic=conservative_phase_region_ledger"
            << " field='"
            << request.conservative_phase.liquid_indicator.field_name
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
  if (result.geometry_transaction) {
    result.geometry_transaction->rollback();
    result.geometry_transaction.reset();
  }
  if (result.changed && !result.original_solution.empty()) {
    scatterFeOrderedSolution(history.u(), result.original_solution);
    history.updateGhosts();
  }
  result.changed = false;
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
  writeEffectiveConfigurationArtifact(sim, params, comm);

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
  applyNewtonToleranceEnvOptions(newton_opts);
  newton_opts.accept_inexact_linear_solutions =
      parseBoolEnv("SVMP_NEWTON_ACCEPT_INEXACT_LINEAR", false);
  applyNewtonPseudoTransientEnvOptions(newton_opts);

  // Modified Newton: reuse Jacobian across multiple iterations.
  // Period 1 = full Newton (default), 2 = rebuild every 2nd iteration, etc.
  if (const char* jrp = std::getenv("SVMP_JACOBIAN_REBUILD_PERIOD")) {
    newton_opts.jacobian_rebuild_period = std::atoi(jrp);
  }

  auto cut_lifecycle =
      std::make_shared<svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle>();
  auto cut_refresh_cache =
      std::make_shared<ActiveCutContextRefreshCache>();
  auto level_set_maintenance = levelSetMaintenanceRequests(params);
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
  // Legacy refreshed/quasi-Newton mode remains available for controlled
  // comparisons.  Production generated geometry uses an outer Picard loop so
  // each inner Newton residual and Jacobian share exactly one frozen G.
  newton_opts.accepted_state_sync_invalidates_residual =
      steady_has_generated_state &&
      !use_steady_external_state_fixed_point;
  auto curvature_projection_cache =
      std::make_shared<CurvatureProjectionCache>();
  auto cut_topology_key = std::make_shared<std::optional<std::uint64_t>>();
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
  newton_opts.synchronize_state =
      [&, cut_lifecycle, cut_refresh_cache, cut_topology_key,
       curvature_projection_cache](
          const svmp::FE::systems::SystemStateView& state,
          StateSyncPoint point) {
        if (use_steady_external_state_fixed_point &&
            point != StateSyncPoint::OuterFixedPointState &&
            point != StateSyncPoint::ProjectedOuterFixedPointState &&
            point != StateSyncPoint::RestoredOuterFixedPointState &&
            point !=
                StateSyncPoint::RestoredProjectedOuterFixedPointState) {
          return;
        }
        if (point == StateSyncPoint::RestoredOuterFixedPointState) {
          cut_refresh_cache->last_signature.reset();
          cut_refresh_cache->last_vector_signature.reset();
          curvature_projection_cache->entries.clear();
        }
        const auto report = refreshActiveCutIntegrationContextCached(
            sim, params, state, *cut_lifecycle, *cut_refresh_cache,
            stateSyncPointName(point));
        logCutTopologyChange(report, point, *cut_topology_key, "steady");
        const bool outer_constraint_construction_pass =
            use_steady_external_state_fixed_point &&
            (point == StateSyncPoint::OuterFixedPointState ||
             point == StateSyncPoint::RestoredOuterFixedPointState);
        if (outer_constraint_construction_pass) {
          return;
        }
        (void)projectLevelSetCurvatureFieldsFromState(
            sim,
            state,
            report.evaluated_state_source_revisions,
            level_set_maintenance,
            -1,
            stateSyncPointName(point),
            /*honor_cadence=*/false,
            curvature_projection_cache.get(),
            /*reuse_cached_on_projection_failure=*/
            allowCachedCurvatureAfterProjectionFailure(point));
        (void)updateLevelSetAdvectionVelocitiesFromState(
            sim,
            state,
            steady_level_set_advection_velocity,
            /*refresh_frozen_algebraic_map=*/
                use_steady_external_state_fixed_point);
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

  initializeLevelSetMaintenanceTargets(sim, level_set_maintenance);
  const auto steady_initial_cut_report =
      refreshActiveCutIntegrationContextCached(
          sim,
          params,
          sim.time_history->u(),
          *cut_lifecycle,
          *cut_refresh_cache,
          "steady_initial");
  initializeConservativePhaseStates(sim, level_set_maintenance);
  (void)projectLevelSetCurvatureFieldsFromState(
      sim,
      stateViewForHistory(*sim.time_history),
      steady_initial_cut_report.evaluated_state_source_revisions,
      level_set_maintenance,
      sim.time_history->stepIndex(),
      "steady_initial",
      /*honor_cadence=*/false,
      curvature_projection_cache.get(),
      /*reuse_cached_on_projection_failure=*/false);
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
            << " auxiliary_residual_norm=" << report.auxiliary_residual_norm << std::endl;

  if (!report.converged) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Steady solve did not converge (Newton reached max iterations). "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  sim.fe_system->commitTimeStep();
  const auto steady_accepted_cut_report =
      refreshActiveCutIntegrationContextCached(
          sim,
          params,
          sim.time_history->u(),
          *cut_lifecycle,
          *cut_refresh_cache,
          "steady_accepted");
  (void)projectLevelSetCurvatureFieldsFromState(
      sim,
      stateViewForHistory(*sim.time_history),
      steady_accepted_cut_report.evaluated_state_source_revisions,
      level_set_maintenance,
      sim.time_history->stepIndex(),
      "steady_accepted",
      /*honor_cadence=*/false,
      curvature_projection_cache.get(),
      /*reuse_cached_on_projection_failure=*/false);
  std::vector<AcceptedVelocityExtensionMapRecord>
      steady_velocity_extension_maps;
  (void)updateLevelSetAdvectionVelocities(
      sim,
      *sim.time_history,
      steady_level_set_advection_velocity,
      &steady_velocity_extension_maps);
  AcceptedVelocityExtensionMapRegistry steady_accepted_extension_maps;
  writeAcceptedVelocityExtensionMapArtifacts(
      params,
      steady_velocity_extension_maps,
      /*accepted_step=*/1u,
      solve_time,
      sim.time_history->dt(),
      sim.time_history->u().valueRevision(),
      activeFESystemCommunicator(*sim.fe_system),
      steady_accepted_extension_maps);
  const auto steady_solution =
      gatherFeOrderedSolution(sim.time_history->u());
  const auto steady_contact_stages =
      evaluateAcceptedFreeSurfaceContactStages(
          sim,
          static_cast<svmp::FE::Real>(solve_time),
          svmp::FE::Real{1.0},
          sim.time_history->u().valueRevision(),
          sim.time_history->u().valueRevision(),
          std::span<const svmp::FE::Real>(steady_solution.data(),
                                          steady_solution.size()));
  recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/1u,
      solve_time,
      sim.time_history->dt(),
      sim.time_history->u().valueRevision(),
      steady_contact_stages);
  sim.fe_system->recordAcceptedMeshTangentialBoundaryPolicies(
      /*accepted_step=*/1u,
      solve_time,
      sim.time_history->dt(),
      sim.time_history->u().valueRevision());
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
  opts.t0 = params.general_simulation_parameters.start_time.defined()
                ? params.general_simulation_parameters.start_time.value()
                : 0.0;
  opts.dt = dt;
  opts.t_end = opts.t0 + static_cast<double>(num_steps) * dt;
  opts.max_steps = num_steps;
  opts.scheme = svmp::FE::timestepping::SchemeKind::GeneralizedAlpha;
  if (params.general_simulation_parameters.spectral_radius_of_infinite_time_step.defined()) {
    opts.generalized_alpha_rho_inf = params.general_simulation_parameters.spectral_radius_of_infinite_time_step.value();
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

  const auto transient_active_cut_requests = activeCutVolumeRequests(params);
  // A first-order generalized-alpha state is (u_n, uDot_n).  Starting an
  // active-cut free-surface problem with uDot_n=0 is generally inconsistent
  // with gravity, pressure, and capillary forces and excites the algorithmic
  // startup mode (the capillary-wave benchmark lost roughly half its apparent
  // stiffness this way).  TimeLoop regularizes only exact zero-mass rows
  // outside a retained cut domain, so the PDE-consistent solve is also the
  // correctness default for active-cut systems.  Keep the environment switch
  // as an explicit legacy/performance comparison opt-out.
  opts.initialize_first_order_rate_from_pde =
      generalizedAlphaPdeRateInitializationRequested(
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

  oopCout() << "[svMultiPhysics::Application] Transient solve: t0=" << opts.t0 << " dt=" << opts.dt
            << " t_end=" << opts.t_end << " max_steps=" << opts.max_steps
            << " scheme=GeneralizedAlpha rho_inf=" << opts.generalized_alpha_rho_inf
            << " pde_udot_init=" << (opts.initialize_first_order_rate_from_pde ? 1 : 0)
            << " last_step_absorb_fraction=" << opts.last_step_absorb_fraction
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
  // Generated cut rules, projected curvature, active-set constraints, and the
  // wet-extension map form one residual-defining package G(u).  The assembled
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
  opts.newton.external_state_fixed_point.max_iterations = std::max(
      1,
      parseIntEnv("SVMP_GENERATED_STATE_OUTER_MAX_ITERATIONS", 12));
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
        << std::endl;
  } else if (has_frozen_algebraic_level_set_extension) {
    oopCout()
        << "[svMultiPhysics::Application] Level-set nonlinear geometry policy: "
           "legacy fixed_topology comparison mode."
        << std::endl;
  }
  // When generated data is refreshed at accepted iterates, reassemble the
  // residual after that refresh before deciding that Newton has converged.
  opts.newton.accepted_state_sync_invalidates_residual =
      refresh_generated_geometry_within_solve &&
      has_transient_generated_state;
  const bool synchronize_line_search_trials =
      synchronizeTransientLineSearchTrials(
          opts.newton.accepted_state_sync_invalidates_residual);
  auto curvature_projection_cache =
      std::make_shared<CurvatureProjectionCache>();
  auto cut_lifecycle =
      std::make_shared<svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle>();
  auto cut_refresh_cache =
      std::make_shared<ActiveCutContextRefreshCache>();
  // Seed wet-volume drift diagnostics from the actual initialized level-set
  // state.  Waiting until the first accepted step silently made that state
  // the reference and hid all first-step volume loss.
  std::map<std::string, svmp::FE::Real> initial_wet_volume_by_key;
  initializeLevelSetMaintenanceTargets(sim, level_set_maintenance);
  const auto initial_cut_report = refreshActiveCutIntegrationContextCached(
      sim, params, sim.time_history->u(), *cut_lifecycle, *cut_refresh_cache,
      "initial");
  initializeConservativePhaseStates(sim, level_set_maintenance);
  if (parseBoolEnv("SVMP_CUT_RULE_DUMP", false)) {
    dumpActiveCutVolumeRulesForProbe(*sim.fe_system,
                                     sim.time_history->stepIndex());
  }
  (void)projectLevelSetCurvatureFieldsFromState(
      sim,
      stateViewForHistory(*sim.time_history),
      initial_cut_report.evaluated_state_source_revisions,
      level_set_maintenance,
      sim.time_history->stepIndex(),
      "initial",
      /*honor_cadence=*/false,
      curvature_projection_cache.get(),
      /*reuse_cached_on_projection_failure=*/false);
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
  AcceptedVelocityExtensionMapRegistry accepted_velocity_extension_maps;
  application::core::LevelSetMaintenanceWorkLedger
      level_set_maintenance_work;
  std::uint64_t level_set_maintenance_transaction_sequence{0u};
  std::uint64_t step_acceptance_attempt{0u};
  std::optional<int> step_acceptance_attempt_step{};
  const auto velocity_extension_artifact_comm =
      activeFESystemCommunicator(*sim.fe_system);
  callbacks.on_step_start = [&](const svmp::FE::timestepping::TimeHistory& h) {
    if (!step_acceptance_attempt_step.has_value() ||
        *step_acceptance_attempt_step != h.stepIndex()) {
      step_acceptance_attempt_step = h.stepIndex();
      step_acceptance_attempt = 0u;
    }
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_start step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << std::endl;
  };
  auto cut_topology_key = std::make_shared<std::optional<std::uint64_t>>();
  applyJacobianCheckGeometryProvenance(
      opts.newton,
      transient_active_cut_requests,
      refresh_generated_geometry_within_solve,
      has_frozen_algebraic_level_set_extension,
      use_transient_external_state_fixed_point);
  const auto expected_cut_request_policy_key =
      activeCutVolumeRequestPolicyKey(transient_active_cut_requests);
  using TransientStateSyncPoint =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  opts.newton.synchronize_state =
      [&, cut_lifecycle, cut_refresh_cache, cut_topology_key,
       curvature_projection_cache](
          const svmp::FE::systems::SystemStateView& state,
          TransientStateSyncPoint point) {
        if (use_transient_external_state_fixed_point &&
            point != TransientStateSyncPoint::OuterFixedPointState &&
            point !=
                TransientStateSyncPoint::ProjectedOuterFixedPointState &&
            point != TransientStateSyncPoint::EndpointCandidateState &&
            point !=
                TransientStateSyncPoint::ProjectedEndpointCandidateState &&
            point != TransientStateSyncPoint::RestoredOuterFixedPointState &&
            point != TransientStateSyncPoint::
                         RestoredProjectedOuterFixedPointState &&
            point != TransientStateSyncPoint::RestoredTimeStepState &&
            point != TransientStateSyncPoint::
                         RestoredProjectedTimeStepState) {
          return;
        }
        if (point ==
                TransientStateSyncPoint::RestoredOuterFixedPointState ||
            point == TransientStateSyncPoint::RestoredTimeStepState) {
          cut_refresh_cache->last_signature.reset();
          cut_refresh_cache->last_vector_signature.reset();
          curvature_projection_cache->entries.clear();
        }
        // The Jacobian may remain a refreshed-frozen/quasi-Newton tangent, but
        // line search globalizes the actual residual R(u,G(u)).  An explicit
        // environment override can freeze trials for a controlled legacy
        // comparison; it is not the production default when G changes the
        // residual.
        if (!use_transient_external_state_fixed_point &&
            point == TransientStateSyncPoint::LineSearchTrialResidual &&
            !synchronize_line_search_trials) {
          return;
        }
        if (!use_transient_external_state_fixed_point &&
            !refresh_generated_geometry_within_solve) {
          return;
        }
        // Refresh-cadence experiment knob: freeze the cut geometry (and the
        // φ-derived curvature/advection data) across ALL within-solve sync
        // points, so the integration domains move once per accepted STEP
        // (before_physics_solve + accepted_step callbacks) instead of after
        // every accepted Newton iterate. CAUTION: this changes Newton's
        // fixed point — the converged quadrature lags the converged φ by
        // one solve.
        if (!use_transient_external_state_fixed_point &&
            per_step_only_generated_state) {
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
        const bool outer_constraint_construction_pass =
            (use_transient_external_state_fixed_point &&
             (point == TransientStateSyncPoint::OuterFixedPointState ||
              point ==
                  TransientStateSyncPoint::RestoredOuterFixedPointState)) ||
            point == TransientStateSyncPoint::EndpointCandidateState ||
            point == TransientStateSyncPoint::RestoredTimeStepState;
        if (outer_constraint_construction_pass) {
          return;
        }
        (void)projectLevelSetCurvatureFieldsFromState(
            sim,
            state,
            report.evaluated_state_source_revisions,
            level_set_maintenance,
            -1,
            stateSyncPointName(point),
            /*honor_cadence=*/false,
            curvature_projection_cache.get(),
            /*reuse_cached_on_projection_failure=*/
            allowCachedCurvatureAfterProjectionFailure(point));
        (void)updateLevelSetAdvectionVelocitiesFromState(
            sim,
            state,
            level_set_advection_velocity,
            /*refresh_frozen_algebraic_map=*/
                use_transient_external_state_fixed_point);
      };
  callbacks.on_before_physics_solve =
      [&](svmp::FE::timestepping::TimeHistory& h, double /*solve_time*/, double /*dt*/) {
        const auto before_solve_cut_report =
            refreshActiveCutIntegrationContextCached(
                sim,
                params,
                h.u(),
                *cut_lifecycle,
                *cut_refresh_cache,
                "before_physics_solve");
        (void)projectLevelSetCurvatureFieldsFromState(
            sim,
            stateViewForHistory(h),
            before_solve_cut_report.evaluated_state_source_revisions,
            level_set_maintenance,
            h.stepIndex(),
            "before_physics_solve",
            /*honor_cadence=*/false,
            curvature_projection_cache.get(),
            /*reuse_cached_on_projection_failure=*/false);
        (void)updateLevelSetAdvectionVelocities(
            sim, h, level_set_advection_velocity);
        if (acceptedPressureUpdateDiagnosticEnabled()) {
          accepted_pressure_update_previous_solution =
              gatherFeOrderedSolution(h.u());
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
              << " rel=" << nr.linear.relative_residual << ")" << std::endl;
  };
  const auto generalized_alpha_first_order =
      svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(
          opts.generalized_alpha_rho_inf);
  const bool has_conservative_phase = std::any_of(
      level_set_maintenance.begin(),
      level_set_maintenance.end(),
      [](const auto& request) {
        return request.conservative_phase.enabled;
      });
  const bool has_bound_preserving_maintenance = std::any_of(
      level_set_maintenance.begin(),
      level_set_maintenance.end(),
      [](const auto& request) {
        return request.bound_preserving.enabled;
      });
  const bool has_precommit_maintenance =
      has_conservative_phase || has_bound_preserving_maintenance;
  const bool has_dynamic_contact_stage = std::any_of(
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations().begin(),
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations().end(),
      [](const auto& declaration) {
        return !declaration.parameters.dynamic_contact_coefficients.empty();
      });
  if (has_dynamic_contact_stage &&
      opts.scheme != svmp::FE::timestepping::SchemeKind::BackwardEuler &&
      opts.scheme != svmp::FE::timestepping::SchemeKind::BDF2 &&
      opts.scheme != svmp::FE::timestepping::SchemeKind::VSVO_BDF &&
      opts.scheme != svmp::FE::timestepping::SchemeKind::GeneralizedAlpha) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic contact-stage provenance currently supports backward Euler, BDF2, VSVO BDF, and generalized-alpha time integration only.");
  }
  std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
      pending_contact_stages;
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
      pending_contact_stage_constraints;
  std::vector<svmp::FE::Real> pending_contact_stage_solution;
  ConservativePhaseCandidateResult pending_phase_candidate;
  ConservativePhaseCandidateResult accepted_phase_candidate;
  std::size_t pending_phase_accepted_row_begin{0u};
  std::size_t pending_phase_rejected_row_begin{0u};
  std::size_t pending_phase_accepted_attempt_begin{0u};
  std::size_t pending_phase_rejected_attempt_begin{0u};
  const auto reject_pending_phase_work = [&] {
    if (!level_set_maintenance_work.transactionActive()) {
      return;
    }
    level_set_maintenance_work.rejectTransaction();
    const auto& attempts =
        level_set_maintenance_work.rejectedAttempts();
    logLevelSetMaintenanceWorkAttempts(
        std::span<const application::core::
                            LevelSetMaintenanceWorkAttempt>(attempts)
            .subspan(pending_phase_rejected_attempt_begin));
    const auto& rows = level_set_maintenance_work.rejectedRows();
    logLevelSetMaintenanceWorkRows(
        std::span<const application::core::
                            LevelSetMaintenanceWorkRow>(rows)
            .subspan(pending_phase_rejected_row_begin));
  };
  const auto commit_pending_phase_work = [&] {
    if (!level_set_maintenance_work.transactionActive()) {
      return;
    }
    level_set_maintenance_work.commitTransaction();
    const auto& attempts =
        level_set_maintenance_work.acceptedAttempts();
    logLevelSetMaintenanceWorkAttempts(
        std::span<const application::core::
                            LevelSetMaintenanceWorkAttempt>(attempts)
            .subspan(pending_phase_accepted_attempt_begin));
    const auto& rows = level_set_maintenance_work.acceptedRows();
    logLevelSetMaintenanceWorkRows(
        std::span<const application::core::
                            LevelSetMaintenanceWorkRow>(rows)
            .subspan(pending_phase_accepted_row_begin));
  };
  callbacks.on_before_step_accept =
      [&](svmp::FE::timestepping::TimeHistory& h,
          const svmp::FE::timestepping::NewtonReport& nr) {
        (void)nr;
        ++step_acceptance_attempt;
        pending_contact_stages.clear();
        pending_contact_stage_constraints.clear();
        pending_contact_stage_solution.clear();
        if (pending_phase_candidate.geometry_transaction) {
          throw std::logic_error(
              "[svMultiPhysics::Application] A conservative phase candidate remained active at the start of a new acceptance attempt.");
        }
        if (level_set_maintenance_work.transactionActive()) {
          throw std::logic_error(
              "[svMultiPhysics::Application] A level-set maintenance work transaction remained active at the start of a new acceptance attempt.");
        }
        pending_phase_candidate = ConservativePhaseCandidateResult{};
        try {
          std::vector<
              application::core::LevelSetAuthoritativeFunctionalValue>
              staged_phase_functionals;
          if (has_precommit_maintenance) {
            pending_phase_accepted_row_begin =
                level_set_maintenance_work.acceptedRows().size();
            pending_phase_rejected_row_begin =
                level_set_maintenance_work.rejectedRows().size();
            pending_phase_accepted_attempt_begin =
                level_set_maintenance_work.acceptedAttempts().size();
            pending_phase_rejected_attempt_begin =
                level_set_maintenance_work.rejectedAttempts().size();
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
                            *sim.fe_system,
                            level_set_advection_velocity,
                            accepted_velocity_extension_maps),
                });
            staged_phase_functionals =
                levelSetMaintenanceFunctionalValues(
                    sim,
                    evaluateCurrentFreeSurfaceDiscreteFunctionals(sim));
          }
          const auto build_contact_stage_candidate =
              [&](std::span<const svmp::FE::Real> endpoint_solution,
                  LevelSetMaintenanceGeometryTransaction*
                      active_transaction) {
                if (!has_dynamic_contact_stage) {
                  return ConservativePhaseContactStageCandidate{};
                }
                const svmp::FE::Real alpha_f =
                    opts.scheme == svmp::FE::timestepping::SchemeKind::
                                       GeneralizedAlpha
                    ? static_cast<svmp::FE::Real>(
                          generalized_alpha_first_order.alpha_f)
                    : svmp::FE::Real{1.0};
                const auto previous_solution =
                    gatherFeOrderedSolution(h.uPrev());
                return buildAcceptedFreeSurfaceContactStageCandidate(
                    sim,
                    params,
                    *cut_lifecycle,
                    *cut_refresh_cache,
                    transient_active_cut_requests,
                    static_cast<svmp::FE::Real>(h.time()) +
                        alpha_f * static_cast<svmp::FE::Real>(h.dt()),
                    alpha_f,
                    h.uPrev().valueRevision(),
                    h.u().valueRevision(),
                    previous_solution,
                    endpoint_solution,
                    active_transaction);
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
                const auto after_functionals =
                    levelSetMaintenanceFunctionalValues(
                        sim,
                        evaluateCurrentFreeSurfaceDiscreteFunctionals(sim));
                level_set_maintenance_work.stageRow(
                    substage,
                    collectiveLevelSetMaintenanceAlgebraicRevision(
                        before_candidate,
                        velocity_extension_artifact_comm),
                    collectiveLevelSetMaintenanceAlgebraicRevision(
                        after_candidate,
                        velocity_extension_artifact_comm),
                    staged_phase_functionals,
                    after_functionals,
                    currentLevelSetVelocityExtensionRevision(
                        *sim.fe_system,
                        level_set_advection_velocity,
                        accepted_velocity_extension_maps));
                staged_phase_functionals = after_functionals;
              };
          pending_phase_candidate = applyConservativePhaseCandidates(
              sim,
              h,
              level_set_maintenance,
              params,
              *cut_lifecycle,
              *cut_refresh_cache,
              transient_active_cut_requests,
              contact_stage_builder,
              has_conservative_phase
                  ? observe_phase_stage
                  : LevelSetMaintenanceStageObserver{});
          if (!pending_phase_candidate.accept_step) {
            rollbackConservativePhaseCandidate(
                h, pending_phase_candidate);
            reject_pending_phase_work();
            return false;
          }
          if (activePressureUpdateRejectOnTriggerEnabled() &&
              acceptedPressureUpdateDiagnosticEnabled() &&
              !accepted_pressure_update_previous_solution.empty()) {
            const auto current_solution = gatherFeOrderedSolution(h.u());
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
              rollbackConservativePhaseCandidate(
                  h, pending_phase_candidate);
              reject_pending_phase_work();
              return false;
            }
          }
          if (has_dynamic_contact_stage) {
            auto contact_stage =
                std::move(pending_phase_candidate.contact_stage);
            if (contact_stage.stages.empty()) {
              const auto endpoint_solution =
                  gatherFeOrderedSolution(h.u());
              contact_stage = build_contact_stage_candidate(
                  endpoint_solution, nullptr);
            }
            pending_contact_stages = std::move(contact_stage.stages);
            pending_contact_stage_constraints =
                std::move(contact_stage.constraints);
            pending_contact_stage_solution =
                std::move(contact_stage.stage_solution);
          }
          const auto bound_result = applyLevelSetBoundPreservingCandidates(
              sim,
              h,
              level_set_maintenance,
              generalized_alpha_first_order.gamma,
              [&](application::core::LevelSetMaintenanceWorkSubstage
                      substage,
                  std::span<const svmp::FE::Real> before_candidate,
                  std::span<const svmp::FE::Real> after_candidate) {
                if (!pending_phase_candidate.geometry_transaction) {
                  pending_phase_candidate.geometry_transaction =
                      std::make_unique<
                          LevelSetMaintenanceGeometryTransaction>(
                          sim,
                          *cut_lifecycle,
                          *cut_refresh_cache,
                          transient_active_cut_requests);
                }
                (void)pending_phase_candidate.geometry_transaction
                    ->refresh(params, after_candidate);
                const auto after_functionals =
                    levelSetMaintenanceFunctionalValues(
                        sim,
                        evaluateCurrentFreeSurfaceDiscreteFunctionals(sim));
                level_set_maintenance_work.stageRow(
                    substage,
                    collectiveLevelSetMaintenanceAlgebraicRevision(
                        before_candidate,
                        velocity_extension_artifact_comm),
                    collectiveLevelSetMaintenanceAlgebraicRevision(
                        after_candidate,
                        velocity_extension_artifact_comm),
                    staged_phase_functionals,
                    after_functionals,
                    currentLevelSetVelocityExtensionRevision(
                        *sim.fe_system,
                        level_set_advection_velocity,
                        accepted_velocity_extension_maps));
                staged_phase_functionals = after_functionals;
              });
          if (!bound_result.accept_step) {
            pending_contact_stages.clear();
            pending_contact_stage_constraints.clear();
            pending_contact_stage_solution.clear();
            rollbackConservativePhaseCandidate(
                h, pending_phase_candidate);
            reject_pending_phase_work();
            return false;
          }
          return true;
        } catch (...) {
          const auto failure = std::current_exception();
          try {
            rollbackConservativePhaseCandidate(
                h, pending_phase_candidate);
          } catch (const std::exception& rollback_error) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] Conservative phase candidate rollback failed: " +
                std::string(rollback_error.what()));
          }
          reject_pending_phase_work();
          std::rethrow_exception(failure);
        }
      };
  callbacks.on_step_candidate_discarded =
      [&](svmp::FE::timestepping::TimeHistory& h) {
        pending_contact_stages.clear();
        pending_contact_stage_constraints.clear();
        pending_contact_stage_solution.clear();
        rollbackConservativePhaseCandidate(h, pending_phase_candidate);
        pending_phase_candidate = ConservativePhaseCandidateResult{};
        reject_pending_phase_work();
      };
  callbacks.on_step_commit_ready =
      [&](svmp::FE::timestepping::TimeHistory& h) {
        const bool changed = pending_phase_candidate.changed;
        if (pending_phase_candidate.geometry_transaction) {
          pending_phase_candidate.geometry_transaction->commit();
          pending_phase_candidate.geometry_transaction.reset();
          oopCout()
              << "[svMultiPhysics::Application] Conservative phase candidate"
              << " step=" << h.stepIndex() + 1
              << " outcome=committed"
              << " geometry_validated_before_commit=true"
              << " accepted_state_change="
              << (changed ? "true" : "false")
              << std::endl;
        }
        commit_pending_phase_work();
        accepted_phase_candidate = std::move(pending_phase_candidate);
        pending_phase_candidate = ConservativePhaseCandidateResult{};
      };
  double vtk_total_time = 0.0;
  callbacks.on_step_accepted = [&](svmp::FE::timestepping::TimeHistory& h) {
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_accepted step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << std::endl;
    writeAcceptedConservativePhaseArtifacts(
        params,
        level_set_maintenance,
        accepted_phase_candidate,
        static_cast<std::uint64_t>(h.stepIndex()),
        static_cast<svmp::FE::Real>(h.time()),
        static_cast<svmp::FE::Real>(h.dt()),
        h.u().valueRevision(),
        svmp::MeshComm::world());
    accepted_phase_candidate = ConservativePhaseCandidateResult{};
    if (acceptedPressureUpdateDiagnosticEnabled() &&
        !accepted_pressure_update_previous_solution.empty()) {
      const auto current_solution = gatherFeOrderedSolution(h.u());
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
    const bool volume_maintenance_due = std::any_of(
        level_set_maintenance.begin(),
        level_set_maintenance.end(),
        [&](const auto& request) {
          return svmp::FE::level_set::shouldApplyLevelSetVolumeCorrection(
              request.volume_correction, h.stepIndex());
        });
    const bool maintenance_work_due = std::any_of(
        level_set_maintenance.begin(),
        level_set_maintenance.end(),
        [&](const auto& request) {
          return !request.conservative_phase.enabled &&
                 (svmp::FE::level_set::shouldReinitializeLevelSet(
                      request.reinitialization, h.stepIndex()) ||
                  svmp::FE::level_set::
                      shouldApplyLevelSetVolumeCorrection(
                          request.volume_correction, h.stepIndex()));
        });
    std::vector<
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        pre_maintenance_functionals;
    if (maintenance_work_due || volume_maintenance_due) {
      pre_maintenance_functionals =
          evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
    }
    std::vector<LevelSetVolumeCorrectionMaintenanceEvent>
        applied_volume_corrections;
    std::vector<std::string> volume_correction_work_logs;
    ActiveCutContextRefreshReport cut_report{};
    std::unique_ptr<LevelSetMaintenanceGeometryTransaction>
        maintenance_geometry_transaction;
    bool level_set_maintenance_changed = false;
    const auto accepted_row_begin =
        level_set_maintenance_work.acceptedRows().size();
    const auto rejected_row_begin =
        level_set_maintenance_work.rejectedRows().size();
    const auto accepted_attempt_begin =
        level_set_maintenance_work.acceptedAttempts().size();
    const auto rejected_attempt_begin =
        level_set_maintenance_work.rejectedAttempts().size();
    auto staged_maintenance_functionals =
        levelSetMaintenanceFunctionalValues(
            sim,
            pre_maintenance_functionals);
    if (maintenance_work_due) {
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
                      *sim.fe_system,
                      level_set_advection_velocity,
                      accepted_velocity_extension_maps),
          });
    }
    try {
      const auto ensure_maintenance_geometry_transaction = [&] {
        if (!maintenance_geometry_transaction) {
          maintenance_geometry_transaction =
              std::make_unique<LevelSetMaintenanceGeometryTransaction>(
                  sim,
                  *cut_lifecycle,
                  *cut_refresh_cache,
                  transient_active_cut_requests);
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
            const auto after_functional_states =
                evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
            auto after_functionals =
                levelSetMaintenanceFunctionalValues(
                    sim,
                    after_functional_states);
            level_set_maintenance_work.stageRow(
                substage,
                collectiveLevelSetMaintenanceAlgebraicRevision(
                    before_candidate,
                    velocity_extension_artifact_comm),
                collectiveLevelSetMaintenanceAlgebraicRevision(
                    after_candidate,
                    velocity_extension_artifact_comm),
                staged_maintenance_functionals,
                after_functionals,
                currentLevelSetVelocityExtensionRevision(
                    *sim.fe_system,
                    level_set_advection_velocity,
                    accepted_velocity_extension_maps));
            staged_maintenance_functionals =
                std::move(after_functionals);
          };
      const LevelSetMaintenanceCandidateValidator validate_candidate =
          [&](std::span<const svmp::FE::Real> candidate,
              std::span<const LevelSetVolumeCorrectionMaintenanceEvent>
                  staged_volume_corrections) {
            ensure_maintenance_geometry_transaction();
            cut_report = maintenance_geometry_transaction->refresh(
                params, candidate);
            if (!staged_volume_corrections.empty()) {
              const auto post_maintenance_functionals =
                  evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
              volume_correction_work_logs =
                  buildLevelSetVolumeCorrectionFreeSurfaceWorkLogs(
                      sim,
                      staged_volume_corrections,
                      pre_maintenance_functionals,
                      post_maintenance_functionals);
            }
          };
      level_set_maintenance_changed = applyLevelSetMaintenance(
          sim,
          h,
          level_set_maintenance,
          pending_contact_stages,
          pending_contact_stage_constraints,
          pending_contact_stage_solution,
          &applied_volume_corrections,
          validate_candidate,
          observe_stage);
      if (maintenance_geometry_transaction) {
        maintenance_geometry_transaction->commit();
        oopCout()
            << "[svMultiPhysics::Application] Level-set maintenance transaction"
            << " diagnostic=level_set_maintenance_transaction"
            << " step=" << h.stepIndex()
            << " outcome=committed"
            << " geometry_validated_before_commit=true"
            << " accepted_state_change="
            << (level_set_maintenance_changed ? "true" : "false")
            << " volume_corrections="
            << applied_volume_corrections.size() << std::endl;
      }
      if (level_set_maintenance_work.transactionActive()) {
        level_set_maintenance_work.commitTransaction();
        const auto& accepted_attempts =
            level_set_maintenance_work.acceptedAttempts();
        logLevelSetMaintenanceWorkAttempts(
            std::span<const application::core::
                                LevelSetMaintenanceWorkAttempt>(
                accepted_attempts).subspan(accepted_attempt_begin));
        const auto& accepted_rows =
            level_set_maintenance_work.acceptedRows();
        logLevelSetMaintenanceWorkRows(
            std::span<const
                application::core::LevelSetMaintenanceWorkRow>(
                accepted_rows).subspan(accepted_row_begin));
      }
    } catch (...) {
      const auto failure = std::current_exception();
      std::string rollback_failure;
      if (maintenance_geometry_transaction) {
        try {
          maintenance_geometry_transaction->rollback();
        } catch (const std::exception& error) {
          rollback_failure = error.what();
        } catch (...) {
          rollback_failure = "unknown rollback failure";
        }
      }
      oopCout()
          << "[svMultiPhysics::Application] Level-set maintenance transaction"
          << " diagnostic=level_set_maintenance_transaction"
          << " step=" << h.stepIndex()
          << " outcome="
          << (rollback_failure.empty() ? "rolled_back"
                                       : "rollback_failed")
          << " accepted_state_change=false"
          << " geometry_state_restored="
          << (rollback_failure.empty() ? "true" : "false")
          << std::endl;
      if (rollback_failure.empty() &&
          level_set_maintenance_work.transactionActive()) {
        level_set_maintenance_work.rejectTransaction();
        const auto& rejected_attempts =
            level_set_maintenance_work.rejectedAttempts();
        logLevelSetMaintenanceWorkAttempts(
            std::span<const application::core::
                                LevelSetMaintenanceWorkAttempt>(
                rejected_attempts).subspan(rejected_attempt_begin));
        const auto& rejected_rows =
            level_set_maintenance_work.rejectedRows();
        logLevelSetMaintenanceWorkRows(
            std::span<const
                application::core::LevelSetMaintenanceWorkRow>(
                rejected_rows).subspan(rejected_row_begin));
      }
      if (!rollback_failure.empty()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set maintenance rollback failed after candidate failure: " +
            rollback_failure);
      }
      std::rethrow_exception(failure);
    }
    if (!level_set_maintenance_changed) {
      cut_report = refreshActiveCutIntegrationContextCached(
          sim, params, h.u(), *cut_lifecycle, *cut_refresh_cache,
          "accepted_step");
    }
    for (const auto& log : volume_correction_work_logs) {
      oopCout() << log << std::endl;
    }
    if (parseBoolEnv("SVMP_CUT_RULE_DUMP", false)) {
      dumpActiveCutVolumeRulesForProbe(*sim.fe_system, h.stepIndex());
    }
    (void)projectLevelSetCurvatureFieldsFromState(
        sim,
        stateViewForHistory(h),
        cut_report.evaluated_state_source_revisions,
        level_set_maintenance,
        h.stepIndex(),
        "accepted_step",
        /*honor_cadence=*/false,
        curvature_projection_cache.get(),
        /*reuse_cached_on_projection_failure=*/false);
    std::vector<AcceptedVelocityExtensionMapRecord>
        accepted_extension_map_records;
    (void)updateLevelSetAdvectionVelocities(
        sim,
        h,
        level_set_advection_velocity,
        &accepted_extension_map_records);
    writeAcceptedVelocityExtensionMapArtifacts(
        params,
        accepted_extension_map_records,
        static_cast<std::uint64_t>(h.stepIndex()),
        h.time(),
        h.dt(),
        h.u().valueRevision(),
        velocity_extension_artifact_comm,
        accepted_velocity_extension_maps);
    recordAcceptedFreeSurfaceDiscreteFunctionals(
        sim,
        static_cast<std::uint64_t>(h.stepIndex()),
        h.time(),
        h.dt(),
        h.u().valueRevision(),
        pending_contact_stages);
    pending_contact_stages.clear();
    pending_contact_stage_constraints.clear();
    pending_contact_stage_solution.clear();
    sim.fe_system->recordAcceptedMeshTangentialBoundaryPolicies(
        static_cast<std::uint64_t>(h.stepIndex()),
        h.time(),
        h.dt(),
        h.u().valueRevision());
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
          << " generated_cell_cache_unchanged_dof_hits="
          << cut_report.generated_cell_cache_unchanged_dof_hits
          << " generated_cell_refresh_candidates="
          << cut_report.generated_cell_refresh_candidates
          << " generated_cell_directly_affected="
          << cut_report.generated_cell_directly_affected
          << " generated_cell_affected_neighborhood="
          << cut_report.generated_cell_affected_neighborhood
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
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_rejected step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << " reason=" << step_reject_reason_to_string(reason)
              << " (newton: converged=" << nr.converged << " iters=" << nr.iterations
              << " outer_iters=" << nr.outer_iterations
              << " inner_iters_total=" << nr.inner_iterations_total
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

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/Maintenance/LevelSetMaintenancePipeline.h"

#include "Backends/Interfaces/GenericVector.h"
#include "Core/RankLog.h"
#include "Dofs/DofHandler.h"
#include "LevelSet/LevelSetGeometry.h"
#include "LevelSet/LevelSetInterfacePreservingRedistancing.h"
#include "LevelSet/LevelSetInterfacePreservingRedistancing3D.h"
#include "LevelSet/LevelSetKinematicReconciliation.h"
#include "LevelSet/LevelSetSignDefinitePatchBounds.h"
#include "LevelSet/LevelSetTransport.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemState.h"
#include "TimeStepping/FieldRepairPublication.h"
#include "TimeStepping/TimeHistory.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
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

std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveMinMaxUint64(local, comm);
}

double globalMaxAbsDifference(
    std::span<const svmp::FE::Real> left,
    std::span<const svmp::FE::Real> right,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveMaxAbsDifference(left, right, comm);
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::captureFeOrderedVectorCollectively(
      solution, comm);
}

std::vector<svmp::FE::Real> capturePostacceptMaintenanceVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::captureFeOrderedVectorCollectively(
      solution, comm);
}

// Publication of maintained-field repairs:
// FE/TimeStepping/FieldRepairPublication.
using svmp::FE::timestepping::StagedPostacceptMaintenanceHistoryPublication;
using svmp::FE::timestepping::makeCommonRepresentationMaintainedFieldPublication;
using svmp::FE::timestepping::makeStagedPostacceptMaintenanceHistoryPublication;
using svmp::FE::timestepping::publishStagedPostacceptMaintenanceHistoryCollectively;
using svmp::FE::timestepping::stageMaintainedFieldPostAcceptPublication;

} // namespace

// Moved from Application/Core/ApplicationDriver.cpp with bodies, messages and
// collective order unchanged.

svmp::FE::level_set::LevelSetVolumeOptions levelSetVolumeOptionsForMaintenance(
    const LevelSetMaintenanceConfiguration& configuration)
{
  return volumeOptionsForCutMaintenance(
      configuration.volume_cut_request,
      configuration.transport.level_set.field_name,
      configuration.isovalue);
}

std::vector<svmp::FE::Real> levelSetTransportVelocityCoefficients(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId velocity_field,
    std::span<const svmp::FE::Real> fe_solution)
{
  const auto count = static_cast<std::size_t>(
      system.fieldDofHandler(velocity_field).getNumDofs());
  const auto& record = system.fieldRecord(velocity_field);
  if (record.source_kind ==
      svmp::FE::systems::FieldSourceKind::PrescribedData) {
    const auto coefficients =
        system.prescribedFieldCoefficients(velocity_field);
    if (coefficients.size() != count) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set kinematic reconciliation "
          "found no complete coefficients for prescribed transport velocity '" +
          record.name + "'.");
    }
    return {coefficients.begin(), coefficients.end()};
  }
  const auto raw_offset = system.fieldDofOffset(velocity_field);
  if (!system.fieldParticipatesInUnknownVector(velocity_field) ||
      raw_offset < 0 ||
      static_cast<std::size_t>(raw_offset) > fe_solution.size() ||
      count > fe_solution.size() - static_cast<std::size_t>(raw_offset)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set kinematic reconciliation "
        "could not read transport velocity '" + record.name +
        "' from the solution vector.");
  }
  const auto begin =
      fe_solution.begin() + static_cast<std::ptrdiff_t>(raw_offset);
  return {begin, begin + static_cast<std::ptrdiff_t>(count)};
}

LevelSetKinematicBaseline makeLevelSetKinematicBaseline(
    const svmp::FE::systems::FESystem& system,
    const LevelSetMaintenanceConfiguration& configuration,
    std::span<const svmp::FE::Real> fe_solution,
    int step)
{
  const auto& transport = configuration.transport;
  const bool with_velocity = transport.kinematic_reconciliation.enabled;
  if (transport.conservative_phase.enabled ||
      (with_velocity &&
       (transport.velocity.source ==
            svmp::FE::level_set::LevelSetVelocitySource::ConstantVector ||
        transport.velocity.source ==
            svmp::FE::level_set::LevelSetVelocitySource::
                MaterialInterfacePhasePair))) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set kinematic reconciliation "
        "for field '" + transport.level_set.field_name +
        "' requires a nodal transport-velocity field and no conservative "
        "phase transport.");
  }
  LevelSetKinematicBaseline baseline;
  baseline.level_set_field =
      system.findFieldByName(transport.level_set.field_name);
  baseline.velocity_field =
      with_velocity ? system.findFieldByName(transport.velocity.field_name)
                    : svmp::FE::INVALID_FIELD_ID;
  if (baseline.level_set_field == svmp::FE::INVALID_FIELD_ID ||
      (with_velocity &&
       baseline.velocity_field == svmp::FE::INVALID_FIELD_ID)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set kinematic reconciliation "
        "could not find field '" + transport.level_set.field_name +
        "' or its transport velocity '" + transport.velocity.field_name +
        "'.");
  }
  const auto offset = static_cast<std::size_t>(
      system.fieldDofOffset(baseline.level_set_field));
  const auto count = static_cast<std::size_t>(
      system.fieldDofHandler(baseline.level_set_field).getNumDofs());
  if (offset > fe_solution.size() || count > fe_solution.size() - offset) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set kinematic reconciliation "
        "field slice exceeds the solution vector.");
  }
  const auto begin = fe_solution.begin() + static_cast<std::ptrdiff_t>(offset);
  baseline.level_set.assign(begin, begin + static_cast<std::ptrdiff_t>(count));
  if (with_velocity) {
    baseline.velocity = levelSetTransportVelocityCoefficients(
        system, baseline.velocity_field, fe_solution);
  }
  baseline.step = step;
  return baseline;
}

LevelSetRedistancingTrigger resolveLevelSetRedistancingTrigger()
{
  const char* text = std::getenv("SVMP_REDISTANCING_TRIGGER_CELLS");
  if (text == nullptr || text[0] == '\0') {
    return LevelSetRedistancingTrigger{.interface_motion_cells = 1.0};
  }
  const double value = std::strtod(text, nullptr);
  if (!(value > 0.0) || !std::isfinite(value)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] SVMP_REDISTANCING_TRIGGER_CELLS must be positive.");
  }
  return LevelSetRedistancingTrigger{.interface_motion_cells = value};
}

namespace {

// Stage 1, kinematic reconciliation of the transport step.
void reconcileMaintainedLevelSetStep(
    const LevelSetMaintenanceHost& host,
    svmp::FE::timestepping::TimeHistory& history,
    const LevelSetMaintenanceRequest& request,
    svmp::FE::FieldId field,
    int completed_step,
    std::span<const LevelSetKinematicBaseline> kinematic_baselines,
    const LevelSetMaintenanceStageObserver& observe_stage,
    std::vector<svmp::FE::Real>& fe_solution,
    bool& changed,
    std::set<svmp::FE::FieldId>& modified_level_set_fields,
    std::vector<std::string>& staged_commit_logs)
{
  // Reconcile the transport step before any representation change: the
  // baseline is the accepted state the transport started from.
  const auto baseline = std::find_if(
      kinematic_baselines.begin(),
      kinematic_baselines.end(),
      [field](const auto& candidate) {
        return candidate.level_set_field == field;
      });
  if (baseline == kinematic_baselines.end() ||
      baseline->step != completed_step - 1) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set kinematic reconciliation "
        "for field '" +
        request.configuration->transport.level_set.field_name +
        "' has no accepted baseline from the previous step.");
  }
  const auto before_reconciliation = fe_solution;
  const auto offset =
      static_cast<std::size_t>(host.fe_system->fieldDofOffset(field));
  const auto count = static_cast<std::size_t>(
      host.fe_system->fieldDofHandler(field).getNumDofs());
  if (offset > fe_solution.size() || count > fe_solution.size() - offset) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set kinematic reconciliation "
        "field slice exceeds the solution vector.");
  }
  const std::span<const svmp::FE::Real> transported(
      fe_solution.data() + offset, count);
  const auto transported_velocity =
      levelSetTransportVelocityCoefficients(
          *host.fe_system, baseline->velocity_field, fe_solution);
  const auto volume_options = levelSetVolumeOptionsForMaintenance(*request.configuration);
  std::vector<svmp::FE::Real> reconciled;
  const auto result =
      svmp::FE::level_set::reconcileLevelSetWithKinematicFlux(
          host.fe_system->meshAccess(),
          host.fe_system->fieldDofHandler(field),
          host.fe_system->fieldDofHandler(baseline->velocity_field),
          volume_options.isovalue,
          volume_options.tolerance,
          static_cast<svmp::FE::Real>(history.dt()),
          baseline->level_set,
          transported,
          baseline->velocity,
          transported_velocity,
          reconciled);
  if (!result.success) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set kinematic reconciliation "
        "failed for field '" +
        request.configuration->transport.level_set.field_name + "': " +
        result.diagnostic);
  }
  if (result.applied) {
    std::copy(reconciled.begin(), reconciled.end(),
              fe_solution.begin() + static_cast<std::ptrdiff_t>(offset));
    changed = true;
    modified_level_set_fields.insert(field);
    if (observe_stage) {
      observe_stage(
          MaintenanceSubstage::
              GeometryReconciliation,
          before_reconciliation,
          fe_solution);
    }
  }
  std::ostringstream log;
  log << std::setprecision(17)
      << "[svMultiPhysics::Application] Level-set kinematic reconciliation"
      << " field='" << request.configuration->transport.level_set.field_name
      << "' step=" << completed_step
      << " dt=" << history.dt()
      << " applied=" << (result.applied ? "true" : "false")
      << " converged=" << (result.converged ? "true" : "false")
      << " iterations=" << result.iterations
      << " previous_interface_cells=" << result.previous_interface_cells
      << " current_interface_cells=" << result.current_interface_cells
      << " corrected_dofs=" << result.corrected_dofs
      << " sign_preserving_redistributed_dofs="
      << result.sign_preserving_redistributed_dofs
      << " sign_preserving_skipped_dofs="
      << result.sign_preserving_skipped_dofs
      << " degeneracy_frozen_dofs=" << result.degeneracy_frozen_dofs
      << " previous_negative_volume=" << result.previous_negative_volume
      << " transported_negative_volume="
      << result.transported_negative_volume
      << " reconciled_negative_volume="
      << result.reconciled_negative_volume
      << " previous_interface_flux=" << result.previous_interface_flux
      << " current_interface_flux=" << result.current_interface_flux
      << " kinematic_volume_change=" << result.kinematic_volume_change
      << " transported_volume_error=" << result.transported_volume_error
      << " reconciled_volume_error=" << result.reconciled_volume_error
      << " max_abs_correction=" << result.max_abs_correction
      << " status='" << result.diagnostic << "'";
  staged_commit_logs.push_back(log.str());
}

// Stage 2, sign-definite patch bounds.
void boundMaintainedLevelSetPatches(
    const LevelSetMaintenanceHost& host,
    const LevelSetMaintenanceRequest& request,
    svmp::FE::FieldId field,
    int completed_step,
    std::span<const LevelSetKinematicBaseline> kinematic_baselines,
    const LevelSetMaintenanceStageObserver& observe_stage,
    std::vector<svmp::FE::Real>& fe_solution,
    bool& changed,
    std::set<svmp::FE::FieldId>& modified_level_set_fields,
    std::vector<std::string>& staged_commit_logs)
{
  // Restore the one-ring maximum principle of exact transport on nodes
  // whose patch lies in one phase.  It changes no cut cell, so it does
  // not interact with the reconciliation above, which only moves nodes
  // of cut cells.
  const auto baseline = std::find_if(
      kinematic_baselines.begin(),
      kinematic_baselines.end(),
      [field](const auto& candidate) {
        return candidate.level_set_field == field;
      });
  if (baseline == kinematic_baselines.end() ||
      baseline->step != completed_step - 1) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set sign-definite patch "
        "bounds for field '" +
        request.configuration->transport.level_set.field_name +
        "' have no accepted baseline from the previous step.");
  }
  // Where characteristics enter through an inflow boundary the value of
  // a boundary node comes from outside its patch.
  for (const auto& open : request.configuration->open_boundaries) {
    if (open.inflow) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set sign-definite patch "
          "bounds for field '" +
          request.configuration->transport.level_set.field_name +
          "' require impermeable or outflow boundaries; face '" +
          open.face_name + "' is a level-set inflow boundary.");
    }
  }
  const auto before_bounds = fe_solution;
  const auto offset =
      static_cast<std::size_t>(host.fe_system->fieldDofOffset(field));
  const auto count = static_cast<std::size_t>(
      host.fe_system->fieldDofHandler(field).getNumDofs());
  if (offset > fe_solution.size() || count > fe_solution.size() - offset) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set sign-definite patch "
        "bounds field slice exceeds the solution vector.");
  }
  const std::span<const svmp::FE::Real> candidate(
      fe_solution.data() + offset, count);
  const auto volume_options = levelSetVolumeOptionsForMaintenance(*request.configuration);
  std::vector<svmp::FE::Real> bounded;
  const auto result =
      svmp::FE::level_set::boundLevelSetOnSignDefinitePatches(
          host.fe_system->meshAccess(),
          host.fe_system->fieldDofHandler(field),
          volume_options.isovalue,
          volume_options.tolerance,
          baseline->level_set,
          candidate,
          bounded);
  if (!result.success) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set sign-definite patch "
        "bounds failed for field '" +
        request.configuration->transport.level_set.field_name + "': " +
        result.diagnostic);
  }
  if (result.applied) {
    std::copy(bounded.begin(), bounded.end(),
              fe_solution.begin() + static_cast<std::ptrdiff_t>(offset));
    changed = true;
    modified_level_set_fields.insert(field);
    if (observe_stage) {
      observe_stage(
          MaintenanceSubstage::Limiting,
          before_bounds,
          fe_solution);
    }
  }
  std::ostringstream log;
  log << std::setprecision(17)
      << "[svMultiPhysics::Application] Level-set sign-definite patch bounds"
      << " field='" << request.configuration->transport.level_set.field_name
      << "' step=" << completed_step
      << " applied=" << (result.applied ? "true" : "false")
      << " sign_definite_dofs=" << result.sign_definite_dofs
      << " bounded_dofs=" << result.bounded_dofs
      << " sign_changes_prevented=" << result.sign_changes_prevented
      << " max_abs_correction=" << result.max_abs_correction
      << " status='" << result.diagnostic << "'";
  staged_commit_logs.push_back(log.str());
}

// Stage 3, interface-preserving redistancing when its trigger fires; it
// advances the request's trigger state.
void redistanceMaintainedLevelSet(
    const LevelSetMaintenanceHost& host,
    svmp::FE::timestepping::TimeHistory& history,
    LevelSetMaintenanceRequest& request,
    svmp::FE::FieldId field,
    int completed_step,
    const LevelSetMaintenanceStageObserver& observe_stage,
    std::vector<svmp::FE::Real>& fe_solution,
    bool& changed,
    std::set<svmp::FE::FieldId>& modified_level_set_fields,
    std::vector<std::string>& staged_commit_logs)
{
  // Interface-preserving redistancing (opt-in): after the transport has
  // been reconciled and bounded, replace the level set by the signed
  // distance to its own zero polygon (polygon ends on the boundary are
  // continued by rays, which keeps the discrete contact angle), then
  // restore the lumped interface position node by node with the
  // kinematic reconciliation at zero velocity.  It runs after the patch
  // bounds because they clamp sign-definite nodes into the previous
  // (distorted) patch ranges and would undo the new values.
  const auto offset =
      static_cast<std::size_t>(host.fe_system->fieldDofOffset(field));
  const auto count = static_cast<std::size_t>(
      host.fe_system->fieldDofHandler(field).getNumDofs());
  if (offset > fe_solution.size() || count > fe_solution.size() - offset) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set interface-preserving "
        "redistancing field slice exceeds the solution vector.");
  }
  const auto velocity_field = host.fe_system->findFieldByName(
      request.configuration->transport.velocity.field_name);
  if (request.configuration->transport.velocity.source ==
          svmp::FE::level_set::LevelSetVelocitySource::ConstantVector ||
      velocity_field == svmp::FE::INVALID_FIELD_ID) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set interface-preserving "
        "redistancing for field '" +
        request.configuration->transport.level_set.field_name +
        "' requires a nodal transport-velocity field (its layout carries "
        "the zero-velocity interface restoration).");
  }
  const auto before_redistancing = fe_solution;
  const std::vector<svmp::FE::Real> input(
      fe_solution.begin() + static_cast<std::ptrdiff_t>(offset),
      fe_solution.begin() + static_cast<std::ptrdiff_t>(offset + count));
  const auto volume_options = levelSetVolumeOptionsForMaintenance(*request.configuration);
  const auto& mesh_access = host.fe_system->meshAccess();
  const auto& phi_dofs = host.fe_system->fieldDofHandler(field);
  // Displacement trigger: redistance once the interface may have moved
  // one cell since the last redistancing (accumulated max |w| dt over
  // the vertices of cut cells against the smallest cut-cell edge).  The
  // distortion that the redistancing removes builds up with interface
  // motion, so the call count does not depend on the time step, and an
  // interface at rest is left alone.  The one-cell threshold is fixed
  // for every case; SVMP_REDISTANCING_TRIGGER_CELLS (diagnostic only)
  // changes it to test the sensitivity.
  const auto transport_velocity = levelSetTransportVelocityCoefficients(
      *host.fe_system, velocity_field, fe_solution);
  // Tetra4 meshes use the zero-surface variants (exact distance to the
  // P1 zero surface, continued across the boundary at the open edges);
  // Triangle3 meshes the zero-polygon ones.
  const bool three_dimensional = mesh_access.dimension() == 3;
  const auto motion =
      three_dimensional
          ? svmp::FE::level_set::levelSetInterfaceMotionScale3D(
                mesh_access, phi_dofs, host.fe_system->fieldDofHandler(velocity_field),
                volume_options.isovalue, input, transport_velocity)
          : svmp::FE::level_set::levelSetInterfaceMotionScale(
                mesh_access, phi_dofs, host.fe_system->fieldDofHandler(velocity_field),
                volume_options.isovalue, input, transport_velocity);
  if (!motion.success) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set interface-preserving "
        "redistancing could not measure the interface motion of field '" +
        request.configuration->transport.level_set.field_name + "': " +
        motion.diagnostic);
  }
  static const double trigger_cells =
      resolveLevelSetRedistancingTrigger().interface_motion_cells;
  auto& accumulated = request.runtime.redistancing_accumulated_displacement;
  accumulated += motion.max_speed * static_cast<svmp::FE::Real>(history.dt());
  const svmp::FE::Real threshold =
      static_cast<svmp::FE::Real>(trigger_cells) * motion.min_edge;
  const bool trigger_fired = motion.min_edge > svmp::FE::Real{0.0} &&
                             accumulated >= threshold;
  if (trigger_fired) {
    std::vector<svmp::FE::Real> redistanced;
    const auto redistance =
        three_dimensional
            ? svmp::FE::level_set::redistanceLevelSetToZeroSurface(
                  mesh_access, phi_dofs, volume_options.isovalue, input,
                  redistanced)
            : svmp::FE::level_set::redistanceLevelSetToZeroPolygon(
                  mesh_access, phi_dofs, volume_options.isovalue, input,
                  redistanced);
    if (!redistance.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set interface-preserving "
          "redistancing failed for field '" +
          request.configuration->transport.level_set.field_name + "': " +
          redistance.diagnostic);
    }
    const auto change_raw =
        three_dimensional
            ? svmp::FE::level_set::measureLevelSetZeroSurfaceChange(
                  mesh_access, phi_dofs, volume_options.isovalue, input,
                  redistanced)
            : svmp::FE::level_set::measureLevelSetZeroPolygonChange(
                  mesh_access, phi_dofs, volume_options.isovalue, input,
                  redistanced);
    std::vector<svmp::FE::Real> restored = redistanced;
    svmp::FE::level_set::LevelSetKinematicReconciliationResult restoration{};
    std::size_t restoration_passes = 0u;
    std::size_t reverted_dofs = 0u;
    if (redistance.applied) {
      auto zero_velocity = levelSetTransportVelocityCoefficients(
          *host.fe_system, velocity_field, fe_solution);
      std::fill(zero_velocity.begin(), zero_velocity.end(),
                svmp::FE::Real{0.0});
      // A DOF that the restoration must freeze (its correction would change
      // the degeneracy class of a cut) would keep an uncorrected share of
      // the interface motion.  Such DOFs keep their input value instead and
      // the restoration is repeated; every pass reverts at least one more
      // DOF, so the loop ends.
      for (;;) {
        ++restoration_passes;
        restoration = svmp::FE::level_set::reconcileLevelSetWithKinematicFlux(
            mesh_access,
            phi_dofs,
            host.fe_system->fieldDofHandler(velocity_field),
            volume_options.isovalue,
            volume_options.tolerance,
            static_cast<svmp::FE::Real>(history.dt()),
            input,
            redistanced,
            zero_velocity,
            zero_velocity,
            restored);
        if (!restoration.success) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set interface-preserving "
              "redistancing could not restore the interface of field '" +
              request.configuration->transport.level_set.field_name + "': " +
              restoration.diagnostic);
        }
        if (!restoration.applied) {
          restored = redistanced;
        }
        bool reverted = false;
        // Revert the whole cells around a frozen DOF: their other DOFs
        // would otherwise keep raw redistanced values whose interface
        // motion the restoration can no longer take back.
        const auto revert_set =
            restoration.degeneracy_frozen_dof_indices.empty()
                ? std::vector<std::size_t>{}
                : three_dimensional
                      ? svmp::FE::level_set::levelSetDofCellClosure3D(
                            mesh_access, phi_dofs,
                            restoration.degeneracy_frozen_dof_indices)
                      : svmp::FE::level_set::levelSetDofCellClosure(
                            mesh_access, phi_dofs,
                            restoration.degeneracy_frozen_dof_indices);
        for (const auto i : revert_set) {
          if (i < redistanced.size() && redistanced[i] != input[i]) {
            redistanced[i] = input[i];
            ++reverted_dofs;
            reverted = true;
          }
        }
        if (!reverted) {
          break;
        }
      }
    }
    const auto change =
        three_dimensional
            ? svmp::FE::level_set::measureLevelSetZeroSurfaceChange(
                  mesh_access, phi_dofs, volume_options.isovalue, input, restored)
            : svmp::FE::level_set::measureLevelSetZeroPolygonChange(
                  mesh_access, phi_dofs, volume_options.isovalue, input, restored);
    const auto volume_before = svmp::FE::level_set::computeLevelSetCutCellVolume(
        mesh_access, phi_dofs, volume_options, input);
    const auto volume_after = svmp::FE::level_set::computeLevelSetCutCellVolume(
        mesh_access, phi_dofs, volume_options, restored);
    if (!change.success || !change_raw.success || !volume_before.success ||
        !volume_after.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set interface-preserving "
          "redistancing could not measure its interface change for field '" +
          request.configuration->transport.level_set.field_name + "': " +
          change.diagnostic + " " + change_raw.diagnostic);
    }
    const bool applied = redistance.applied && restored != input;
    if (applied) {
      std::copy(restored.begin(), restored.end(),
                fe_solution.begin() + static_cast<std::ptrdiff_t>(offset));
      changed = true;
      modified_level_set_fields.insert(field);
      if (observe_stage) {
        observe_stage(
            MaintenanceSubstage::
                Reinitialization,
            before_redistancing,
            fe_solution);
      }
    }
    std::ostringstream log;
    log << std::setprecision(17)
        << "[svMultiPhysics::Application] Level-set interface-preserving redistancing"
        << " field='" << request.configuration->transport.level_set.field_name
        << "' step=" << completed_step
        << " applied=" << (applied ? "true" : "false")
        << " segments=" << redistance.interface_segments
        << " open_ends=" << redistance.open_ends
        << " redistanced_dofs=" << redistance.redistanced_dofs
        << " max_abs_change=" << redistance.max_abs_change
        << " raw_max_crossing_shift=" << change_raw.max_crossing_shift
        << " raw_max_open_end_shift=" << change_raw.max_open_end_shift
        << " restoration_converged="
        << (restoration.converged ? "true" : "false")
        << " restoration_iterations=" << restoration.iterations
        << " restoration_corrected_dofs=" << restoration.corrected_dofs
        << " restoration_frozen_dofs=" << restoration.degeneracy_frozen_dofs
        << " restoration_skipped_dofs="
        << restoration.sign_preserving_skipped_dofs
        << " restoration_passes=" << restoration_passes
        << " reverted_dofs=" << reverted_dofs
        << " max_crossing_shift=" << change.max_crossing_shift
        << " max_open_end_shift=" << change.max_open_end_shift
        << " cells_with_changed_cut=" << change.cells_with_changed_cut
        << " open_ends_before=" << change.open_ends_first
        << " open_ends_after=" << change.open_ends_second
        << " negative_volume_before=" << volume_before.negative_volume
        << " negative_volume_after=" << volume_after.negative_volume
        << " trigger_call=" << (request.runtime.redistancing_calls + 1)
        << " accumulated_displacement_over_h=" << (accumulated / motion.min_edge)
        << " trigger_cells=" << trigger_cells
        << " time=" << history.time()
        << " status='" << redistance.diagnostic << "'";
    staged_commit_logs.push_back(log.str());
    accumulated = svmp::FE::Real{0.0};
    ++request.runtime.redistancing_calls;
  }
}

// Stage 4, publication: one common delta of every maintained field on
// every stored state of the accepted history.
void publishMaintainedLevelSetChanges(
    const LevelSetMaintenanceHost& host,
    svmp::FE::timestepping::TimeHistory& history,
    const std::vector<svmp::FE::Real>& fe_solution,
    const std::vector<svmp::FE::Real>& accepted_solution_before_maintenance,
    const std::set<svmp::FE::FieldId>& modified_level_set_fields,
    std::vector<std::string>& staged_commit_logs,
    bool* rate_publication_started)
{
  // Publish each representation change through one explicit field mapping:
  // a common delta on every stored state, with the rate unchanged.
  const auto comm = activeFESystemCommunicator(*host.fe_system);
  std::optional<StagedPostacceptMaintenanceHistoryPublication>
      staged_history_publication;
  std::size_t synchronized_dofs = 0u;
  const auto local_history_depth = history.historyDepth();
  if (globalAnyBool(local_history_depth <= 0, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Postaccept maintenance publication "
        "requires a positive history depth on every FE communicator rank.");
  }
  const auto [minimum_history_depth, maximum_history_depth] =
      globalMinMaxUint64(
          static_cast<std::uint64_t>(local_history_depth), comm);
  if (minimum_history_depth != maximum_history_depth) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Postaccept maintenance history "
        "depth differs across the FE communicator.");
  }
  std::vector<svmp::FE::backends::GenericVector*> publication_targets;
  std::vector<std::vector<svmp::FE::Real>> synchronized_history;
  std::exception_ptr local_staging_failure;
  try {
    publication_targets.resize(
        static_cast<std::size_t>(local_history_depth) + 1u);
    publication_targets.front() = &history.u();
    synchronized_history.resize(
        static_cast<std::size_t>(local_history_depth));
    for (int k = 1; k <= local_history_depth; ++k) {
      publication_targets[static_cast<std::size_t>(k)] =
          &history.uPrevK(k);
    }
  } catch (...) {
    local_staging_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_staging_failure,
      comm,
      "postaccept_maintenance_history_capture_schedule");

  std::vector<svmp::FE::Real> current_solution =
      capturePostacceptMaintenanceVectorCollectively(
          *publication_targets.front(), comm);
  for (int k = 1; k <= local_history_depth; ++k) {
    auto captured = capturePostacceptMaintenanceVectorCollectively(
        *publication_targets[static_cast<std::size_t>(k)], comm);
    synchronized_history[static_cast<std::size_t>(k - 1)].swap(captured);
  }

  local_staging_failure = nullptr;
  try {
    staged_history_publication.emplace(
        makeStagedPostacceptMaintenanceHistoryPublication(
            std::move(current_solution),
            std::move(synchronized_history)));

    for (const auto field : modified_level_set_fields) {
      const auto raw_offset = host.fe_system->fieldDofOffset(field);
      const auto raw_count =
          host.fe_system->fieldDofHandler(field).getNumDofs();
      if (raw_offset < 0 || raw_count <= 0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set maintenance found an "
            "invalid maintained-field history layout.");
      }
      const auto offset = static_cast<std::size_t>(raw_offset);
      const auto count = static_cast<std::size_t>(raw_count);
      if (offset > fe_solution.size() ||
          count > fe_solution.size() - offset ||
          accepted_solution_before_maintenance.size() !=
              fe_solution.size()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set maintenance field "
            "slice exceeds the FE history vector.");
      }
      std::vector<svmp::FE::Real> representation_delta(count);
      for (std::size_t i = 0u; i < count; ++i) {
        const auto index = offset + i;
        representation_delta[i] =
            fe_solution[index] -
            accepted_solution_before_maintenance[index];
      }
      stageMaintainedFieldPostAcceptPublication(
          makeCommonRepresentationMaintainedFieldPublication(
              field, offset, representation_delta),
          *staged_history_publication);
      synchronized_dofs += count;
    }
  } catch (...) {
    local_staging_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_staging_failure,
      comm,
      "postaccept_maintenance_history_publication_staging");
  if (!staged_history_publication.has_value()) {
    throw std::logic_error(
        "[svMultiPhysics::Application] Postaccept maintenance history "
        "publication staging remained unavailable after collective "
        "preparation.");
  }
  const auto current_previous_delta = globalMaxAbsDifference(
      std::span<const svmp::FE::Real>(
          staged_history_publication->current.data(),
          staged_history_publication->current.size()),
      std::span<const svmp::FE::Real>(
          staged_history_publication->history.front().data(),
          staged_history_publication->history.front().size()),
      comm);
  std::ostringstream log;
  log
      << "[svMultiPhysics::Application] Level-set maintenance synchronized"
      << " step=" << history.stepIndex()
      << " representation_delta=all_history_levels"
      << " temporal_increments=preserved"
      << " rate_state=unchanged_consistently"
      << " rate_synchronized_fields=0"
      << " synchronized_fields=" << modified_level_set_fields.size()
      << " synchronized_dofs=" << synchronized_dofs
      << " current_previous_max_abs_delta=" << current_previous_delta
      << " history_depth=" << history.historyDepth();
  staged_commit_logs.push_back(log.str());

  // Every size, field-slice, collective, and diagnostic invariant has now
  // passed.  Publish the staged coefficient arrays only after those checks
  // so a rejected candidate leaves every accepted history vector untouched.
  publishStagedPostacceptMaintenanceHistoryCollectively(
      history,
      *staged_history_publication,
      comm,
      rate_publication_started);
  // uDot and uDDot are not staged, so they retain both their values and
  // their backend content revisions.
}

} // namespace

bool applyLevelSetMaintenance(
    const LevelSetMaintenanceHost& host,
    svmp::FE::timestepping::TimeHistory& history,
    std::vector<LevelSetMaintenanceRequest>& requests,
    const LevelSetMaintenanceCandidateValidator& validate_candidate,
    const LevelSetMaintenanceStageObserver& observe_stage,
    std::vector<std::string>* deferred_commit_logs,
    bool* rate_publication_started,
    std::span<const LevelSetKinematicBaseline> kinematic_baselines)
{
  if (rate_publication_started != nullptr) {
    *rate_publication_started = false;
  }
  if (!host.fe_system || requests.empty()) {
    return false;
  }

  // Treat maintenance across every configured level-set field as one
  // transaction.  A later request can fail after an earlier correction has
  // been computed, but no accounting/target state may advance unless every
  // request succeeds and the common FE history update is complete.
  auto staged_requests = requests;
  std::vector<std::string> staged_commit_logs;
  bool changed = false;
  const auto comm = activeFESystemCommunicator(*host.fe_system);
  auto fe_solution = gatherFeOrderedSolution(history.u(), comm);
  const auto accepted_solution_before_maintenance = fe_solution;
  std::set<svmp::FE::FieldId> modified_level_set_fields;
  for (auto& request : staged_requests) {
    // Conservative requests are maintained inside their prospective step
    // transaction.  Maintaining them again here after acceptance would expose
    // an intermediate geometry and apply a second representation change
    // outside the phase-mass reconciliation contract.
    if (request.configuration->transport.conservative_phase.enabled) {
      continue;
    }
    const int completed_step = history.stepIndex();
    const bool do_kinematic =
        svmp::FE::level_set::shouldApplyLevelSetKinematicReconciliation(
            request.configuration->transport.kinematic_reconciliation,
            completed_step);
    const bool do_patch_bounds =
        svmp::FE::level_set::shouldApplyLevelSetSignDefinitePatchBounds(
            request.configuration->transport.sign_definite_patch_bounds,
            completed_step);
    const bool do_redistancing =
        svmp::FE::level_set::shouldApplyLevelSetInterfacePreservingRedistancing(
            request.configuration->transport.interface_preserving_redistancing,
            completed_step);
    if (!do_kinematic && !do_patch_bounds && !do_redistancing) {
      continue;
    }

    const auto field =
        host.fe_system->findFieldByName(request.configuration->transport.level_set.field_name);
    if (globalMinDouble(
            field != svmp::FE::INVALID_FIELD_ID ? 1.0 : 0.0,
            activeFESystemCommunicator(*host.fe_system)) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set maintenance could not find field '" +
          request.configuration->transport.level_set.field_name + "'.");
    }

    if (do_kinematic) {
      reconcileMaintainedLevelSetStep(
          host, history, request, field, completed_step, kinematic_baselines,
          observe_stage, fe_solution, changed, modified_level_set_fields,
          staged_commit_logs);
    }

    if (do_patch_bounds) {
      boundMaintainedLevelSetPatches(
          host, request, field, completed_step, kinematic_baselines,
          observe_stage, fe_solution, changed, modified_level_set_fields,
          staged_commit_logs);
    }

    if (do_redistancing) {
      redistanceMaintainedLevelSet(
          host, history, request, field, completed_step, observe_stage,
          fe_solution, changed, modified_level_set_fields,
          staged_commit_logs);
    }
  }

  if (changed && validate_candidate) {
    validate_candidate(fe_solution);
  }

  if (changed) {
    publishMaintainedLevelSetChanges(
        host,
        history,
        fe_solution,
        accepted_solution_before_maintenance,
        modified_level_set_fields,
        staged_commit_logs,
        rate_publication_started);
  }
  requests = std::move(staged_requests);
  if (deferred_commit_logs != nullptr) {
    *deferred_commit_logs = std::move(staged_commit_logs);
  } else {
    for (const auto& log : staged_commit_logs) {
      svmp::FE::rank_log::oopCout() << log << std::endl;
    }
  }
  return changed;
}

} // namespace level_set
} // namespace FE
} // namespace svmp

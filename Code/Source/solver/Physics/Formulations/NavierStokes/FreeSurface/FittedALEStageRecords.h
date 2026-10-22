/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

/**
 * @file
 * @brief The fitted-ALE normal operator-stage measurement records of an FE
 *        system: the registered measurements, the pending group of a
 *        candidate and the accepted history.
 *
 * The records are an attachment of the FE system (FESystem::attachment), so
 * they live as long as the system.  fittedALERecords(system) is the
 * interface that FESystem had for them.  They observe the mesh-normal
 * constraints and tangential policies of the FE system and freeze its
 * mesh-normal declarations at their first collective stage
 * (FESystem::freezeMeshNormalBoundaryConstraintDeclarations); their
 * reductions are boundary functionals of the system's
 * BoundaryReductionService.  The record data types are in FESystem.h.
 */

#include "FE/Core/Types.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/OperatorStageMetadata.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/// The records of one FE system (its attachment).
struct FittedALEStageRecordStore {
  std::vector<svmp::FE::systems::FittedALENormalMeasurementDeclaration> declarations{};
  std::vector<svmp::FE::systems::FittedALENormalOperatorStageHistoryRecord> pending{};
  std::vector<svmp::FE::systems::FittedALENormalOperatorStageHistoryRecord> history{};
  bool declarations_frozen{false};
  bool transaction_active{false};
};

/**
 * @brief The fitted-ALE normal operator-stage measurement records of an FE
 * system.
 *
 * A lightweight handle on the system and its record store; it reads the
 * mesh-boundary declarations of the system when it is used.
 */
class FittedALEStageRecords {
public:
  explicit FittedALEStageRecords(svmp::FE::systems::FESystem& system);

  /**
   * @brief Register raw operator-stage moments for one bound fitted-ALE
   * fluid-normal constraint.
   *
   * Registration is deterministic and an exact duplicate is a no-op.  It
   * must occur after the constraint's reciprocal consumer binding is
   * complete and before a measurement stage or accepted history exists.
   */
  void registerFittedALENormalOperatorStageMeasurement(
      svmp::FE::FieldId mesh_displacement_field,
      int boundary_marker,
      std::optional<std::array<svmp::FE::Real, 3>>
          prescribed_tangential_mesh_velocity = std::nullopt);
  [[nodiscard]] std::span<
      const svmp::FE::systems::FittedALENormalMeasurementDeclaration>
  fittedALENormalOperatorStageMeasurementDeclarations() const noexcept {
    return fitted_ale_normal_measurement_declarations_;
  }
  /**
   * @brief Evaluate every registered fitted-ALE normal declaration at the
   * supplied converged operator stage.
   *
   * `state.u`/`state.u_vector` supplies the current stage and
   * `state.u_prev`/`state.u_prev_vector` supplies an exact first-rate alias.
   * Only fields named by `metadata.derivative_fields` are advertised as
   * derivatives.  Results remain pending until explicit accepted-step
   * publication.
   */
  void stageFittedALENormalOperatorStageMeasurements(
      const svmp::FE::systems::SystemStateView& state,
      svmp::FE::systems::OperatorStageMeasurementMetadata metadata);
  /** Discard a rejected candidate without throwing or publishing history. */
  void discardPendingFittedALENormalOperatorStageMeasurements() noexcept;
  /**
   * @brief Collectively publish the pending group after irreversible step
   * acceptance.
   *
   * The accepted step/time/dt must exactly match the prospective metadata
   * captured at staging.  This API is intended for the accepted-step hook,
   * not a commit-ready hook.
   */
  void commitPendingFittedALENormalOperatorStageMeasurements(
      std::uint64_t accepted_step,
      svmp::FE::Real accepted_time,
      svmp::FE::Real dt);
  [[nodiscard]] std::span<
      const svmp::FE::systems::FittedALENormalOperatorStageHistoryRecord>
  pendingFittedALENormalOperatorStageMeasurements() const noexcept {
    return pending_fitted_ale_normal_measurements_;
  }
  [[nodiscard]] std::span<
      const svmp::FE::systems::FittedALENormalOperatorStageHistoryRecord>
  fittedALENormalOperatorStageMeasurementHistory() const noexcept {
    return fitted_ale_normal_measurement_history_;
  }

private:
  // The FE system state that the record code reads, under the names it had
  // as FESystem members.
  struct FieldRegistryAccess {
    const svmp::FE::systems::FESystem* system;
    [[nodiscard]] const svmp::FE::systems::FieldRecord& get(svmp::FE::FieldId field) const
    {
      return system->fieldRecord(field);
    }
    [[nodiscard]] bool has(svmp::FE::FieldId field) const noexcept
    {
      return system->hasField(field);
    }
  };

  // A declaration or history sequence of the FE system, read through its
  // accessor at every use.  Pointer iterators, so begin() and end() of
  // separate uses always agree.
  template <class T,
            std::span<const T> (svmp::FE::systems::FESystem::*Accessor)() const noexcept>
  struct SystemSequence {
    const svmp::FE::systems::FESystem* system;
    [[nodiscard]] const T* begin() const noexcept
    {
      return (system->*Accessor)().data();
    }
    [[nodiscard]] const T* end() const noexcept
    {
      const auto sequence = (system->*Accessor)();
      return sequence.data() + sequence.size();
    }
    [[nodiscard]] bool empty() const noexcept
    {
      return (system->*Accessor)().empty();
    }
  };

  [[nodiscard]] bool isSetup() const noexcept { return system_->isSetup(); }
  [[nodiscard]] const svmp::FE::dofs::DofHandler& dofHandler() const noexcept
  {
    return system_->dofHandler();
  }
  [[nodiscard]] const svmp::FE::assembly::IMeshAccess& meshAccess() const
  {
    return system_->meshAccess();
  }
  [[nodiscard]] bool meshBoundaryHistoryTransactionActive() const noexcept
  {
    return system_->meshBoundaryHistoryTransactionActive();
  }
  [[nodiscard]] std::span<const svmp::FE::analysis::BoundaryConditionDescriptor>
  boundaryConditionDescriptors() const noexcept
  {
    return system_->boundaryConditionDescriptors();
  }
  [[nodiscard]] std::span<const std::string>
  boundaryConditionDescriptorOperatorTags() const noexcept
  {
    return system_->boundaryConditionDescriptorOperatorTags();
  }
  [[nodiscard]] const svmp::FE::systems::MeshNormalBoundaryConstraintDeclaration&
  validatedFittedALENormalConstraint(svmp::FE::FieldId mesh_displacement_field,
                                     int boundary_marker) const
  {
    return system_->validatedFittedALENormalConstraint(
        mesh_displacement_field, boundary_marker);
  }
  [[nodiscard]] svmp::FE::systems::BoundaryReductionService& boundaryReductionService(
      svmp::FE::FieldId primary_field)
  {
    return system_->boundaryReductionService(primary_field);
  }
  [[nodiscard]] svmp::FE::systems::BoundaryReductionService*
  boundaryReductionServiceIfPresent(svmp::FE::FieldId primary_field) noexcept
  {
    return system_->boundaryReductionServiceIfPresent(primary_field);
  }
  void bindSecondaryFields(svmp::FE::systems::BoundaryReductionService& svc,
                           svmp::FE::FieldId primary_fid,
                           const std::vector<svmp::FE::FieldId>& referenced_fields)
  {
    system_->bindSecondaryFields(svc, primary_fid, referenced_fields);
  }

  svmp::FE::systems::FESystem* system_;
  std::vector<svmp::FE::systems::FittedALENormalMeasurementDeclaration>&
      fitted_ale_normal_measurement_declarations_;
  std::vector<svmp::FE::systems::FittedALENormalOperatorStageHistoryRecord>&
      pending_fitted_ale_normal_measurements_;
  std::vector<svmp::FE::systems::FittedALENormalOperatorStageHistoryRecord>&
      fitted_ale_normal_measurement_history_;
  bool& fitted_ale_normal_measurement_declarations_frozen_;
  bool& fitted_ale_normal_measurement_transaction_active_;
  FieldRegistryAccess field_registry_;
  SystemSequence<svmp::FE::systems::MeshNormalBoundaryConstraintDeclaration,
                 &svmp::FE::systems::FESystem::meshNormalBoundaryConstraints>
      mesh_normal_boundary_constraints_;
  SystemSequence<svmp::FE::systems::MeshNormalBoundaryConstraintHistoryRecord,
                 &svmp::FE::systems::FESystem::meshNormalBoundaryConstraintHistory>
      mesh_normal_boundary_constraint_history_;
  SystemSequence<svmp::FE::systems::MeshTangentialBoundaryPolicyDeclaration,
                 &svmp::FE::systems::FESystem::meshTangentialBoundaryPolicies>
      mesh_tangential_boundary_policies_;
  SystemSequence<svmp::FE::systems::MeshTangentialBoundaryPolicyHistoryRecord,
                 &svmp::FE::systems::FESystem::meshTangentialBoundaryPolicyHistory>
      mesh_tangential_boundary_policy_history_;
};

/// Read-only fitted-ALE records of an FE system; empty without records.
class ConstFittedALEStageRecords {
public:
  explicit ConstFittedALEStageRecords(
      const svmp::FE::systems::FESystem& system) noexcept;

  [[nodiscard]] std::span<
      const svmp::FE::systems::FittedALENormalMeasurementDeclaration>
  fittedALENormalOperatorStageMeasurementDeclarations() const noexcept;
  [[nodiscard]] std::span<
      const svmp::FE::systems::FittedALENormalOperatorStageHistoryRecord>
  pendingFittedALENormalOperatorStageMeasurements() const noexcept;
  [[nodiscard]] std::span<
      const svmp::FE::systems::FittedALENormalOperatorStageHistoryRecord>
  fittedALENormalOperatorStageMeasurementHistory() const noexcept;

private:
  const FittedALEStageRecordStore* store_{nullptr};
};

/// The fitted-ALE records of @p system (created on first use).
[[nodiscard]] FittedALEStageRecords fittedALERecords(svmp::FE::systems::FESystem& system);

/// The fitted-ALE records of @p system, read-only.
[[nodiscard]] ConstFittedALEStageRecords fittedALERecords(
    const svmp::FE::systems::FESystem& system) noexcept;

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

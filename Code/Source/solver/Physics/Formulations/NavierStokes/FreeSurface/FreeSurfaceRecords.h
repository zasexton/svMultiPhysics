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
 * @brief The free-surface accepted-step records of an FE system: the
 *        discrete-functional, residual-work and exterior-boundary energy
 *        declarations and the accepted functional history.
 *
 * The records are an attachment of the FE system (FESystem::attachment), so
 * they live as long as the system.  records(system) is the interface that
 * FESystem had for them; the record data types are in
 * FE/Systems/FreeSurfaceRecordTypes.h for now.
 */

#include "FE/Assembly/Assembler.h"
#include "FE/Core/Types.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FreeSurfaceRecordTypes.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/// The records of one FE system (its attachment).
struct FreeSurfaceRecordStore {
  std::vector<svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
      discrete_functional_declarations{};
  std::vector<svmp::FE::systems::FreeSurfaceResidualWorkDeclaration>
      residual_work_declarations{};
  std::optional<svmp::FE::systems::FreeSurfaceExternalBoundaryEnergyDeclaration>
      external_boundary_energy_declaration{};
  std::vector<svmp::FE::systems::FreeSurfaceDiscreteFunctionalHistoryRecord>
      discrete_functional_history{};
};

/**
 * @brief The free-surface records of an FE system.
 *
 * A lightweight handle on the system and its record store.  Declarations
 * are made before setup; accepted functional groups are recorded after each
 * accepted step (collective where the FE system is).
 */
class FreeSurfaceRecords {
public:
  explicit FreeSurfaceRecords(svmp::FE::systems::FESystem& system);

  void declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration declaration);
  void declareFreeSurfaceResidualWork(
      svmp::FE::systems::FreeSurfaceResidualWorkDeclaration declaration);
  [[nodiscard]] std::span<
      const svmp::FE::systems::FreeSurfaceResidualWorkDeclaration>
  freeSurfaceResidualWorkDeclarations() const noexcept {
    return free_surface_residual_work_declarations_;
  }
  void declareFreeSurfaceExternalBoundaryEnergy(
      svmp::FE::systems::FreeSurfaceExternalBoundaryEnergyDeclaration declaration);
  [[nodiscard]] const std::optional<
      svmp::FE::systems::FreeSurfaceExternalBoundaryEnergyDeclaration>&
  freeSurfaceExternalBoundaryEnergyDeclaration() const noexcept {
    return free_surface_external_boundary_energy_declaration_;
  }
  [[nodiscard]] std::span<
      const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
  freeSurfaceDiscreteFunctionalDeclarations() const noexcept {
    return free_surface_discrete_functional_declarations_;
  }
  void recordAcceptedFreeSurfaceDiscreteFunctionals(
      std::uint64_t accepted_step,
      svmp::FE::Real accepted_time,
      svmp::FE::Real dt,
      std::uint64_t pre_maintenance_endpoint_state_revision,
      std::uint64_t state_revision,
      std::span<const svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
          states,
      std::optional<std::uint64_t> extension_map_revision =
          std::nullopt);
  [[nodiscard]] std::span<
      const svmp::FE::systems::FreeSurfaceDiscreteFunctionalHistoryRecord>
  freeSurfaceDiscreteFunctionalHistory() const noexcept {
    return free_surface_discrete_functional_history_;
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
  struct MeshAccessPointer {
    const svmp::FE::systems::FESystem* system;
    explicit operator bool() const noexcept { return system->hasMeshAccess(); }
    const svmp::FE::assembly::IMeshAccess* operator->() const
    {
      return &system->meshAccess();
    }
  };

  [[nodiscard]] bool isSetup() const noexcept { return system_->isSetup(); }
  [[nodiscard]] bool hasOperator(const svmp::FE::systems::OperatorTag& op) const noexcept
  {
    return system_->hasOperator(op);
  }
  [[nodiscard]] bool fieldParticipatesInUnknownVector(
      svmp::FE::FieldId field) const
  {
    return system_->fieldParticipatesInUnknownVector(field);
  }
  void invalidateSetup() noexcept { system_->invalidateSetup(); }

  svmp::FE::systems::FESystem* system_;
  std::vector<svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>&
      free_surface_discrete_functional_declarations_;
  std::vector<svmp::FE::systems::FreeSurfaceResidualWorkDeclaration>&
      free_surface_residual_work_declarations_;
  std::optional<svmp::FE::systems::FreeSurfaceExternalBoundaryEnergyDeclaration>&
      free_surface_external_boundary_energy_declaration_;
  std::vector<svmp::FE::systems::FreeSurfaceDiscreteFunctionalHistoryRecord>&
      free_surface_discrete_functional_history_;
  FieldRegistryAccess field_registry_;
  MeshAccessPointer mesh_access_;
};

/// Read-only free-surface records of an FE system; empty without records.
class ConstFreeSurfaceRecords {
public:
  explicit ConstFreeSurfaceRecords(
      const svmp::FE::systems::FESystem& system) noexcept;

  [[nodiscard]] std::span<
      const svmp::FE::systems::FreeSurfaceResidualWorkDeclaration>
  freeSurfaceResidualWorkDeclarations() const noexcept;
  [[nodiscard]] const std::optional<
      svmp::FE::systems::FreeSurfaceExternalBoundaryEnergyDeclaration>&
  freeSurfaceExternalBoundaryEnergyDeclaration() const noexcept;
  [[nodiscard]] std::span<
      const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
  freeSurfaceDiscreteFunctionalDeclarations() const noexcept;
  [[nodiscard]] std::span<
      const svmp::FE::systems::FreeSurfaceDiscreteFunctionalHistoryRecord>
  freeSurfaceDiscreteFunctionalHistory() const noexcept;

private:
  const FreeSurfaceRecordStore* store_{nullptr};
};

/// The free-surface records of @p system (created on first use).
[[nodiscard]] FreeSurfaceRecords records(svmp::FE::systems::FESystem& system);

/// The free-surface records of @p system, read-only.
[[nodiscard]] ConstFreeSurfaceRecords records(
    const svmp::FE::systems::FESystem& system) noexcept;

/// Accepted-stage contact-centroid kinematics from two consecutive history
/// records (see FreeSurfaceAcceptedContactLineKinematics).
[[nodiscard]] std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactLineKinematics>
deriveFreeSurfaceAcceptedContactLineKinematics(
    const svmp::FE::systems::FreeSurfaceDiscreteFunctionalHistoryRecord& previous,
    std::uint64_t accepted_step,
    const svmp::FE::systems::FreeSurfaceAcceptedContactStageState& current);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

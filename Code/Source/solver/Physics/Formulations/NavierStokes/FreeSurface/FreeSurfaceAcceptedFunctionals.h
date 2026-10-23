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
 * @brief Accepted free-surface discrete functionals: evaluation at the
 *        current state, endpoint powers, active-volume energies and
 *        dissipation, exterior-pressure powers, backward-Euler kinetic work,
 *        maintenance functional values and the accepted record.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRunSupport.h"

#include "FE/Core/Types.h"
#include "FE/LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshComm.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

std::uint64_t freeSurfaceDiscreteFunctionalDeclarationContentKey(
    const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration&
        declaration) noexcept;

std::vector<svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
evaluateCurrentFreeSurfaceDiscreteFunctionals(
    const FreeSurfaceSystemHandle& sim);

void attachAcceptedFreeSurfaceActiveVolumeEnergies(
    const FreeSurfaceSystemHandle& sim,
    std::span<const svmp::FE::Real> accepted_solution,
    std::span<
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        accepted_states);

std::uint64_t collectiveFreeSurfaceFieldRevision(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field,
    std::span<const svmp::FE::Real> solution,
    const svmp::MeshComm& comm,
    std::string_view diagnostic_context);

std::vector<svmp::FE::level_set::LevelSetAuthoritativeFunctionalValue>
levelSetMaintenanceFunctionalValues(
    const FreeSurfaceSystemHandle& sim,
    std::span<const svmp::FE::systems::
                        AcceptedFreeSurfaceDiscreteFunctionalState> states,
    std::span<const svmp::FE::Real> solution);

struct FreeSurfaceAcceptanceCoverage {
  bool has_functional{false};
  bool has_dynamic_contact{false};
};

FreeSurfaceAcceptanceCoverage
requireCommunicatorConsistentFreeSurfaceAcceptanceCoverage(
    std::span<
        const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const svmp::MeshComm& comm);

std::vector<
    svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
recordAcceptedFreeSurfaceDiscreteFunctionals(
    const FreeSurfaceSystemHandle& sim,
    std::uint64_t accepted_step,
    svmp::FE::Real accepted_time,
    svmp::FE::Real dt,
    std::uint64_t pre_maintenance_endpoint_state_revision,
    std::uint64_t state_revision,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        contact_stages = {},
    std::span<const svmp::FE::Real> accepted_solution = {},
    std::span<const svmp::FE::Real>
        backward_euler_previous_solution = {},
    std::optional<std::uint64_t>
        accepted_extension_map_revision = std::nullopt,
    bool* backward_euler_kinetic_work_bound = nullptr);

void recordInitialFreeSurfaceDiscreteFunctionalBaseline(
    const FreeSurfaceSystemHandle& sim,
    std::uint64_t initial_step,
    svmp::FE::Real initial_time,
    std::span<const svmp::FE::Real> initial_solution,
    bool record_backward_euler_kinetic_baseline = false,
    std::optional<std::uint64_t>
        initial_extension_map_revision = std::nullopt);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

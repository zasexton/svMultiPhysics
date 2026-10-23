/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_LEVELSET_ACCEPTED_FUNCTIONAL_RECORDER_H
#define SVMP_FE_LEVELSET_ACCEPTED_FUNCTIONAL_RECORDER_H

/**
 * @file AcceptedFunctionalRecorder.h
 * @brief Evaluation and recording of the declared discrete functionals of
 *        an FE system on its authoritative level-set geometry snapshot.
 *
 * The declared functionals (FE/Systems/FunctionalRecords) are evaluated on
 * the snapshot of the system's cut-integration context at an accepted
 * endpoint: the stored functionals, endpoint powers, active-volume energies
 * and dissipation, exterior-pressure powers and backward-Euler kinetic
 * work, with communicator-consistent coverage and revisions.  The accepted
 * group is recorded in the functional records, and the level-set
 * maintenance reads the same values (LevelSetAuthoritativeFunctionalValue).
 *
 * Functions that take an FESystemHandle are collective over the system's
 * active communicator and must be called on every rank in the same order.
 */

#include "Core/Types.h"
#include "LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h"
#include "Systems/FESystem.h"
#include "Systems/FESystemHandle.h"
#include "Systems/FreeSurfaceRecordTypes.h"
#include "Mesh/Core/MeshComm.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace level_set {

/// A content key of one functional declaration (stable across ranks).
std::uint64_t acceptedFunctionalDeclarationContentKey(
    const systems::FreeSurfaceDiscreteFunctionalDeclaration&
        declaration) noexcept;

/// The declared functionals evaluated on the current snapshot.
std::vector<systems::AcceptedFreeSurfaceDiscreteFunctionalState>
evaluateCurrentAcceptedFunctionals(const systems::FESystemHandle& sim);

/// Attach the active-volume energies of @p accepted_solution.
void attachAcceptedActiveVolumeEnergies(
    const systems::FESystemHandle& sim,
    std::span<const Real> accepted_solution,
    std::span<systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        accepted_states);

/// A communicator-consistent revision of one field's values.
std::uint64_t collectiveFieldRevision(
    const systems::FESystem& system,
    FieldId field,
    std::span<const Real> solution,
    const svmp::MeshComm& comm,
    std::string_view diagnostic_context);

/// The functional values that the level-set maintenance ledger reads.
std::vector<LevelSetAuthoritativeFunctionalValue>
levelSetMaintenanceFunctionalValues(
    const systems::FESystemHandle& sim,
    std::span<const systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        states,
    std::span<const Real> solution);

/// Which accepted functionals the declarations require on every rank.
struct AcceptedFunctionalCoverage {
    bool has_functional{false};
    bool has_dynamic_contact{false};
};

/// The coverage of @p declarations, required equal on every rank.
AcceptedFunctionalCoverage
requireCommunicatorConsistentAcceptedFunctionalCoverage(
    std::span<const systems::FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const svmp::MeshComm& comm);

/**
 * @brief Evaluate the declared functionals at an accepted endpoint and
 *        record the group in the functional records.
 *
 * @return the recorded states, in declaration order
 */
std::vector<systems::AcceptedFreeSurfaceDiscreteFunctionalState>
recordAcceptedFunctionals(
    const systems::FESystemHandle& sim,
    std::uint64_t accepted_step,
    Real accepted_time,
    Real dt,
    std::uint64_t pre_maintenance_endpoint_state_revision,
    std::uint64_t state_revision,
    std::span<const systems::FreeSurfaceAcceptedContactStageState>
        contact_stages = {},
    std::span<const Real> accepted_solution = {},
    std::span<const Real> backward_euler_previous_solution = {},
    std::optional<std::uint64_t> accepted_extension_map_revision =
        std::nullopt,
    bool* backward_euler_kinetic_work_bound = nullptr);

/**
 * @brief Record the initial accepted functional group (dt = 0), with the
 *        accepted contact stages of the initial state
 *        (Interfaces/ContactLineStageObservables).
 */
void recordInitialAcceptedFunctionalBaseline(
    const systems::FESystemHandle& sim,
    std::uint64_t initial_step,
    Real initial_time,
    std::span<const Real> initial_solution,
    bool record_backward_euler_kinetic_baseline = false,
    std::optional<std::uint64_t> initial_extension_map_revision =
        std::nullopt);

} // namespace level_set
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_LEVELSET_ACCEPTED_FUNCTIONAL_RECORDER_H

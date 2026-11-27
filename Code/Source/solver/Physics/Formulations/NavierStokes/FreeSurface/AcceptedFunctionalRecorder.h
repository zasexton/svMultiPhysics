/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FREE_SURFACE_ACCEPTED_FUNCTIONAL_RECORDER_H
#define SVMP_PHYSICS_FREE_SURFACE_ACCEPTED_FUNCTIONAL_RECORDER_H

/**
 * @file AcceptedFunctionalRecorder.h
 * @brief Evaluation and recording of the declared discrete functionals of
 *        an FE system on its authoritative level-set geometry snapshot.
 *
 * The declared functionals (FunctionalRecords) are evaluated on
 * the snapshot of the system's cut-integration context at an accepted
 * endpoint: the stored functionals, endpoint powers, active-volume energies
 * and dissipation, exterior-pressure powers and backward-Euler kinetic
 * work, with communicator-consistent coverage and revisions.  Their values
 * are the measured 0-forms the formulation declares (Systems/FunctionalForms,
 * D49 steps E3 and E4), evaluated with an exact reduction.  The accepted
 * group is recorded in the functional records, and the level-set
 * maintenance reads the same values (LevelSetAuthoritativeFunctionalValue).
 *
 * Functions that take an FESystemHandle are collective over the system's
 * active communicator and must be called on every rank in the same order.
 */

#include "FE/Core/Types.h"
#include "FE/Forms/FormExpr.h"
#include "FE/LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FESystemHandle.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRecordTypes.h"
#include "Mesh/Core/MeshComm.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/// A content key of one functional declaration (stable across ranks).
std::uint64_t acceptedFunctionalDeclarationContentKey(
    const FreeSurfaceDiscreteFunctionalDeclaration&
        declaration) noexcept;

/**
 * @brief The accepted-functional states of the current geometry without
 *        their values (collective).
 *
 * One state per declaration, in declaration order: the interface, the
 * geometry revision of its snapshot, the parameters and the walls, with
 * communicator-consistent declarations and snapshots.  Every value is zero;
 * the values are the declared functionals (evaluateDeclaredAcceptedFunctionals,
 * recordAcceptedFunctionals, levelSetMaintenanceFunctionalValues).
 */
std::vector<AcceptedFreeSurfaceDiscreteFunctionalState>
currentAcceptedFunctionalLayouts(const FE::systems::FESystemHandle& sim);

/**
 * @brief The accepted-functional states of the current geometry with the
 *        values of the declared functionals on @p solution (collective).
 *
 * The layouts, the solution channels the declarations name (endpoint power,
 * active-volume energy, dissipation, exterior-pressure power; only with a
 * nonempty @p solution) and the values of the measured 0-forms the
 * formulation declares, with the exact reduction.  Throws when a recorded
 * quantity has no declared functional.
 */
std::vector<AcceptedFreeSurfaceDiscreteFunctionalState>
evaluateDeclaredAcceptedFunctionals(
    const FE::systems::FESystemHandle& sim,
    std::span<const FE::Real> solution);

/**
 * @brief One free-surface functional of an interface as a measured 0-form
 *        (D49 steps E3 and E4).
 *
 * @p quantity is a functional the formulation declares
 * ("free_surface.<interface_marker>.<quantity>", Systems/FunctionalForms.h),
 * or a wall quantity the recorder states on the current snapshot with the
 * coefficients of the discrete-functional declaration:
 * "wall.<b>.wetted_area", "wall.<b>.young_energy" (a wall with an angle) and
 * "wall.<b>.contact_measure" (empty when the snapshot has no domain of that
 * wall), or their sums over the walls, "wetted_wall_area",
 * "young_wall_energy" and "contact_measure" (the zero form when the
 * snapshot has no wall domain).  Empty when the quantity is not declared.
 * Not collective; every rank states the same form.
 */
std::optional<FE::forms::FormExpr> freeSurfaceFunctionalForm(
    const FE::systems::FESystemHandle& sim,
    int interface_marker,
    std::string_view quantity);

/**
 * @brief The values of free-surface functionals of one interface on
 *        @p solution and the current snapshot (collective).
 *
 * Each entry of @p sums is a list of freeSurfaceFunctionalForm quantities
 * whose sum is evaluated as one form, with the exact reduction of
 * Systems/FunctionalFormEvaluator.h, so the value is the same at every rank
 * count.  Empty on every rank when a quantity is not declared on some rank.
 * An empty @p solution suffices for forms that read no field.
 */
std::optional<std::vector<FE::Real>> evaluateDeclaredFreeSurfaceFunctionals(
    const FE::systems::FESystemHandle& sim,
    int interface_marker,
    std::span<const std::vector<std::string_view>> sums,
    std::span<const FE::Real> solution);

/// A communicator-consistent revision of one field's values.
std::uint64_t collectiveFieldRevision(
    const FE::systems::FESystem& system,
    FE::FieldId field,
    std::span<const FE::Real> solution,
    const svmp::MeshComm& comm,
    std::string_view diagnostic_context);

/// The functional values that the level-set maintenance ledger reads.
std::vector<FE::level_set::LevelSetAuthoritativeFunctionalValue>
levelSetMaintenanceFunctionalValues(
    const FE::systems::FESystemHandle& sim,
    std::span<const AcceptedFreeSurfaceDiscreteFunctionalState>
        states,
    std::span<const FE::Real> solution);

/// Which accepted functionals the declarations require on every rank.
struct AcceptedFunctionalCoverage {
    bool has_functional{false};
    bool has_dynamic_contact{false};
};

/// The coverage of @p declarations, required equal on every rank.
AcceptedFunctionalCoverage
requireCommunicatorConsistentAcceptedFunctionalCoverage(
    std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const svmp::MeshComm& comm);

/**
 * @brief Evaluate the declared functionals at an accepted endpoint and
 *        record the group in the functional records.
 *
 * @return the recorded states, in declaration order
 */
std::vector<AcceptedFreeSurfaceDiscreteFunctionalState>
recordAcceptedFunctionals(
    const FE::systems::FESystemHandle& sim,
    std::uint64_t accepted_step,
    FE::Real accepted_time,
    FE::Real dt,
    std::uint64_t pre_maintenance_endpoint_state_revision,
    std::uint64_t state_revision,
    std::span<const FreeSurfaceAcceptedContactStageState>
        contact_stages = {},
    std::span<const FE::Real> accepted_solution = {},
    std::span<const FE::Real> backward_euler_previous_solution = {},
    std::optional<std::uint64_t> accepted_extension_map_revision =
        std::nullopt,
    bool* backward_euler_kinetic_work_bound = nullptr);

/**
 * @brief Record the initial accepted functional group (dt = 0), with the
 *        accepted contact stages of the initial state
 *        (Interfaces/ContactLineStageObservables).
 */
void recordInitialAcceptedFunctionalBaseline(
    const FE::systems::FESystemHandle& sim,
    std::uint64_t initial_step,
    FE::Real initial_time,
    std::span<const FE::Real> initial_solution,
    bool record_backward_euler_kinetic_baseline = false,
    std::optional<std::uint64_t> initial_extension_map_revision =
        std::nullopt);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

#endif // SVMP_PHYSICS_FREE_SURFACE_ACCEPTED_FUNCTIONAL_RECORDER_H

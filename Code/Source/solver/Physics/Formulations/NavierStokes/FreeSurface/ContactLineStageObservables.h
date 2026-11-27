/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file ContactLineStageObservables.h
 * @ingroup fe_interfaces
 * @brief Contact-line observables of an accepted stage: the dynamic contact
 *        state (contact angles, contact-line speeds, wall slip and their
 *        dissipations) of every declared interface with a dynamic contact
 *        law, evaluated on its authoritative geometry snapshot at the stage
 *        solution, reduced over the FE communicator, with the stage's
 *        temporal provenance and revision, and their binding to the accepted
 *        endpoint revision.
 *
 * The functions were moved from Physics/Formulations/NavierStokes/
 * FreeSurface/FreeSurfaceContactStages with bodies, messages and collective
 * order unchanged.  The declarations of the interfaces (contact laws, walls,
 * equilibrium angles) are the formulation's data, passed by the caller.
 * Every function that takes a communicator, or evaluates stages, is
 * collective.
 */

#include "FE/Core/Types.h"
#include "FE/LevelSet/LevelSetWallContactConstraint.h"
#include "FE/Forms/FormExpr.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRecordTypes.h"
#include "FE/Systems/SystemCollectives.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp::FE {
namespace systems {
class FESystem;
} // namespace systems

} // namespace svmp::FE

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/// True for the exact first-order generalized-alpha parameter family of the
/// time loop (alpha_m = 2 alpha_f - 1/2, gamma = alpha_f) with finite,
/// admissible values.
[[nodiscard]] bool firstOrderGeneralizedAlphaContactProvenanceValid(
    const FreeSurfaceFirstOrderGeneralizedAlphaProvenance&
        provenance) noexcept;

/// Equality of two stage reals within 512 ulp of their scale.
[[nodiscard]] bool contactStageRealValuesMatch(FE::Real left, FE::Real right) noexcept;

/// The composite revision of an accepted contact stage: the previous and
/// endpoint state revisions, the snapshot revision, the stage time and
/// alpha_f, the generalized-alpha provenance and the stage solution.
[[nodiscard]] std::uint64_t acceptedContactStageRevision(
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::uint64_t snapshot_revision,
    FE::Real stage_time,
    FE::Real stage_alpha_f,
    std::span<const FE::Real> stage_solution,
    const std::optional<
        FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha = std::nullopt);

/// The accepted contact stages of the @p declarations with a dynamic contact
/// law, on the current authoritative geometry snapshots of @p system at
/// @p stage_solution.  Collective over the FE system's communicator.
[[nodiscard]] std::vector<FreeSurfaceAcceptedContactStageState>
evaluateAcceptedFreeSurfaceContactStages(
    const FE::systems::FESystem& system,
    std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    FE::Real stage_time,
    FE::Real stage_alpha_f,
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::span<const FE::Real> stage_solution,
    const std::optional<
        FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha = std::nullopt);

/// Binds @p stages to the accepted endpoint @p endpoint_state_revision:
/// recomputes their composite revisions with @p stage_solution.  Collective.
void bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
    std::span<FreeSurfaceAcceptedContactStageState> stages,
    std::uint64_t endpoint_state_revision,
    std::span<const FE::Real> stage_solution,
    const FE::systems::CollectiveCommunicator& comm);

/**
 * @brief The accepted wall-contact cells of the accepted contact stages:
 *        for every declaration with a dynamic Ren-E contact law, the
 *        retained, owned contact rules of its stage's geometry snapshot on
 *        the dynamic-contact walls, canonicalized across the FE
 *        communicator.  Collective.
 *
 * Moved from the Navier-Stokes FreeSurfaceContactStages with body and
 * messages unchanged.
 */
[[nodiscard]] std::vector<FE::level_set::LevelSetWallContactConstraint>
captureAcceptedContactStageWallConstraints(
    const FE::systems::FESystem& system,
    std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    std::span<const FreeSurfaceAcceptedContactStageState>
        accepted_contact_stages);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

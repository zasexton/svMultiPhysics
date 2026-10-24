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

#include "Core/Types.h"
#include "Forms/FormExpr.h"
#include "Systems/FreeSurfaceRecordTypes.h"
#include "Systems/SystemCollectives.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
} // namespace systems

namespace interfaces {

/// True for the exact first-order generalized-alpha parameter family of the
/// time loop (alpha_m = 2 alpha_f - 1/2, gamma = alpha_f) with finite,
/// admissible values.
[[nodiscard]] bool firstOrderGeneralizedAlphaContactProvenanceValid(
    const systems::FreeSurfaceFirstOrderGeneralizedAlphaProvenance&
        provenance) noexcept;

/// Equality of two stage reals within 512 ulp of their scale.
[[nodiscard]] bool contactStageRealValuesMatch(Real left, Real right) noexcept;

/// The composite revision of an accepted contact stage: the previous and
/// endpoint state revisions, the snapshot revision, the stage time and
/// alpha_f, the generalized-alpha provenance and the stage solution.
[[nodiscard]] std::uint64_t acceptedContactStageRevision(
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::uint64_t snapshot_revision,
    Real stage_time,
    Real stage_alpha_f,
    std::span<const Real> stage_solution,
    const std::optional<
        systems::FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha = std::nullopt);

/// The accepted contact stages of the @p declarations with a dynamic contact
/// law, on the current authoritative geometry snapshots of @p system at
/// @p stage_solution.  Collective over the FE system's communicator.
[[nodiscard]] std::vector<systems::FreeSurfaceAcceptedContactStageState>
evaluateAcceptedFreeSurfaceContactStages(
    const systems::FESystem& system,
    std::span<const systems::FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    Real stage_time,
    Real stage_alpha_f,
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::span<const Real> stage_solution,
    const std::optional<
        systems::FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha = std::nullopt);

/// One declared wall functional of one wall as a measured 0-form.
struct DeclaredWallFunctionalForm {
    int wall{-1};
    std::string quantity{};
    std::string name{};   ///< "<prefix>wall.<wall>.<quantity>"
    forms::FormExpr form{};
};

/**
 * @brief The declared wall functionals of one interface on a snapshot.
 *
 * The wall functionals declared on @p system (Systems/FunctionalForms.h)
 * under @p prefix whose quantity is one of @p quantities, as measured
 * 0-forms of each wall of @p walls over its domains in @p snapshot: the
 * interface's contact lines with the wall and the wall's parts on
 * @p liquid_side.  Named "<prefix>wall.<b>.<quantity>", by wall in the order
 * of @p walls and then in declaration order; a wall without an integrand or
 * without a domain of that kind has none.  Not collective: the snapshot
 * lists the same wall domains on every rank.
 */
[[nodiscard]] std::vector<DeclaredWallFunctionalForm> declaredWallFunctionalForms(
    const systems::FESystem& system,
    std::string_view prefix,
    const FreeSurfaceGeometrySnapshot& snapshot,
    std::span<const int> walls,
    geometry::CutIntegrationSide liquid_side,
    std::span<const std::string_view> quantities);

/// Binds @p stages to the accepted endpoint @p endpoint_state_revision:
/// recomputes their composite revisions with @p stage_solution.  Collective.
void bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
    std::span<systems::FreeSurfaceAcceptedContactStageState> stages,
    std::uint64_t endpoint_state_revision,
    std::span<const Real> stage_solution,
    const systems::CollectiveCommunicator& comm);

} // namespace interfaces
} // namespace FE
} // namespace svmp

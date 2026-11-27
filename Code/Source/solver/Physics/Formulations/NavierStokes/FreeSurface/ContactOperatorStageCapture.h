/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file ContactOperatorStageCapture.h
 * @ingroup fe_timestepping
 * @brief Capture of a converged operator stage: the first-order
 *        generalized-alpha candidate-stage observation of a solve attempt
 *        (step and attempt, stage and rate times, scheme parameters, base
 *        mesh revision), the exact field slices of the stage state that a
 *        later reconstruction overlays, their cross-rank consensus, and the
 *        temporal coordinates of the stage for endpoint and generalized-alpha
 *        schemes.
 *
 * The fields are those of the declared interfaces with a dynamic contact law
 * (level set and velocity), and the boundary fingerprint covers their walls.
 */

#include "FE/Core/Types.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRecordTypes.h"
#include "FE/Systems/SystemCollectives.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace svmp::FE {
namespace assembly {
class IMeshAccess;
} // namespace assembly
namespace backends {
class GenericVector;
} // namespace backends
namespace systems {
class FESystem;
} // namespace systems

} // namespace svmp::FE

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/// The exact values of one field of an operator-stage state.
struct DynamicContactOperatorStageFieldSlice {
  FE::FieldId field{FE::INVALID_FIELD_ID};
  FE::GlobalIndex offset{0};
  FE::GlobalIndex count{0};
  std::vector<FE::Real> values{};
};

/// The retained field slices of an operator-stage state.
struct DynamicContactOperatorStageState {
  FE::GlobalIndex full_state_size{0};
  std::vector<DynamicContactOperatorStageFieldSlice> fields{};
};

/// One first-order generalized-alpha candidate-stage observation.
struct DynamicContactFirstOrderGeneralizedAlphaObservation {
  int step_index{0};
  int attempt_index{0};
  FE::Real step_start_time{0.0};
  FE::Real step_end_time{0.0};
  FE::Real state_time{0.0};
  FE::Real rate_time{0.0};
  FE::timestepping::CandidateStageMeshRevision mesh_revision{};
  std::uint64_t contact_wall_boundary_fingerprint{0u};
  FreeSurfaceFirstOrderGeneralizedAlphaProvenance provenance{};
  DynamicContactOperatorStageState operator_stage_state{};
};

/// The base mesh and coordinate frame of @p captured still match
/// @p live_mesh (generated-domain epochs may advance).
[[nodiscard]] bool dynamicContactStageMeshRevisionMatches(
    const FE::timestepping::CandidateStageMeshRevision& captured,
    const FE::assembly::IMeshAccess& live_mesh);

/// Rank-local fingerprint of the boundary faces of the dynamic-contact walls
/// of @p declarations.
[[nodiscard]] std::uint64_t dynamicContactWallBoundaryFingerprint(
    const FE::assembly::IMeshAccess& mesh,
    std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
        declarations);

/// The captured mesh revision and wall fingerprint match @p live_mesh.
[[nodiscard]] bool dynamicContactStageProvenanceMatches(
    const DynamicContactFirstOrderGeneralizedAlphaObservation& captured,
    const FE::assembly::IMeshAccess& live_mesh,
    std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
        declarations);

/// The observation of a complete temporal-order-one generalized-alpha
/// candidate stage (throws std::invalid_argument otherwise).
[[nodiscard]] DynamicContactFirstOrderGeneralizedAlphaObservation
makeDynamicContactFirstOrderGeneralizedAlphaObservation(
    const FE::timestepping::CandidateStageObservation& observation);

/// The exact owned values of the level-set and velocity fields of the
/// dynamic-contact declarations in @p state, agreed on every rank.
[[nodiscard]] DynamicContactOperatorStageState
captureDynamicContactOperatorStageState(
    const FE::systems::FESystem& system,
    std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const FE::backends::GenericVector& state,
    const FE::systems::CollectiveCommunicator& comm);

/// Writes the retained slices of @p exact_state into @p stage_solution after
/// checking them against the current field layout.
void overlayDynamicContactOperatorStageState(
    const FE::systems::FESystem& system,
    std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const DynamicContactOperatorStageState& exact_state,
    std::span<FE::Real> stage_solution);

/// Requires the metadata and values of @p observation to be bitwise
/// identical on every rank of @p comm.
void requireDynamicContactOperatorStageObservationConsensus(
    const DynamicContactFirstOrderGeneralizedAlphaObservation& observation,
    const FE::systems::CollectiveCommunicator& comm);

/// Stage time, alpha_f and generalized-alpha provenance of a contact stage.
struct FreeSurfaceContactStageTemporalCoordinates {
  FE::Real stage_time{0.0};
  FE::Real stage_alpha_f{1.0};
  std::optional<FreeSurfaceFirstOrderGeneralizedAlphaProvenance>
      first_order_generalized_alpha{};
};

/// The endpoint stage of an endpoint scheme, or the authentic
/// generalized-alpha stage of @p generalized_alpha_observation.
[[nodiscard]] FreeSurfaceContactStageTemporalCoordinates
resolveFreeSurfaceContactStageTemporalCoordinates(
    FE::timestepping::SchemeKind scheme,
    const FE::timestepping::TimeHistory& history,
    int expected_attempt_index,
    const std::optional<DynamicContactFirstOrderGeneralizedAlphaObservation>&
        generalized_alpha_observation);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

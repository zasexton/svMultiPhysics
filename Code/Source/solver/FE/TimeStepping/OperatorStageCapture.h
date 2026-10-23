/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file OperatorStageCapture.h
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
 * Moved from Physics/Formulations/NavierStokes/FreeSurface/
 * FreeSurfaceContactStages with bodies, messages and collective order
 * unchanged.  Functions that take a communicator are collective.
 */

#include "Core/Types.h"
#include "Systems/FreeSurfaceRecordTypes.h"
#include "Systems/SystemCollectives.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
class IMeshAccess;
} // namespace assembly
namespace backends {
class GenericVector;
} // namespace backends
namespace systems {
class FESystem;
} // namespace systems

namespace timestepping {

/// The exact values of one field of an operator-stage state.
struct DynamicContactOperatorStageFieldSlice {
  FieldId field{INVALID_FIELD_ID};
  GlobalIndex offset{0};
  GlobalIndex count{0};
  std::vector<Real> values{};
};

/// The retained field slices of an operator-stage state.
struct DynamicContactOperatorStageState {
  GlobalIndex full_state_size{0};
  std::vector<DynamicContactOperatorStageFieldSlice> fields{};
};

/// One first-order generalized-alpha candidate-stage observation.
struct DynamicContactFirstOrderGeneralizedAlphaObservation {
  int step_index{0};
  int attempt_index{0};
  Real step_start_time{0.0};
  Real step_end_time{0.0};
  Real state_time{0.0};
  Real rate_time{0.0};
  CandidateStageMeshRevision mesh_revision{};
  std::uint64_t contact_wall_boundary_fingerprint{0u};
  systems::FreeSurfaceFirstOrderGeneralizedAlphaProvenance provenance{};
  DynamicContactOperatorStageState operator_stage_state{};
};

/// The base mesh and coordinate frame of @p captured still match
/// @p live_mesh (generated-domain epochs may advance).
[[nodiscard]] bool dynamicContactStageMeshRevisionMatches(
    const CandidateStageMeshRevision& captured,
    const assembly::IMeshAccess& live_mesh);

/// Rank-local fingerprint of the boundary faces of the dynamic-contact walls
/// of @p declarations.
[[nodiscard]] std::uint64_t dynamicContactWallBoundaryFingerprint(
    const assembly::IMeshAccess& mesh,
    std::span<const systems::FreeSurfaceDiscreteFunctionalDeclaration>
        declarations);

/// The captured mesh revision and wall fingerprint match @p live_mesh.
[[nodiscard]] bool dynamicContactStageProvenanceMatches(
    const DynamicContactFirstOrderGeneralizedAlphaObservation& captured,
    const assembly::IMeshAccess& live_mesh,
    std::span<const systems::FreeSurfaceDiscreteFunctionalDeclaration>
        declarations);

/// The observation of a complete temporal-order-one generalized-alpha
/// candidate stage (throws std::invalid_argument otherwise).
[[nodiscard]] DynamicContactFirstOrderGeneralizedAlphaObservation
makeDynamicContactFirstOrderGeneralizedAlphaObservation(
    const CandidateStageObservation& observation);

/// The exact owned values of the level-set and velocity fields of the
/// dynamic-contact declarations in @p state, agreed on every rank.
[[nodiscard]] DynamicContactOperatorStageState
captureDynamicContactOperatorStageState(
    const systems::FESystem& system,
    std::span<const systems::FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const backends::GenericVector& state,
    const systems::CollectiveCommunicator& comm);

/// Writes the retained slices of @p exact_state into @p stage_solution after
/// checking them against the current field layout.
void overlayDynamicContactOperatorStageState(
    const systems::FESystem& system,
    std::span<const systems::FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const DynamicContactOperatorStageState& exact_state,
    std::span<Real> stage_solution);

/// Requires the metadata and values of @p observation to be bitwise
/// identical on every rank of @p comm.
void requireDynamicContactOperatorStageObservationConsensus(
    const DynamicContactFirstOrderGeneralizedAlphaObservation& observation,
    const systems::CollectiveCommunicator& comm);

/// Stage time, alpha_f and generalized-alpha provenance of a contact stage.
struct FreeSurfaceContactStageTemporalCoordinates {
  Real stage_time{0.0};
  Real stage_alpha_f{1.0};
  std::optional<systems::FreeSurfaceFirstOrderGeneralizedAlphaProvenance>
      first_order_generalized_alpha{};
};

/// The endpoint stage of an endpoint scheme, or the authentic
/// generalized-alpha stage of @p generalized_alpha_observation.
[[nodiscard]] FreeSurfaceContactStageTemporalCoordinates
resolveFreeSurfaceContactStageTemporalCoordinates(
    SchemeKind scheme,
    const TimeHistory& history,
    int expected_attempt_index,
    const std::optional<DynamicContactFirstOrderGeneralizedAlphaObservation>&
        generalized_alpha_observation);

} // namespace timestepping
} // namespace FE
} // namespace svmp

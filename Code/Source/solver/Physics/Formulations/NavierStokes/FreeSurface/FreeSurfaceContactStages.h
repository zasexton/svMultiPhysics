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
 * @brief Accepted free-surface contact stages: first-order
 *        generalized-alpha operator-stage observations of the dynamic
 *        contact state, stage temporal coordinates and revisions, the
 *        accepted contact-stage evaluation, its wall constraints and its
 *        endpoint binding.
 */

#include "FE/Assembly/Assembler.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Core/Types.h"
#include "FE/Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "FE/LevelSet/LevelSetWallContactConstraint.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FESystemHandle.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"
#include "Mesh/Core/MeshComm.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

bool firstOrderGeneralizedAlphaContactProvenanceValid(
    const svmp::FE::systems::
        FreeSurfaceFirstOrderGeneralizedAlphaProvenance& provenance) noexcept;

bool contactStageRealValuesMatch(
    svmp::FE::Real left, svmp::FE::Real right) noexcept;

struct DynamicContactOperatorStageFieldSlice {
  svmp::FE::FieldId field{svmp::FE::INVALID_FIELD_ID};
  svmp::FE::GlobalIndex offset{0};
  svmp::FE::GlobalIndex count{0};
  std::vector<svmp::FE::Real> values{};
};

struct DynamicContactOperatorStageState {
  svmp::FE::GlobalIndex full_state_size{0};
  std::vector<DynamicContactOperatorStageFieldSlice> fields{};
};

struct DynamicContactFirstOrderGeneralizedAlphaObservation {
  int step_index{0};
  int attempt_index{0};
  svmp::FE::Real step_start_time{0.0};
  svmp::FE::Real step_end_time{0.0};
  svmp::FE::Real state_time{0.0};
  svmp::FE::Real rate_time{0.0};
  svmp::FE::timestepping::CandidateStageMeshRevision mesh_revision{};
  std::uint64_t contact_wall_boundary_fingerprint{0u};
  svmp::FE::systems::FreeSurfaceFirstOrderGeneralizedAlphaProvenance
      provenance{};
  DynamicContactOperatorStageState operator_stage_state{};
};

bool dynamicContactStageMeshRevisionMatches(
    const svmp::FE::timestepping::CandidateStageMeshRevision& captured,
    const svmp::FE::assembly::IMeshAccess& live_mesh);

std::uint64_t dynamicContactWallBoundaryFingerprint(
    const svmp::FE::assembly::IMeshAccess& mesh,
    std::span<const svmp::FE::systems::
                        FreeSurfaceDiscreteFunctionalDeclaration>
        declarations);

bool dynamicContactStageProvenanceMatches(
    const DynamicContactFirstOrderGeneralizedAlphaObservation& captured,
    const svmp::FE::assembly::IMeshAccess& live_mesh,
    std::span<const svmp::FE::systems::
                        FreeSurfaceDiscreteFunctionalDeclaration>
        declarations);

DynamicContactFirstOrderGeneralizedAlphaObservation
makeDynamicContactFirstOrderGeneralizedAlphaObservation(
    const svmp::FE::timestepping::CandidateStageObservation& observation);

DynamicContactOperatorStageState captureDynamicContactOperatorStageState(
    const svmp::FE::systems::FESystem& system,
    std::span<const svmp::FE::systems::
                        FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const svmp::FE::backends::GenericVector& state,
    const svmp::MeshComm& comm);

void overlayDynamicContactOperatorStageState(
    const svmp::FE::systems::FESystem& system,
    std::span<const svmp::FE::systems::
                        FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const DynamicContactOperatorStageState& exact_state,
    std::span<svmp::FE::Real> stage_solution);

void requireDynamicContactOperatorStageObservationConsensus(
    const DynamicContactFirstOrderGeneralizedAlphaObservation& observation,
    const svmp::MeshComm& comm);

struct FreeSurfaceContactStageTemporalCoordinates {
  svmp::FE::Real stage_time{0.0};
  svmp::FE::Real stage_alpha_f{1.0};
  std::optional<svmp::FE::systems::
                    FreeSurfaceFirstOrderGeneralizedAlphaProvenance>
      first_order_generalized_alpha{};
};

FreeSurfaceContactStageTemporalCoordinates
resolveFreeSurfaceContactStageTemporalCoordinates(
    svmp::FE::timestepping::SchemeKind scheme,
    const svmp::FE::timestepping::TimeHistory& history,
    int expected_attempt_index,
    const std::optional<DynamicContactFirstOrderGeneralizedAlphaObservation>&
        generalized_alpha_observation);

std::uint64_t acceptedContactStageRevision(
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::uint64_t snapshot_revision,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::span<const svmp::FE::Real> stage_solution,
    const std::optional<svmp::FE::systems::
                            FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha = std::nullopt);

std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
evaluateAcceptedFreeSurfaceContactStages(
    const svmp::FE::systems::FESystemHandle& sim,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::span<const svmp::FE::Real> stage_solution,
    const std::optional<svmp::FE::systems::
                            FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha = std::nullopt);

std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
captureAcceptedContactStageWallConstraints(
    const svmp::FE::systems::FESystemHandle& sim,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        accepted_contact_stages);

void bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
    std::span<
        svmp::FE::systems::FreeSurfaceAcceptedContactStageState> stages,
    std::uint64_t endpoint_state_revision,
    std::span<const svmp::FE::Real> stage_solution,
    const svmp::MeshComm& comm);

void assertAcceptedFreeSurfaceContactStageEndpointUnchanged(
    std::uint64_t expected_endpoint_algebraic_revision,
    std::span<const svmp::FE::Real> endpoint_solution,
    const svmp::MeshComm& comm);

/// Record the initial accepted functional group
/// (FE/LevelSet/AcceptedFunctionalRecorder) with the accepted contact stages
/// of the initial state.
void recordInitialFreeSurfaceDiscreteFunctionalBaseline(
    const svmp::FE::systems::FESystemHandle& sim,
    std::uint64_t initial_step,
    svmp::FE::Real initial_time,
    std::span<const svmp::FE::Real> initial_solution,
    bool record_backward_euler_kinetic_baseline = false,
    std::optional<std::uint64_t>
        initial_extension_map_revision = std::nullopt);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface

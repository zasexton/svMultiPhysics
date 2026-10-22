/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_LEVELSET_LEVELSETCONSERVATIVEPHASEWORKFLOW_H
#define SVMP_FE_LEVELSET_LEVELSETCONSERVATIVEPHASEWORKFLOW_H

/**
 * @file LevelSetConservativePhaseWorkflow.h
 * @brief Accepted-step workflow of the conservative liquid-phase indicator q.
 *
 * For every level-set equation with a conservative phase indicator the
 * workflow
 * - initializes q from the cut geometry on its fixed P1 phase graph;
 * - binds the accepted graph and checks that it and the background mesh stay
 *   unchanged through the physical solve;
 * - stages, once per step attempt, the Backward Euler endpoint velocity
 *   sampled at the graph nodes, with exact collective provenance checks;
 * - applies a candidate: advances q on the fixed graph, stages the
 *   transported and limited solutions, refreshes the cut geometry inside a
 *   transaction, reconciles the geometry with the transported nodal moments
 *   (LevelSetPhaseGeometryReconciliation), validates and logs the result;
 * - rolls a candidate back, and publishes the flux artifacts of an accepted
 *   step.
 *
 * The functions were moved from Application/Core/ApplicationDriver.cpp with
 * bodies, messages and collective order unchanged.  The services they need
 * that are not in FE yet are supplied by the caller through
 * ConservativePhaseWorkflowHost.  Unless noted otherwise every function is
 * collective over the FE system's active communicator, and a rank-local
 * failure is agreed on before any rank acts on it.
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetConservativePhaseOperator.h"
#include "LevelSet/LevelSetConservativePhaseRegions.h"
#include "LevelSet/LevelSetConservativePhaseState.h"
#include "LevelSet/LevelSetGeometry.h"
#include "LevelSet/LevelSetMaintenanceSchedule.h"
#include "LevelSet/LevelSetMaintenanceSubstage.h"
#include "LevelSet/LevelSetOptions.h"
#include "LevelSet/LevelSetPhaseGeometryReconciliation.h"
#include "LevelSet/LevelSetWallContactConstraint.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace svmp {
namespace FE {

namespace backends {
class GenericVector;
} // namespace backends

namespace systems {
struct SystemStateView;
} // namespace systems

namespace timestepping {
class TimeHistory;
struct CandidateStageObservation;
} // namespace timestepping

namespace level_set {

// ---------------------------------------------------------------------------
// Requests and host
// ---------------------------------------------------------------------------

/**
 * @brief Non-owning view of one level-set maintenance request.
 *
 * The workflow reads the request's transport options (with the conservative
 * phase options), its cut-volume request and its isovalue, and keeps the
 * request's conservative-phase state (the initialization flag and the
 * accepted phase graph) in caller-owned storage.  The caller keeps every
 * referenced object alive while the view is used.  Requests that do not
 * enable the conservative phase are skipped but keep their position: the
 * results of the workflow are indexed like the requests.
 */
struct ConservativePhaseRequest {
    /// The request's immutable configuration.
    struct Configuration {
        const LevelSetTransportOptions& transport;
        const std::optional<ActiveCutVolumeRequest>& volume_cut_request;
        /// Isovalue of the request's level-set volume.
        double isovalue{0.0};
    };

    /// The request's conservative-phase state.
    struct Runtime {
        bool& conservative_phase_initialized;
        std::optional<LevelSetP1PhaseTransportGraph>& conservative_phase_graph;
    };

    const Configuration* configuration{nullptr};
    Runtime runtime;
};

/// Result of ConservativePhaseGeometryTransaction::refresh.
struct ConservativePhaseGeometryRefresh {
    /// True when the refresh installed a newly generated cut context.
    bool refreshed{false};
};

/**
 * @brief Transaction over the generated cut geometry of the FE system.
 *
 * While active, refresh() regenerates the cut-integration context from a
 * candidate FE-ordered solution; rollback() restores the geometry, the
 * level-set mesh fields and the refresh cache of the begin; commit()
 * publishes the current geometry.  Destroying an active transaction rolls it
 * back.  The host supplies it: LevelSetGeometryTransaction
 * (LevelSetGeometryService.h) implements it.
 */
class ConservativePhaseGeometryTransaction {
public:
    virtual ~ConservativePhaseGeometryTransaction() = default;

    /// Regenerates the cut context from @p candidate.  Collective.
    [[nodiscard]] virtual ConservativePhaseGeometryRefresh refresh(
        std::span<const Real> candidate) = 0;

    [[nodiscard]] virtual bool active() const noexcept = 0;

    /// True once commit() has started; rollback() then throws.
    [[nodiscard]] virtual bool publicationStarted() const noexcept = 0;

    virtual void commit() = 0;
    virtual void rollback() = 0;
};

/// Observer of each staged maintenance substage, with the FE-ordered solution
/// before and after it (for example a maintenance energy ledger).
using ConservativePhaseStageObserver = std::function<void(
    MaintenanceSubstage, std::span<const Real>, std::span<const Real>)>;

/**
 * @brief Services of the caller that the workflow uses.
 *
 * Each callback stands in for an FE service that does not exist yet; its
 * documentation names the step that replaces it.  The functions below list
 * the members they use; the others may stay empty.  The callbacks are called
 * on every rank, at the same points, and are collective where the service
 * they stand for is.
 */
struct ConservativePhaseWorkflowHost {
    /// The FE system; the workflow does nothing (or reports) without one.
    systems::FESystem* fe_system{nullptr};

    /// The solution history of the run, for initializeConservativePhaseStates.
    timestepping::TimeHistory* time_history{nullptr};

    /// Begins a transaction over the cut geometry
    /// (LevelSetGeometryTransaction, for example from
    /// LevelSetGeometryService::beginTransaction).
    std::function<std::unique_ptr<ConservativePhaseGeometryTransaction>()>
        begin_geometry_transaction{};

    /// Canonical schedule words of all of the caller's maintenance requests
    /// for the prospective accepted endpoint of step @p completed_step; equal
    /// words on every rank mean identical maintenance requests (S9:
    /// maintenance pipeline).
    std::function<CanonicalLevelSetMaintenanceSchedule(int completed_step)>
        prospective_request_schedule{};

    /// Wall-contact constraints of @p level_set_field at @p endpoint_time,
    /// resolved against the accepted contact stages (S9: wall-aware
    /// maintenance context; the contact data are a Physics input).
    std::function<std::vector<LevelSetWallContactConstraint>(
        timestepping::TimeHistory& history,
        FieldId level_set_field,
        Real endpoint_time,
        std::span<const systems::FreeSurfaceAcceptedContactStageState>
            accepted_contact_stages,
        std::span<const LevelSetWallContactConstraint>
            accepted_contact_stage_constraints)>
        wall_contact_constraints{};

    /// Values of @p field at the vertices of the primary mesh, vertex-major
    /// with @p components values per vertex; @p action names the caller in
    /// errors.  Rank-local (C3b: vertex-field evaluation of the level-set
    /// curvature service).
    std::function<std::vector<double>(
        FieldId field,
        const systems::SystemStateView& state,
        std::size_t components,
        std::string_view action)>
        evaluate_vertex_field{};

    /// Number of vertices of the primary mesh; set with
    /// evaluate_vertex_field (C3b).
    std::function<std::size_t()> primary_mesh_vertex_count{};

    /// Communicator-wide fingerprint of the content of a distributed FE
    /// vector, nonzero for any content (moves to FE with the algebraic-state
    /// revision helpers, S9).
    std::function<std::uint64_t(const backends::GenericVector& state)>
        operator_state_fingerprint{};
};

// ---------------------------------------------------------------------------
// Candidate records
// ---------------------------------------------------------------------------

/// Contact stages of a staged candidate and the solution they were taken at.
// Moved from Application/Core/ApplicationDriver.cpp (lines 23770-23776 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.
struct ConservativePhaseContactStageCandidate {
  std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
      stages{};
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
      constraints{};
  std::vector<svmp::FE::Real> stage_solution{};
};

/// Builds the contact stages of a staged candidate solution, refreshing the
/// geometry through the given transaction (Physics: the free-surface contact
/// stage).  Called once per candidate, before the post-transport geometry
/// refresh.
using ConservativePhaseContactStageCallback =
    std::function<ConservativePhaseContactStageCandidate(
        std::span<const Real>, ConservativePhaseGeometryTransaction*)>;

// Accepted-graph bindings, the staged endpoint-velocity snapshot, the
// per-request maintenance ledger and the candidate result.
// Moved from Application/Core/ApplicationDriver.cpp (lines 24099-24178 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.
struct ConservativePhaseFixedGraphBinding {
  bool enabled{false};
  std::string phase_field_name{};
  svmp::FE::level_set::LevelSetP1PhaseGraphIdentity identity{};
  std::uint64_t geometry_revision{0u};
  std::uint64_t topology_revision{0u};
  std::uint64_t ownership_revision{0u};
  std::uint64_t numbering_revision{0u};
};

struct ConservativePhaseCandidateStageRequest {
  bool enabled{false};
  std::string phase_field_name{};
  std::string velocity_field_name{};
  svmp::FE::level_set::LevelSetVelocitySource velocity_source{
      svmp::FE::level_set::LevelSetVelocitySource::ConstantVector};
  int material_interface_marker{-1};
  svmp::FE::level_set::LevelSetP1PhaseGraphIdentity graph_identity{};
  std::uint64_t graph_geometry_revision{0u};
  std::uint64_t graph_topology_revision{0u};
  std::uint64_t graph_ownership_revision{0u};
  std::uint64_t graph_numbering_revision{0u};
  std::vector<std::array<svmp::FE::Real, 3>> sampled_nodal_velocity{};
};

struct ConservativePhaseCandidateStageSnapshot {
  svmp::FE::level_set::LevelSetP1PhaseSplitScheme scheme{
      svmp::FE::level_set::LevelSetP1PhaseSplitScheme::
          BackwardEulerExplicitIndicatorEndpointVelocity};
  int temporal_order{0};
  std::uint64_t prospective_step{0u};
  std::uint64_t attempt{0u};
  svmp::FE::Real step_start_time{0.0};
  svmp::FE::Real step_end_time{0.0};
  svmp::FE::Real q_input_time{0.0};
  svmp::FE::Real velocity_state_time{0.0};
  svmp::FE::Real time_step{0.0};
  std::uint64_t operator_state_revision{0u};
  std::vector<std::uint64_t> request_schedule_words{};
  std::vector<ConservativePhaseCandidateStageRequest> requests{};
};

// The post_reinitialization_* measures are taken after the geometry refresh
// that follows the transport, before the local reconciliation; they keep
// their names, which the log and the accepted-step readers use.
struct ConservativePhaseMaintenanceStageLedger {
  svmp::FE::Real raw_post_transport_phase_measure{0.0};
  svmp::FE::Real post_limit_phase_measure{0.0};
  svmp::FE::Real raw_post_transport_geometry_measure{0.0};
  svmp::FE::Real post_reinitialization_geometry_measure{0.0};
  svmp::FE::Real post_correction_phase_measure{0.0};
  svmp::FE::Real post_correction_geometry_measure{0.0};
  svmp::FE::Real maximum_nodal_boundary_mass_transfer{0.0};
  svmp::FE::Real boundary_mass_tolerance{0.0};
  svmp::FE::level_set::LevelSetConservativePhaseBoundaryFluxPolicy
      boundary_flux_policy{
          svmp::FE::level_set::
              LevelSetConservativePhaseBoundaryFluxPolicy::
                  ClosedDomainDiscreteQFluxOnly};
  ConservativePhaseMomentMismatch post_reinitialization_mismatch{};
  ConservativePhaseMomentMismatch post_correction_mismatch{};
  svmp::FE::level_set::LevelSetP1PhaseTransportStageResult
      transport_stage{};
  svmp::FE::level_set::LevelSetPhaseRegionLedgerResult region_ledger{};
  ConservativePhaseGeometryReconciliationResult reconciliation{};
  std::optional<svmp::FE::level_set::LevelSetP1PhaseSplitStageProvenance>
      split_stage_provenance{};
};

struct ConservativePhaseCandidateResult {
  bool accept_step{true};
  bool changed{false};
  std::vector<svmp::FE::Real> original_solution{};
  ConservativePhaseContactStageCandidate contact_stage{};
  std::vector<ConservativePhaseMaintenanceStageLedger>
      maintenance_ledgers{};
  std::unique_ptr<ConservativePhaseGeometryTransaction>
      geometry_transaction{};
};

// ---------------------------------------------------------------------------
// Exact comparisons and the artifact fingerprint
// ---------------------------------------------------------------------------

// Moved from Application/Core/ApplicationDriver.cpp (lines 24180-24192, 24533-24537 and 24734-24765 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.

inline bool sameConservativePhaseGraphIdentity(
    const svmp::FE::level_set::LevelSetP1PhaseGraphIdentity& left,
    const svmp::FE::level_set::LevelSetP1PhaseGraphIdentity& right) noexcept
{
  return left.dimension == right.dimension &&
         left.nodes == right.nodes && left.edges == right.edges &&
         left.geometry_revision == right.geometry_revision &&
         left.topology_revision == right.topology_revision &&
         left.ownership_revision == right.ownership_revision &&
         left.numbering_revision == right.numbering_revision &&
         left.dof_layout_revision == right.dof_layout_revision &&
         left.content_revision == right.content_revision;
}

inline bool sameConservativePhaseStageRealBits(double left, double right) noexcept
{
  return std::bit_cast<std::uint64_t>(left) ==
         std::bit_cast<std::uint64_t>(right);
}

template <typename Value>
void mixConservativePhaseArtifactFingerprint(
    std::uint64_t& fingerprint,
    const Value& value) noexcept
{
  static_assert(std::is_trivially_copyable_v<Value>);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  for (std::size_t index = 0u; index < sizeof(Value); ++index) {
    fingerprint ^= static_cast<std::uint64_t>(bytes[index]);
    fingerprint *= 1099511628211ull;
  }
}

inline void mixConservativePhaseArtifactFingerprint(
    std::uint64_t& fingerprint,
    std::string_view value) noexcept
{
  mixConservativePhaseArtifactFingerprint(fingerprint, value.size());
  for (const char character : value) {
    fingerprint ^=
        static_cast<std::uint64_t>(static_cast<unsigned char>(character));
    fingerprint *= 1099511628211ull;
  }
}

inline void mixConservativePhaseArtifactFingerprint(
    std::uint64_t& fingerprint,
    const std::string& value) noexcept
{
  mixConservativePhaseArtifactFingerprint(
      fingerprint, std::string_view(value));
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------

/**
 * @brief The accepted phase graph of @p request, rebuilt when it is stale on
 *        any rank.
 *
 * Throws instead of rebuilding when @p allow_graph_rebuild is false.
 */
[[nodiscard]] LevelSetP1PhaseTransportGraph& requireCurrentConservativePhaseGraph(
    systems::FESystem& system,
    ConservativePhaseRequest& request,
    bool allow_graph_rebuild = true);

/// Projection options of @p request: its generated interface marker, liquid
/// side and invariant tolerance.  Rank-local.
[[nodiscard]] LevelSetP1PhaseProjectionOptions conservativePhaseProjectionOptions(
    const systems::FESystem& system,
    const ConservativePhaseRequest& request);

/// Projection of the current cut geometry onto the phase graph of @p request.
[[nodiscard]] LevelSetP1PhaseProjectionResult projectCurrentConservativePhaseGeometry(
    systems::FESystem& system,
    ConservativePhaseRequest& request,
    bool allow_graph_rebuild = true);

/**
 * @brief Sets q of every enabled request, not yet initialized, to the
 *        projection of the cut geometry, in u and every history level, with
 *        zero rates.
 *
 * Uses host.fe_system and host.time_history; does nothing without them.
 */
void initializeConservativePhaseStates(
    const ConservativePhaseWorkflowHost& host,
    std::span<ConservativePhaseRequest> requests);

/**
 * @brief Velocity of @p request at the nodes of @p graph in @p state.
 *
 * A constant velocity, or the vertex average of the request's velocity field
 * (or the common trace velocity of its material-interface phase pair).
 * Uses host.fe_system, host.evaluate_vertex_field and
 * host.primary_mesh_vertex_count.
 */
[[nodiscard]] std::vector<std::array<Real, 3>> sampleConservativePhaseVelocity(
    const ConservativePhaseWorkflowHost& host,
    const ConservativePhaseRequest& request,
    const LevelSetP1PhaseTransportGraph& graph,
    const systems::SystemStateView& state);

/// Flags (1) the phase-graph nodes of the parent cells of @p constraints.
[[nodiscard]] std::vector<std::uint8_t> conservativePhaseContactProtectedNodes(
    const systems::FESystem& system,
    const ConservativePhaseRequest& request,
    const LevelSetP1PhaseTransportGraph& graph,
    std::span<const LevelSetWallContactConstraint> constraints);

/// True when @p velocity has the same size and bits on every rank.
[[nodiscard]] bool collectiveConservativePhaseVelocityBitsAgree(
    std::span<const std::array<Real, 3>> velocity,
    const systems::CollectiveCommunicator& comm);

/// True when the canonical words and sampled velocities of @p stage agree
/// on every rank.
[[nodiscard]] bool collectiveConservativePhaseCandidateStageSnapshotAgrees(
    const ConservativePhaseCandidateStageSnapshot& stage,
    const systems::CollectiveCommunicator& comm);

/// Appends the canonical words of a split-stage provenance (rank-local).
void appendConservativePhaseSplitStageProvenanceWords(
    std::vector<std::uint64_t>& words,
    const std::optional<LevelSetP1PhaseSplitStageProvenance>& provenance);

/// Binds the accepted phase graph of every request (identity and mesh
/// revisions), with identical graph identities on every rank.
[[nodiscard]] std::vector<ConservativePhaseFixedGraphBinding>
captureConservativePhaseFixedGraphBindings(
    const systems::FESystem& system,
    std::span<const ConservativePhaseRequest> requests,
    const systems::CollectiveCommunicator& comm);

/// Throws on every rank when a bound graph or the background mesh changed
/// on any rank; @p phase names the check point.
void requireConservativePhaseFixedGraphBindingsCurrent(
    const systems::FESystem& system,
    std::span<const ConservativePhaseRequest> requests,
    std::span<const ConservativePhaseFixedGraphBinding> bindings,
    const systems::CollectiveCommunicator& comm,
    std::string_view phase);

/**
 * @brief Stages the endpoint velocity and provenance of one Backward Euler
 *        step attempt.
 *
 * Uses host.fe_system, host.prospective_request_schedule,
 * host.operator_state_fingerprint and the members that
 * sampleConservativePhaseVelocity uses.
 */
[[nodiscard]] ConservativePhaseCandidateStageSnapshot
buildConservativePhaseCandidateStageSnapshot(
    const ConservativePhaseWorkflowHost& host,
    timestepping::TimeHistory& history,
    std::span<ConservativePhaseRequest> requests,
    std::span<const ConservativePhaseFixedGraphBinding> fixed_graph_bindings,
    const timestepping::CandidateStageObservation& observation,
    const systems::CollectiveCommunicator& comm);

/**
 * @brief Writes the flux artifacts that are due at an accepted step.
 *
 * Files go to <results_folder>/conservative_phase_flux (the working
 * directory when @p results_folder is empty), written by rank 0; a failed
 * write removes the files of the same step.
 */
void writeAcceptedConservativePhaseArtifacts(
    std::string_view results_folder,
    std::span<const ConservativePhaseRequest> requests,
    const ConservativePhaseCandidateResult& candidate,
    std::uint64_t accepted_step,
    Real accepted_time,
    Real time_step,
    std::uint64_t state_revision,
    const systems::CollectiveCommunicator& comm);

/**
 * @brief Applies the conservative-phase candidate of the step attempt staged
 *        in @p temporal_stage to history.u().
 *
 * On success the result keeps the active geometry transaction for the
 * caller's commit or rollback.  A Courant or reconciliation rejection
 * returns accept_step = false with the transaction rolled back.  Uses
 * host.fe_system, host.begin_geometry_transaction,
 * host.prospective_request_schedule and host.wall_contact_constraints.
 */
[[nodiscard]] ConservativePhaseCandidateResult applyConservativePhaseCandidates(
    const ConservativePhaseWorkflowHost& host,
    timestepping::TimeHistory& history,
    std::span<ConservativePhaseRequest> requests,
    ConservativePhaseCandidateStageSnapshot& temporal_stage,
    std::span<const ConservativePhaseFixedGraphBinding> fixed_graph_bindings,
    std::uint64_t expected_attempt,
    const ConservativePhaseContactStageCallback& contact_stage_builder = {},
    const ConservativePhaseStageObserver& observe_stage = {});

/// Rolls back the geometry transaction of @p result and restores
/// history.u(); rethrows the first failure.
void rollbackConservativePhaseCandidate(
    timestepping::TimeHistory& history,
    ConservativePhaseCandidateResult& result);

} // namespace level_set
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_LEVELSET_LEVELSETCONSERVATIVEPHASEWORKFLOW_H

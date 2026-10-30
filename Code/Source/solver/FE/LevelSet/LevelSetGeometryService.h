/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetGeometryService.h
 * @brief The generated cut-geometry lifecycle of a level-set run
 *
 * Regenerates the cut integration context of an FE system from a level-set
 * state for every resolved cut-volume request (ActiveCutVolumeRequest): the
 * generated interface domains, the interface-boundary intersections and sharp
 * active boundaries, the authoritative free-surface geometry snapshots, the
 * cut-adjacent facet sets, the constraint state and the communicator-wide
 * diagnostics.  A refresh cache keyed on a content signature keeps, resyncs
 * or reinstalls an installed context when the generating input is provably
 * unchanged; the restore retention reinstalls the geometry of an attempt's
 * entry state; LevelSetGeometryTransaction checkpoints and rolls back the
 * geometry, the level-set mesh fields and the cache.  LevelSetGeometryService
 * holds these pieces for one run: the resolved requests, the generated
 * interface lifecycle, the refresh cache and the cut-topology attempt
 * protocol.
 *
 * Moved from Application/Core/ApplicationDriver.cpp with bodies, messages,
 * environment names and collective order unchanged.  The Application
 * services the geometry uses come from LevelSetGeometryHost.  Unless noted
 * otherwise every function is collective over the FE system's active
 * communicator.  Requires the Mesh integration (FE_WITH_MESH): the level-set
 * values are also kept in a vertex field of the primary mesh.
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "LevelSet/LevelSetCutTopology.h"
#include "LevelSet/LevelSetGeometry.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Systems/SystemCollectives.h"
#include "TimeStepping/GeneratedState.h"
#include "TimeStepping/StepParticipants.h"
#include "Mesh/Mesh.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {

class LogRecord;

namespace assembly {
class CutIntegrationContext;
class IMeshAccess;
} // namespace assembly

namespace backends {
class GenericVector;
} // namespace backends

namespace interfaces {
class FreeSurfaceGeometrySnapshotCache;
class FreeSurfaceGeometrySnapshotReuseCache;
class LevelSetInterfaceDomain;
} // namespace interfaces

namespace systems {
class FESystem;
struct SystemStateView;
} // namespace systems

namespace level_set {

/**
 * @brief The caller's objects and services that the cut geometry uses.
 *
 * The snapshot caches live with the caller (today SimulationComponents), so
 * that their lifetime and statistics do not depend on who refreshes.  The
 * member names are those of SimulationComponents, so the moved bodies read
 * them unchanged.
 */
struct LevelSetGeometryHost {
  /// The FE system whose cut integration context is regenerated.
  svmp::FE::systems::FESystem* fe_system{nullptr};

  /// The primary mesh: the level-set values are synchronized into, and
  /// checkpointed from, its vertex field of the level-set field's name.
  svmp::Mesh* primary_mesh{nullptr};

  /// Free-surface geometry snapshots by revision key (created on first use).
  std::unique_ptr<svmp::FE::interfaces::FreeSurfaceGeometrySnapshotCache>&
      free_surface_geometry_snapshot_cache;

  /// Previous snapshot build of each generated domain (keyed by domain id),
  /// for reusing its full-cell records in the next rebuild.
  std::map<std::string,
           std::unique_ptr<
               svmp::FE::interfaces::FreeSurfaceGeometrySnapshotReuseCache>>&
      free_surface_geometry_snapshot_reuse_caches;

  /// Checks that an equation-level cut-volume request has a cut-volume
  /// consumer for its resolved interface marker (Application: provenance
  /// messages of the input).  Optional.
  std::function<void(const svmp::FE::systems::FESystem&,
                     const ActiveCutVolumeRequest&,
                     int resolved_marker)>
      validate_cut_volume_consumer{};

  /// True when a consumer of the generated cut data reads ghost cells beyond
  /// ActiveCutVolumeRequest::generated_cut_ghost_layers (Application: from the curvature recovery
  /// modes and the assembly ghost policy).  Unset: every layer is kept.
  std::function<bool(const std::vector<ActiveCutVolumeRequest>&)>
      generated_cut_needs_every_ghost_layer{};

  /// True when the FE system declares two-fluid accepted-stage diagnostics,
  /// which integrate over the volume points of full-cell snapshot records
  /// without the mesh: the snapshot then keeps those points.
  bool two_fluid_accepted_stage_diagnostics_declared{false};
};

// ---------------------------------------------------------------------------
// State of the refresh cache, the checkpoints and the transaction
// ---------------------------------------------------------------------------

struct ActiveLevelSetMeshFieldCheckpoint {
  struct Field {
    std::string name{};
    std::size_t components{0};
    std::size_t entity_count{0};
    std::vector<double> values{};
  };

  bool valid{false};
  svmp::MeshRevisionState revisions{};
  std::vector<Field> fields{};
};

struct GhostCutWithholding {
  std::size_t fragments{0u};
  std::size_t regions{0u};
  std::size_t sensitivity_records{0u};
};

struct ActiveCutContextRefreshSignature {
  enum class SolutionSignatureKind : std::uint8_t {
    LevelSetFieldByteHash = 1,
    VectorValueRevision = 2
  };

  std::uint64_t request_policy_key{0};
  std::uint64_t mesh_geometry_revision{0};
  std::uint64_t mesh_topology_revision{0};
  std::uint64_t mesh_ownership_revision{0};
  std::uint64_t mesh_numbering_revision{0};
  std::uint64_t mesh_field_layout_revision{0};
  std::uint64_t mesh_label_revision{0};
  std::uint64_t mesh_active_configuration_epoch{0};
  std::uint64_t mesh_coordinate_configuration_key{0};
  std::uint64_t system_space_revision{0};
  std::uint64_t system_dof_layout_revision{0};
  std::uint64_t system_block_layout_revision{0};
  // Content key of the level-set field DOF layouts that the generated
  // geometry reads (see activeCutLevelSetDofLayoutKey).
  std::uint64_t level_set_dof_layout_key{0};
  SolutionSignatureKind solution_signature_kind{
      SolutionSignatureKind::LevelSetFieldByteHash};
  std::uint64_t solution_hash{0};
  std::size_t solution_size{0};

  [[nodiscard]] bool operator==(
      const ActiveCutContextRefreshSignature& other) const noexcept
  {
    // FESystem::setup() advances the system DOF and block layout revisions
    // whenever it runs, also when it redistributes an identical layout (for
    // example after a mesh-field value change in every time step). A
    // level-set byte hash together with the DOF layout content key already
    // identifies the geometry input, so the counters are compared only for
    // value-revision signatures, whose counter is meaningful only for an
    // unchanged vector layout.
    const bool layout_revisions_match =
        solution_signature_kind !=
            SolutionSignatureKind::VectorValueRevision ||
        (system_dof_layout_revision == other.system_dof_layout_revision &&
         system_block_layout_revision ==
             other.system_block_layout_revision);
    return request_policy_key == other.request_policy_key &&
           mesh_geometry_revision == other.mesh_geometry_revision &&
           mesh_topology_revision == other.mesh_topology_revision &&
           mesh_ownership_revision == other.mesh_ownership_revision &&
           mesh_numbering_revision == other.mesh_numbering_revision &&
           mesh_field_layout_revision == other.mesh_field_layout_revision &&
           mesh_label_revision == other.mesh_label_revision &&
           mesh_active_configuration_epoch ==
               other.mesh_active_configuration_epoch &&
           mesh_coordinate_configuration_key ==
               other.mesh_coordinate_configuration_key &&
           system_space_revision == other.system_space_revision &&
           layout_revisions_match &&
           level_set_dof_layout_key == other.level_set_dof_layout_key &&
           solution_signature_kind == other.solution_signature_kind &&
           solution_hash == other.solution_hash &&
           solution_size == other.solution_size;
  }
};

struct ActiveCutContextRefreshCache {
  std::optional<ActiveCutContextRefreshSignature> last_signature{};
  std::optional<ActiveCutContextRefreshSignature> last_vector_signature{};
  std::map<int, std::uint64_t> evaluated_state_source_revisions{};
  std::optional<std::uint64_t> topology_key{};
  std::function<void(
      const ActiveCutContextRefreshReport&,
      std::string_view)> observer{};
  // Context installed by the most recent rebuild through this cache, i.e.
  // the one described by last_signature while it is still installed.
  std::weak_ptr<const svmp::FE::assembly::CutIntegrationContext>
      last_context{};

  // Generated geometry of the entry state of the current nonlinear attempt.
  // A rollback to that state (RestoredOuterFixedPointState, and
  // RestoredTimeStepState when the accepted state has the same level set)
  // reinstalls it instead of rebuilding an identical context; see
  // reinstallRetainedActiveCutContextForRestoredState. It keeps one extra
  // context, including its free-surface snapshot, alive from the second
  // rebuild of the attempt until the attempt ends, so it is opt-in
  // (SVMP_RETAIN_RESTORED_CUT_CONTEXT=1): the peak memory of a 3D run grows
  // by one snapshot while only failed attempts can use it.
  struct RetainedContext {
    ActiveCutContextRefreshSignature signature{};
    std::shared_ptr<const svmp::FE::assembly::CutIntegrationContext>
        context{};
    std::map<int, std::uint64_t> evaluated_state_source_revisions{};
    std::optional<std::uint64_t> topology_key{};
  };
  enum class RestorePinState : std::uint8_t {
    Idle,     // no attempt entry is being recorded
    Armed,    // before_physics_solve ran; the entry sync has not started
    Pinning,  // inside the entry synchronization of the attempt
  };
  std::optional<RetainedContext> retained_restore_context{};
  RestorePinState restore_pin_state{RestorePinState::Idle};

  // Exact reinstall of the most recently replaced context. While the window
  // is open, a rebuild first moves the installed context (when this cache
  // describes it) into reinstall_candidate, and a later refresh whose state
  // has the same generated-geometry input reinstalls it instead of
  // rebuilding identical geometry (the two contexts swap roles). The window
  // spans the endpoint-candidate synchronization and the dynamic-contact
  // stage reconstruction of the final candidate gate, which refreshes the
  // stage geometry of the last outer pass and then the endpoint again. At
  // most one context is retained, and only the installed one is moved into
  // the slot before a rebuild, so at most two contexts are alive at any
  // time, as during any rebuild.
  bool reinstall_window_open{false};
  std::optional<RetainedContext> reinstall_candidate{};

  void closeReinstallWindow() noexcept
  {
    reinstall_window_open = false;
    reinstall_candidate.reset();
  }

  void invalidateGeneratedState() noexcept
  {
    last_signature.reset();
    last_vector_signature.reset();
    evaluated_state_source_revisions.clear();
    topology_key.reset();
    last_context.reset();
    retained_restore_context.reset();
    restore_pin_state = RestorePinState::Idle;
    closeReinstallWindow();
  }
};

struct PrescribedFieldCoefficientCheckpoint {
  struct Field {
    svmp::FE::FieldId id{svmp::FE::INVALID_FIELD_ID};
    bool had_coefficients{false};
    std::vector<svmp::FE::Real> coefficients{};
  };

  std::vector<Field> fields{};
};

struct CurrentCandidateProducerLinkageResult {
  bool success{false};
  std::string diagnostic{"current_candidate_producer_linkage_unavailable"};
};

// ---------------------------------------------------------------------------
// Collective helpers, diagnostics and vertex fields
// ---------------------------------------------------------------------------

/// Sorted union of the boundary markers of every rank.  Collective.
[[nodiscard]] std::vector<int> communicatorWideBoundaryMarkers(
    const svmp::FE::assembly::IMeshAccess& mesh,
    const svmp::FE::systems::CollectiveCommunicator& comm);

/// Opt-in (SVMP_CUT_MEMORY_REPORT=1) storage report of the cut geometry: the
/// cut_geometry_memory record of @p phase, with the fields @p details adds.
[[nodiscard]] bool cutGeometryMemoryReportEnabled();
void logCutGeometryMemory(
    const char* phase,
    const std::function<void(svmp::FE::LogRecord&)>& details = {});

[[nodiscard]] const char* activeSideName(LevelSetActiveSide side) noexcept;

/// Cut (partial) cells of @p marker on @p active_side, sorted.  Rank-local.
[[nodiscard]] std::vector<svmp::FE::MeshIndex> activeCutCellsForMarkerAndSide(
    const svmp::FE::assembly::CutIntegrationContext& context,
    int marker,
    LevelSetActiveSide active_side);

/// Logs a topology change of a refresh at a synchronization point (trace
/// diagnostic) and records the topology key.  Rank-local.
void logCutTopologyChange(
    const ActiveCutContextRefreshReport& report,
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point,
    std::optional<std::uint64_t>& previous_topology_key,
    const char* solve_kind);

/// Policy key of @p requests, which must agree on every rank.
[[nodiscard]] std::uint64_t requireActiveCutRequestConsensus(
    const svmp::FE::systems::FESystem& system,
    const std::vector<ActiveCutVolumeRequest>& requests);

/// Values of @p field at the vertices of @p mesh, vertex-major with
/// @p components values per vertex; @p action names the caller in errors.
/// Rank-local.
[[nodiscard]] std::vector<double> evaluateVertexField(
    const svmp::FE::systems::FESystem& system,
    const svmp::Mesh& mesh,
    svmp::FE::FieldId field,
    const svmp::FE::systems::SystemStateView& state,
    std::size_t components,
    std::string_view action);

/// Writes the level-set values of @p fe_solution into the primary mesh's
/// vertex fields; returns the number of fields that changed.  Rank-local.
std::size_t syncActiveLevelSetVertexFieldsFromSolution(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::span<const svmp::FE::Real> fe_solution);

/// Face-adjacency distance of every stored cell from the owned cells (owned
/// cells are layer 0; unreachable cells get INT_MAX).  Rank-local.
[[nodiscard]] std::vector<int> ghostCellLayers(
    const svmp::FE::assembly::IMeshAccess& mesh);

GhostCutWithholding withholdGhostCutLayersBeyond(
    svmp::FE::interfaces::LevelSetInterfaceDomain& domain,
    const std::vector<int>& layers,
    int keep_layers);

// ---------------------------------------------------------------------------
// Refresh, cache and restore retention
// ---------------------------------------------------------------------------

/// Regenerates and installs the cut integration context from @p fe_solution
/// (FE-ordered) and rebuilds the constraint state.
ActiveCutContextRefreshReport refreshActiveCutIntegrationContextFromSolution(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    std::span<const svmp::FE::Real> fe_solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    const char* provenance,
    const char* solution_source = nullptr,
    bool request_consensus_already_checked = false);

/// Calls the cache's observer with @p report.  Rank-local.
void observeActiveCutContextRefresh(
    const ActiveCutContextRefreshCache& cache,
    const ActiveCutContextRefreshReport& report,
    const char* provenance);

/// Content signature of the generating input in an FE-ordered solution, or
/// nullopt when the level-set layout is unavailable.  Rank-local.
[[nodiscard]] std::optional<ActiveCutContextRefreshSignature> activeCutContextRefreshSignature(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    std::span<const svmp::FE::Real> fe_solution);

/// Content or value-revision signature of a distributed solution.  Rank-local.
[[nodiscard]] std::optional<ActiveCutContextRefreshSignature> activeCutContextRefreshSignature(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& requests,
    const svmp::FE::backends::GenericVector& fe_solution);

/// Cached refresh from an FE-ordered solution: keeps, resyncs or reinstalls
/// the installed context when the signature proves it current, otherwise
/// rebuilds and records the new signature.
ActiveCutContextRefreshReport refreshActiveCutIntegrationContextFromSolutionCached(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    std::span<const svmp::FE::Real> fe_solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance,
    const char* solution_source = nullptr);

/// Cached refresh from a distributed solution vector.
ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCached(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    svmp::FE::backends::GenericVector& solution,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance);

/// Cached refresh from a state view (its vector when present).
ActiveCutContextRefreshReport refreshActiveCutIntegrationContextCached(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
    ActiveCutContextRefreshCache& cache,
    const char* provenance);

/// Starts recording the entry geometry of an attempt (before_physics_solve).
void armActiveCutRestoreRetention(
    ActiveCutContextRefreshCache& cache) noexcept;

/// Drops the retained entry geometry (accepted step).
void releaseActiveCutRestoreRetention(
    ActiveCutContextRefreshCache& cache) noexcept;

/// Retains the installed context at the attempt entry (opt-in:
/// SVMP_RETAIN_RESTORED_CUT_CONTEXT=1).  Rank-local.
void retainActiveCutContextAtAttemptEntry(
    const LevelSetGeometryHost& sim,
    ActiveCutContextRefreshCache& cache,
    bool outer_fixed_point_state,
    bool projected_outer_fixed_point_state);

/// Reinstalls the retained entry geometry for a restored state that provably
/// has the same generating input; nullopt when the caller must refresh.
std::optional<ActiveCutContextRefreshReport>
reinstallRetainedActiveCutContextForRestoredState(
    const LevelSetGeometryHost& sim,
    const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
    const svmp::FE::systems::SystemStateView& state,
    ActiveCutContextRefreshCache& cache,
    const char* provenance);

// ---------------------------------------------------------------------------
// Geometry transaction
// ---------------------------------------------------------------------------

/**
 * @brief Transaction over the generated cut geometry of the FE system.
 *
 * The constructor checkpoints the refresh cache, the level-set mesh fields
 * and the prescribed fields, and begins the lifecycle and cut-context
 * transactions.  While active, refresh() regenerates the geometry of a
 * candidate solution through the cache; rollback() restores everything of
 * the begin; commit() publishes the current geometry.  Destroying an active
 * transaction rolls it back.  Every step is collective.  It is the geometry
 * transaction of the conservative-phase workflow
 * (ConservativePhaseGeometryTransaction); its one-argument refresh uses the
 * requests of the begin.
 */
class LevelSetGeometryTransaction
    : public ConservativePhaseGeometryTransaction {
public:
  LevelSetGeometryTransaction(
      LevelSetGeometryHost sim,
      svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
      ActiveCutContextRefreshCache& refresh_cache,
      const std::vector<ActiveCutVolumeRequest>& requests,
      std::span<const svmp::FE::FieldId> prescribed_fields = {});

  LevelSetGeometryTransaction(const LevelSetGeometryTransaction&) = delete;
  LevelSetGeometryTransaction& operator=(const LevelSetGeometryTransaction&) = delete;

  ~LevelSetGeometryTransaction() override;

  /// Refreshes the cut geometry from @p candidate; @p force_rebuild drops
  /// the cached signatures first.
  [[nodiscard]] ActiveCutContextRefreshReport refresh(
      const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
      std::span<const svmp::FE::Real> candidate,
      bool force_rebuild = false);

  /// refresh() with the requests of the begin.
  [[nodiscard]] ActiveCutContextRefreshReport refreshCandidate(
      std::span<const svmp::FE::Real> candidate,
      bool force_rebuild = false);

  /// The conservative-phase workflow's refresh: refreshCandidate(candidate).
  [[nodiscard]] ConservativePhaseGeometryRefresh refresh(
      std::span<const svmp::FE::Real> candidate) override;

  /// The requests of the begin.
  [[nodiscard]] const std::vector<ActiveCutVolumeRequest>& requests()
      const noexcept;

  [[nodiscard]] bool active() const noexcept override;

  [[nodiscard]] bool publicationStarted() const noexcept override;

  /// Whether the installed geometry was generated from @p candidate by the
  /// refresh that produced @p refresh_report.
  [[nodiscard]] CurrentCandidateProducerLinkageResult currentCandidateProducerLinkage(
      const std::vector<ActiveCutVolumeRequest>& active_cut_requests,
      std::span<const svmp::FE::Real> candidate,
      const ActiveCutContextRefreshReport& refresh_report) const;

  void commit() override;
  void rollback() override;

private:
  LevelSetGeometryHost sim_;
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle_;
  ActiveCutContextRefreshCache& refresh_cache_;
  std::vector<ActiveCutVolumeRequest> requests_{};
  ActiveCutContextRefreshCache refresh_cache_backup_{};
  ActiveLevelSetMeshFieldCheckpoint mesh_field_checkpoint_{};
  PrescribedFieldCoefficientCheckpoint prescribed_field_checkpoint_{};
  bool lifecycle_transaction_active_{false};
  bool system_transaction_active_{false};
  bool publication_started_{false};
  bool active_{false};
};

// ---------------------------------------------------------------------------
// The cut geometry of one run
// ---------------------------------------------------------------------------

/**
 * @brief The generated cut geometry of one run.
 *
 * Holds the host, the resolved cut-volume requests of the run (parsed once),
 * the generated interface lifecycle, the refresh cache and the cut-topology
 * attempt tracker, and refreshes, restores and checkpoints the geometry
 * through the functions above with them.  Not copyable: the cache observer
 * refers to the tracker.
 */
class LevelSetGeometryService {
public:
  LevelSetGeometryService(LevelSetGeometryHost host,
                          std::vector<ActiveCutVolumeRequest> requests);

  LevelSetGeometryService(const LevelSetGeometryService&) = delete;
  LevelSetGeometryService& operator=(const LevelSetGeometryService&) = delete;

  [[nodiscard]] const LevelSetGeometryHost& host() const noexcept
  {
    return host_;
  }

  [[nodiscard]] const std::vector<ActiveCutVolumeRequest>& requests()
      const noexcept
  {
    return requests_;
  }

  [[nodiscard]] svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle&
  lifecycle() noexcept
  {
    return lifecycle_;
  }

  [[nodiscard]] ActiveCutContextRefreshCache& refreshCache() noexcept
  {
    return refresh_cache_;
  }

  [[nodiscard]] TransientCutTopologyAttemptTracker& topologyTracker() noexcept
  {
    return topology_tracker_;
  }

  /// The topology key of the last logCutTopologyChange.
  [[nodiscard]] std::optional<std::uint64_t>& lastLoggedTopologyKey() noexcept
  {
    return last_logged_topology_key_;
  }

  /// Every refresh through the cache is observed by the attempt tracker.
  void trackTopology();

  /// refreshActiveCutIntegrationContextCached with the run's requests.
  ActiveCutContextRefreshReport refreshCached(
      svmp::FE::backends::GenericVector& solution,
      const char* provenance);
  ActiveCutContextRefreshReport refreshCached(
      const svmp::FE::systems::SystemStateView& state,
      const char* provenance);

  /// reinstallRetainedActiveCutContextForRestoredState with the run's
  /// requests.
  std::optional<ActiveCutContextRefreshReport> reinstallForRestoredState(
      const svmp::FE::systems::SystemStateView& state,
      const char* provenance);

  /// retainActiveCutContextAtAttemptEntry.
  void retainAtAttemptEntry(bool outer_fixed_point_state,
                            bool projected_outer_fixed_point_state);

  /// A geometry transaction over the run's requests.
  [[nodiscard]] std::unique_ptr<LevelSetGeometryTransaction> beginTransaction(
      std::span<const svmp::FE::FieldId> prescribed_fields = {});

private:
  LevelSetGeometryHost host_;
  std::vector<ActiveCutVolumeRequest> requests_;
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle_{};
  ActiveCutContextRefreshCache refresh_cache_{};
  TransientCutTopologyAttemptTracker topology_tracker_{};
  std::optional<std::uint64_t> last_logged_topology_key_{};
};

/**
 * @brief The cut geometry of a solve as a generated-state participant.
 *
 * The Geometry role of the GeneratedStateRegistry of the Application's
 * steady and transient solves: prepare() resets the refresh cache at restore
 * points (and, transient, opens the reinstall window for the endpoint of a
 * dynamic-contact stage), synchronize() refreshes or reinstalls the cut
 * context and publishes its source value revisions, and invalidate() drops
 * the generated state after a rejected topology change.  A transient
 * participant that tracks the cut topology provides the outer fixed point's
 * epoch hooks from the attempt tracker.
 */
class LevelSetGeometryParticipant final
    : public svmp::FE::timestepping::IGeneratedStateParticipant {
public:
  struct Options {
    svmp::FE::timestepping::GeneratedStateSolve solve{
        svmp::FE::timestepping::GeneratedStateSolve::Transient};
    /// Transient: the attempt tracker provides the epoch hooks.
    bool track_topology{false};
    /// Transient: keep the context replaced at the endpoint candidate so
    /// the final candidate gate of a dynamic-contact stage can reinstall it.
    bool reinstall_after_contact_stage_endpoint{false};
    /// Transient: the request policy key every refresh must report.
    std::uint64_t expected_request_policy_key{0u};
  };

  LevelSetGeometryParticipant(std::shared_ptr<LevelSetGeometryService> service,
                              Options options);

  [[nodiscard]] std::string_view name() const noexcept override
  {
    return "level_set_geometry";
  }
  [[nodiscard]] svmp::FE::timestepping::GeneratedStateRole role()
      const noexcept override
  {
    return svmp::FE::timestepping::GeneratedStateRole::Geometry;
  }

  void prepare(const svmp::FE::timestepping::GeneratedStateSync& sync) override;
  void synchronize(const svmp::FE::systems::SystemStateView& state,
                   svmp::FE::timestepping::GeneratedStateSync& sync) override;
  void invalidate() override;

  [[nodiscard]] bool providesEpochs() const noexcept override;
  [[nodiscard]] bool epochDiscontinuity(
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
      const override;
  void acknowledgeEpochDiscontinuity(
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
      override;
  [[nodiscard]] bool epochRevisited(
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
      const override;
  void acknowledgeFrozenEpoch() override;

private:
  std::shared_ptr<LevelSetGeometryService> service_;
  Options options_;
};

/**
 * @brief The cut geometry's part of each time step, as TimeLoop hooks.
 *
 * The attempt protocol of the cut topology (the service's
 * TransientCutTopologyAttemptTracker) and the restore retention of the
 * refresh cache, at the points of the step where the transient driver ran
 * them.  Two parts keep that order around the caller's own callbacks;
 * addTo() registers both:
 * - Entry (StepParticipantOrder::BeforeBase): step start begins the topology
 *   attempt; before the physics solve arms the restore retention, refreshes
 *   the cut geometry from the attempt-entry state ("before_physics_solve")
 *   and requires the accepted topology baseline that this refresh seeds;
 *   step accepted releases the retention.
 * - Exit (StepParticipantOrder::AfterBase, whose candidate-discarded hook
 *   runs before the caller's): candidate discarded closes the reinstall
 *   window and discards the attempt, so that the caller's rollback refresh
 *   is not observed by the attempt; commit ready completes the attempt after
 *   the caller's commit.
 * Without topology tracking only the retention, the reinstall window and the
 * attempt-entry refresh remain.  Every hook is collective where the refresh
 * is.
 */
class LevelSetGeometryStepParticipant final
    : public svmp::FE::timestepping::IStepParticipant {
public:
  enum class Part : std::uint8_t {
    Entry,
    Exit
  };

  struct Options {
    /// The attempt tracker follows the cut topology of each attempt.
    bool track_topology{false};
  };

  LevelSetGeometryStepParticipant(
      std::shared_ptr<LevelSetGeometryService> service,
      Part part,
      Options options);

  /// Registers the entry part (BeforeBase) and the exit part (AfterBase) of
  /// @p service with @p list; returns the entry part, whose
  /// attemptEntryReport() the caller's before-physics-solve hook reads.
  static std::shared_ptr<LevelSetGeometryStepParticipant> addTo(
      svmp::FE::timestepping::StepParticipantList& list,
      std::shared_ptr<LevelSetGeometryService> service,
      Options options);

  [[nodiscard]] svmp::FE::timestepping::StepHook hooks()
      const noexcept override;

  void onStepStart(const svmp::FE::timestepping::TimeHistory& history) override;
  bool onBeforePhysicsSolve(svmp::FE::timestepping::TimeHistory& history,
                            double solve_time,
                            double dt) override;
  void onStepCandidateDiscarded(
      svmp::FE::timestepping::TimeHistory& history) override;
  void onStepCommitReady(svmp::FE::timestepping::TimeHistory& history) override;
  void onStepAccepted(svmp::FE::timestepping::TimeHistory& history) override;

  /// The report of the latest attempt-entry refresh.
  [[nodiscard]] const ActiveCutContextRefreshReport& attemptEntryReport()
      const noexcept
  {
    return attempt_entry_report_;
  }

private:
  std::shared_ptr<LevelSetGeometryService> service_;
  Part part_;
  Options options_;
  ActiveCutContextRefreshReport attempt_entry_report_{};
};

} // namespace level_set
} // namespace FE
} // namespace svmp

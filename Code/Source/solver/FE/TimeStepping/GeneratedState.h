/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file GeneratedState.h
 * @brief Generated state of a nonlinear solve: the synchronization points
 *        at which NewtonSolver asks its owner to regenerate state that the
 *        residual depends on (cut geometry, projected fields, extensions,
 *        lagged references), and the policy predicates of those points.
 *
 * The predicates were moved from Application/Core/ApplicationDriver.cpp with
 * bodies and environment names unchanged; the schedule of a solve is compiled
 * from the declarations of its participants (compileGeneratedStateSchedule).
 *
 * GeneratedStateRegistry composes the owners of generated state
 * (participants) into NewtonSolver's synchronization hook.  At a
 * synchronization point it runs, in this order:
 *   1. the point filter of the policy (with the outer fixed point only the
 *      outer, endpoint and restore points synchronize);
 *   2. prepare() of every participant (rank-local resets, for example of a
 *      cache at a restore point);
 *   3. the solve gates of a transient policy (line-search trials, refresh
 *      within the solve, per-step-only state);
 *   4. synchronize() of the Geometry participants, which publish the source
 *      value revisions of the geometry they generated;
 *   5. nothing more on a constraint-construction pass (the outer fixed
 *      point's constraint pass, the endpoint candidate, the restored
 *      time-step state);
 *   6. synchronize() of the DerivedField, Extension and Reference
 *      participants.
 * Participants run by role and, within a role, in registration order.  This
 * is the order of the generated-state synchronization of the Application's
 * steady and transient solves.
 */

#include "Systems/FieldRegistry.h"
#include "TimeStepping/NewtonSolver.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

/**
 * @brief Provenance token of a synchronization point, as the generated-state
 *        owner logs it and passes it to its refreshes (for example
 *        "outer_fixed_point", "endpoint_candidate", "restored_time_step").
 *
 * The tokens are part of log formats and of the cut-topology attempt
 * protocol, which matches on them; they differ from NewtonSolver's own
 * trace tokens and must not be unified with them.
 */
[[nodiscard]] const char* generatedStateSyncPointName(
    NewtonOptions::StateSynchronizationPoint point) noexcept;

/// True for the points that restore the accepted time-step state
/// (RestoredTimeStepState, RestoredProjectedTimeStepState).
[[nodiscard]] bool restoresAcceptedTimeStepGeneratedState(
    NewtonOptions::StateSynchronizationPoint point) noexcept;

/// True when a frozen algebraic level-set extension map is refreshed at
/// @p point: with the outer fixed point, and at the projected restored
/// time-step state.
[[nodiscard]] bool refreshesFrozenLevelSetExtensionAtStateSync(
    NewtonOptions::StateSynchronizationPoint point,
    bool use_external_state_fixed_point) noexcept;

/// Whether line-search trials synchronize generated state; defaults to
/// @p residual_defining_state_changes (override: SVMP_SYNC_LINE_SEARCH_TRIALS).
[[nodiscard]] bool synchronizeTransientLineSearchTrials(
    bool residual_defining_state_changes);

// ---------------------------------------------------------------------------
// Generated-state registry
// ---------------------------------------------------------------------------

/// What a participant generates; participants run in this order.
enum class GeneratedStateRole : std::uint8_t {
    Geometry,      ///< generated geometry (cut integration context)
    DerivedField,  ///< fields projected from it (for example curvature)
    Extension,     ///< extended fields (for example an advection velocity)
    Reference,     ///< lagged copies of the state (semi-implicit terms)
};

[[nodiscard]] const char* generatedStateRoleName(GeneratedStateRole role) noexcept;

enum class GeneratedStateSolve : std::uint8_t { Steady, Transient };

/// When the generated state is synchronized.
struct GeneratedStatePolicy {
    GeneratedStateSolve solve{GeneratedStateSolve::Transient};
    /// The outer fixed point regenerates the state between frozen inner
    /// Newton solves.
    bool outer_fixed_point{false};
    /// Transient without the outer fixed point: refresh at the solve's
    /// synchronization points.
    bool refresh_within_solve{false};
    /// Transient without the outer fixed point: synchronize line-search
    /// trials (synchronizeTransientLineSearchTrials).
    bool synchronize_line_search_trials{true};
    /// Transient without the outer fixed point: refresh once per step only.
    bool per_step_only{false};
};

// ---------------------------------------------------------------------------
// Selection of the generated-state schedule
// ---------------------------------------------------------------------------

/// The generated-state schedule of a solve and the Newton settings that
/// follow from it (GeneratedStatePlan::selection).
struct GeneratedStateSelection {
    GeneratedStatePolicy policy{};
    /// Some generated state is requested.
    bool generated_state{false};
    /// NewtonOptions::accepted_state_sync_invalidates_residual.
    bool accepted_state_sync_invalidates_residual{false};
    /// The cut geometry tracks its topology; with a transient outer fixed
    /// point it then provides the epochs.
    bool track_cut_topology{false};
    /// Jacobian-check geometry provenance: the synchronization regenerates
    /// geometry within the solve.
    bool jacobian_check_refreshes_geometry_within_solve{false};
    /// Jacobian-check geometry provenance: a frozen algebraic extension.
    bool jacobian_check_frozen_algebraic_extension{false};
};

/// Whether generated state is refreshed once per step only (default false;
/// override: SVMP_CUT_REFRESH_PER_STEP_ONLY).  It switches the transient
/// outer fixed point off and changes Newton's fixed point.
[[nodiscard]] bool perStepOnlyGeneratedState();

// ---------------------------------------------------------------------------
// Outer fixed-point preset
// ---------------------------------------------------------------------------

/// Overrides of the transient outer fixed-point preset; the owner reads them
/// from its input (unset: the preset's value).
struct OuterFixedPointOverrides {
    /// Pass limit; at least 1 is used.
    std::optional<int> max_passes{};
    /// 1-based pass whose update is the first relaxed one; 0 disables the
    /// relaxation, 1 relaxes every update.
    std::optional<int> relaxation_start_pass{};
    /// 1-based pass from which a two-state constraint cycle ends the
    /// attempt; 0 disables.
    std::optional<int> cycle_exit_pass{};
    /// Cut-topology restarts per step; negative follows the pass limit.
    std::optional<int> topology_restarts{};
};

/// The transient preset (decision D28): plain passes before the relaxation
/// starts, and the pass limit.
inline constexpr int kTransientOuterFixedPointPlainPasses = 12;
inline constexpr int kTransientOuterFixedPointMaxPasses = 30;
/// The steady pass limit.
inline constexpr int kSteadyOuterFixedPointMaxPasses = 12;

/**
 * @brief The outer fixed-point options of a steady or transient solve.
 *
 * Steady: at most kSteadyOuterFixedPointMaxPasses passes, no relaxation, no
 * topology restarts, no cycle exit; @p overrides are ignored.  Transient:
 * at most kTransientOuterFixedPointMaxPasses passes; the updates of the first
 * kTransientOuterFixedPointPlainPasses passes are plain and later ones use
 * the safeguarded delta-squared relaxation, so a step that converges within
 * the plain passes is bitwise identical to the plain loop; the topology
 * restart budget is the pass limit; a constraint cycle ends the attempt from
 * the first pass beyond the plain ones.  The overrides apply in that order:
 * passes, restarts, relaxation, cycle exit.  `enabled` is left false: the
 * selected schedule decides it.
 */
[[nodiscard]] NewtonOptions::ExternalStateFixedPointOptions outerFixedPointPreset(
    GeneratedStateSolve solve,
    const OuterFixedPointOverrides& overrides = {});

// ---------------------------------------------------------------------------
// Declarations of the generated state
// ---------------------------------------------------------------------------
//
// Each participant states, per solve purpose, what it reads (with the version
// of the state it reads) and what it writes (with the phase of the write).
// The reads of the residual come from a walk of the installed formulations
// (residualGeneratedStateReads).  The declarations add information; they do
// not change what the participants do.

/// The solve purpose a declaration is stated for.
enum class GeneratedStatePurpose : std::uint8_t {
    SteadySolve,
    TransientAttempt,
    GeneralizedAlphaStage,
    EndpointCandidate,
    OuterRestore,
    TimeStepRestore,
    CertificateProbe,
    Initialization,
    AcceptedStep,
};

[[nodiscard]] const char* generatedStatePurposeName(GeneratedStatePurpose purpose) noexcept;

/// Whether the synchronizations of @p purpose project the state onto the
/// regenerated constraints (raw then projected points).  The certificate
/// probe, the initialization and the accepted step do not.
[[nodiscard]] bool generatedStatePurposeProjects(GeneratedStatePurpose purpose) noexcept;

/// The version of a resource a read sees.
enum class StateVersion : std::uint8_t {
    CurrentTrial,      ///< the state at the synchronization point (raw)
    ProjectedStage,    ///< after this refresh's constraints were applied
    FrozenPass,        ///< what the last refresh installed, held through the inner solve
    AcceptedEndpoint,  ///< u_n, or the restored accepted state
    AcceptedHistory,   ///< u^{n-k} of one field, k >= 1
};

enum class ResourceKind : std::uint8_t {
    Field,              ///< the coefficients of a field
    GeneratedDomain,    ///< the generated integration domain of an interface marker
    AffineConstraints,  ///< the affine constraints of the system
    FrozenRows,         ///< rows a constraint kernel holds fixed through the inner solve
    TrialCoordinates,   ///< the coordinates of a moving domain
    AuxiliaryInput,     ///< an auxiliary input of the residual
};

struct GeneratedResource {
    ResourceKind kind{ResourceKind::Field};
    FieldId field{INVALID_FIELD_ID};
    int marker{-1};
    std::string operator_tag{};
    int steps_back{0};

    friend bool operator==(const GeneratedResource&, const GeneratedResource&) = default;
};

/// When a write happens within a synchronization or a step.
enum class WritePhase : std::uint8_t {
    ConstraintConstruction,  ///< the raw pass that builds the constraints
    ProjectedRefresh,        ///< the projected pass
    AttemptEntry,            ///< before the physics solve of an attempt
    AcceptedCommit,          ///< after an accepted step
    InnerTransaction,        ///< inside the inner solve (a moving mesh)
};

/// Whether a participant's writes select the outer fixed point.
enum class OuterEnablement : std::uint8_t {
    Inferred,             ///< the dependency graph decides
    RequiredWhenPresent,  ///< selects it when present (today's request kinds)
    FollowsSchedule,      ///< refreshed by the selected schedule, never selects it
    Never,                ///< never selects it (for example a certificate probe)
};

struct GeneratedRead {
    GeneratedResource what{};
    StateVersion version{StateVersion::CurrentTrial};
};

struct GeneratedWrite {
    GeneratedResource what{};
    WritePhase phase{WritePhase::ProjectedRefresh};
    bool residual_defining{false};        ///< changes the residual at a fixed state
    bool requires_fresh_residual{false};  ///< acceptance needs a residual assembled after it
    bool writes_state_vector{false};      ///< writes entries of the state vector
};

struct GeneratedStateDescriptor {
    std::vector<GeneratedRead> reads{};
    std::vector<GeneratedWrite> writes{};
    OuterEnablement enablement{OuterEnablement::Inferred};
    /// False: registered, but generates nothing (for example no requests).
    bool present{true};
};

/// One synchronization: the point and what the Geometry role produced.
struct GeneratedStateSync {
    NewtonOptions::StateSynchronizationPoint point{};
    /// RestoredOuterFixedPointState or RestoredTimeStepState.
    bool restoring_generated_state{false};
    /// restoresAcceptedTimeStepGeneratedState(point).
    bool restoring_time_step_state{false};
    /// Source value revisions of the generated geometry, by interface marker
    /// (written by the Geometry participants).
    std::map<int, std::uint64_t> evaluated_state_source_revisions{};
    /// Reference synchronization at TimeStepAttemptHistory, or the owner's initial
    /// reference snapshot. Every history slot holds a physical accepted solution.
    bool attempt_entry{false};
};

/**
 * @brief An owner of generated state.
 *
 * synchronize() regenerates the state from a solution state; it is called on
 * every rank at the same points and may be collective.  prepare() runs at
 * every synchronized point before the solve gates and must be rank-local.
 * invalidate() drops cached generated state (for example after a rejected
 * topology change).  One participant may provide the cut-topology epochs of
 * the outer fixed point (NewtonOptions::external_state_* hooks).
 */
class IGeneratedStateParticipant {
public:
    virtual ~IGeneratedStateParticipant() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual GeneratedStateRole role() const noexcept = 0;

    virtual void prepare(const GeneratedStateSync& /*sync*/) {}
    virtual void synchronize(const systems::SystemStateView& state,
                             GeneratedStateSync& sync) = 0;
    virtual void invalidate() {}

    /// What the participant reads and writes for @p purpose.  The default
    /// declares nothing (present = false).
    [[nodiscard]] virtual GeneratedStateDescriptor describe(
        GeneratedStatePurpose purpose) const;

    /// True when the participant provides the epoch hooks below.
    [[nodiscard]] virtual bool providesEpochs() const noexcept { return false; }
    [[nodiscard]] virtual bool epochDiscontinuity(
        NewtonOptions::StateSynchronizationPoint /*point*/) const
    {
        return false;
    }
    virtual void acknowledgeEpochDiscontinuity(
        NewtonOptions::StateSynchronizationPoint /*point*/)
    {
    }
    [[nodiscard]] virtual bool epochRevisited(
        NewtonOptions::StateSynchronizationPoint /*point*/) const
    {
        return false;
    }
    virtual void acknowledgeFrozenEpoch() {}
};

/// A participant made of callbacks, for owners that are not FE services yet.
class FunctionGeneratedStateParticipant final : public IGeneratedStateParticipant {
public:
    struct Callbacks {
        std::function<void(const GeneratedStateSync&)> prepare{};
        std::function<void(const systems::SystemStateView&, GeneratedStateSync&)>
            synchronize{};
        std::function<void()> invalidate{};
        /// Unset: the participant declares nothing.
        std::function<GeneratedStateDescriptor(GeneratedStatePurpose)> describe{};
    };

    FunctionGeneratedStateParticipant(std::string name,
                                      GeneratedStateRole role,
                                      Callbacks callbacks);

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }
    [[nodiscard]] GeneratedStateRole role() const noexcept override { return role_; }
    void prepare(const GeneratedStateSync& sync) override;
    void synchronize(const systems::SystemStateView& state,
                     GeneratedStateSync& sync) override;
    void invalidate() override;
    [[nodiscard]] GeneratedStateDescriptor describe(
        GeneratedStatePurpose purpose) const override;

private:
    std::string name_;
    GeneratedStateRole role_;
    Callbacks callbacks_;
};

/**
 * @brief The generated state of a solve, synchronized in a fixed order.
 *
 * Create with std::make_shared: bind() keeps the registry alive in the
 * Newton options it fills.
 */
class GeneratedStateRegistry
    : public std::enable_shared_from_this<GeneratedStateRegistry> {
public:
    explicit GeneratedStateRegistry(GeneratedStatePolicy policy);

    /// Adds a participant; it runs after the earlier ones of its role.
    void add(std::shared_ptr<IGeneratedStateParticipant> participant);

    [[nodiscard]] const GeneratedStatePolicy& policy() const noexcept { return policy_; }

    /// The participants in run order.
    [[nodiscard]] const std::vector<std::shared_ptr<IGeneratedStateParticipant>>&
    participants() const noexcept
    {
        return participants_;
    }

    /// Whether @p point is synchronized at all (step 1).
    [[nodiscard]] bool synchronizes(NewtonOptions::StateSynchronizationPoint point) const noexcept;

    /// Whether only the Geometry role runs at @p point (step 5).
    [[nodiscard]] bool constraintConstructionPass(
        NewtonOptions::StateSynchronizationPoint point) const noexcept;

    /// Steps 1-6 at @p point.
    void synchronize(const systems::SystemStateView& state,
                     NewtonOptions::StateSynchronizationPoint point) const;

    /// invalidate() of every participant, in run order.
    void invalidate() const;

    /// Installs synchronize() as NewtonOptions::synchronize_state and, for a
    /// transient outer fixed point with an epoch-providing participant, the
    /// four external-state epoch hooks.  At most one participant may provide
    /// epochs.
    void bind(NewtonOptions& options) const;

private:
    GeneratedStatePolicy policy_;
    std::vector<std::shared_ptr<IGeneratedStateParticipant>> participants_{};
};

/**
 * @brief The Reference participant of the shadow fields of @p system
 *        (systems::FieldBinding::lagged and FieldBinding::previous)
 *
 * At every synchronization it copies, DOF by DOF, the source field's
 * coefficients of the synchronized state into each lagged shadow.  A
 * previous(u, k) shadow holds u at step n - k on every scheme: at an attempt
 * entry (GeneratedStateSync::attempt_entry) it copies the source's value k
 * steps back (SystemStateView::u_history, or u_prev and u_prev2) and keeps
 * it; at the other synchronizations of the attempt, where a scheme may hold
 * a rate in a history slot during the stage solve (first-order
 * generalized-alpha, Newmark), it writes the kept value.  Until the first
 * attempt entry it copies the synchronized state's history.  The copy is
 * rank-local, so the shadows are the same at any rank count.  Created after
 * setup; a shadow whose DOF map differs from its source's is refused on
 * every rank.
 */
[[nodiscard]] std::shared_ptr<IGeneratedStateParticipant> makeShadowFieldParticipant(
    systems::FESystem& system);

/// The declaration of the shadow participant of @p system for @p purpose
/// (what its describe() returns): each lagged shadow reads its source as the
/// synchronization leaves it and each previous(u, k) shadow the accepted
/// value k steps back; both are written at each projected refresh and, in a
/// transient attempt, at its entry; they follow the schedule.
[[nodiscard]] GeneratedStateDescriptor shadowFieldDescriptor(
    const systems::FESystem& system, GeneratedStatePurpose purpose);

// ---------------------------------------------------------------------------
// Declarations without a participant, residual reads and the checks
// ---------------------------------------------------------------------------

/// The accepted-step level-set maintenance of @p level_set_fields (it runs
/// in TimeLoop callbacks, not in the registry): it reads the endpoint
/// candidate and the accepted baseline and writes each field at the
/// accepted commit.  Present iff @p level_set_fields is not empty.
[[nodiscard]] GeneratedStateDescriptor maintenanceGeneratedStateDescriptor(
    const std::vector<FieldId>& level_set_fields);

/// The coupled mesh motion of a moving domain: it reads the mesh
/// displacement and moves the trial coordinates inside the inner solve.  It
/// never selects the outer fixed point.  Present iff @p mesh_displacement is
/// a field.
[[nodiscard]] GeneratedStateDescriptor movingDomainGeneratedStateDescriptor(
    FieldId mesh_displacement);

/**
 * @brief The generated-state reads of the contributing residuals of
 *        @p system, from a walk of their residual trees.
 *
 * A prescribed field read by a tree (DiscreteField or StateField) is a
 * FrozenPass read of that field.  An integral on a generated domain is a
 * FrozenPass read of the domain's marker: a cut-volume integral, a boundary
 * integral on a generated active subset, and an interface or interior-facet
 * integral whose marker the system registered as a generated embedded
 * interface.  previousSolution(k) is an AcceptedHistory read.  Unknowns,
 * frozen iterates and auxiliary inputs make no generated-state read, and
 * records that do not contribute to the problem analysis (work channels,
 * diagnostics) are skipped.  The walk changes no tree, record or kernel.
 * Each read appears once, in order of first occurrence.
 */
[[nodiscard]] std::vector<GeneratedRead> residualGeneratedStateReads(
    const systems::FESystem& system);

/// A participant's or a declaration's descriptor, with its place in the
/// run order.
struct DeclaredGeneratedState {
    std::string name{};
    GeneratedStateRole role{GeneratedStateRole::Geometry};
    /// Position in the registry's run order (declarations without a
    /// participant come after the participants).
    std::size_t position{0};
    bool registry_participant{true};
    GeneratedStateDescriptor descriptor{};
};

/// The descriptors of @p registry's participants for @p purpose, in run
/// order, followed by @p extra declarations.
[[nodiscard]] std::vector<DeclaredGeneratedState> declaredGeneratedState(
    const GeneratedStateRegistry& registry,
    GeneratedStatePurpose purpose,
    std::vector<DeclaredGeneratedState> extra = {});

enum class GeneratedStateFindingKind : std::uint8_t {
    MissingProducer,
    ConflictingWriters,
    IncompatibleStageRequest,
    OrderInconsistency,
};

[[nodiscard]] const char* generatedStateFindingKindName(GeneratedStateFindingKind kind) noexcept;

struct GeneratedStateFinding {
    GeneratedStateFindingKind kind{GeneratedStateFindingKind::MissingProducer};
    /// The reader, the second writer, or "residual".
    std::string participant{};
    GeneratedResource resource{};
    std::string message{};
};

/**
 * @brief The rank-local checks of the declarations of one purpose.
 *
 * - Missing producer: a generated domain, or one of @p generated_values (for
 *   example a lagged shadow), that a contributing residual reads and no
 *   declaration writes.
 * - Conflicting writers: two writes of one resource in one phase.
 * - Incompatible stage request: a ProjectedStage read in a purpose that does
 *   not project; a FrozenPass read of a resource no declaration writes.
 * - Order consistency: a participant reads, in the raw or projected pass, a
 *   resource that a participant later in the run order writes in that pass.
 *
 * Returns the findings in that order; it refuses nothing.
 */
[[nodiscard]] std::vector<GeneratedStateFinding> checkGeneratedStateDeclarations(
    GeneratedStatePurpose purpose,
    const std::vector<DeclaredGeneratedState>& declarations,
    const std::vector<GeneratedRead>& residual_reads,
    const std::vector<GeneratedResource>& generated_values);

// ---------------------------------------------------------------------------
// Schedule compiler
// ---------------------------------------------------------------------------

/// The inputs of compileGeneratedStateSchedule besides the declarations.
struct GeneratedStateScheduleInputs {
    GeneratedStateSolve solve{GeneratedStateSolve::Transient};
    GeneratedStatePurpose purpose{GeneratedStatePurpose::TransientAttempt};
    /// The unknowns of the Newton block (one node: the block).
    std::vector<FieldId> block_fields{};
    /// SVMP_CUT_REFRESH_PER_STEP_ONLY (perStepOnlyGeneratedState); a steady
    /// solve ignores it.
    bool per_step_only{false};
    OuterFixedPointOverrides overrides{};
    /// The field convergence criteria the installs declared
    /// (FESystem::fieldConvergenceCriteria).
    std::vector<systems::FieldConvergenceCriterion> field_criteria{};
};

/// The schedule of a solve, compiled from the declarations.
struct GeneratedStatePlan {
    GeneratedStateSelection selection{};
    NewtonOptions::ExternalStateFixedPointOptions fixed_point{};
    /// NewtonOptions::field_residual_criteria: the declared criteria, in
    /// their order.
    std::vector<NewtonOptions::FieldResidualCriterion> field_criteria{};
    /// Why the outer fixed point runs: "inferred(<participants>)" for the
    /// writers of the component that regenerates the block, and
    /// "envelope(<participants>)" for the participants that select it
    /// without being in that component (accepted-step maintenance).  Empty
    /// when it does not run.
    std::vector<std::string> reasons{};
    /// The participants of the regenerating component.
    std::vector<std::string> component{};
    /// Registry participants per pass, in run order.
    std::vector<std::string> constraint_construction_order{};
    std::vector<std::string> projected_refresh_order{};
    /// FNV-1a fingerprint of the plan (purpose, participants, phases,
    /// enablement, resources, policy flags, preset values).
    std::uint64_t fingerprint{0};
};

/**
 * @brief The schedule of a solve from the declarations of its generated
 *        state and the reads of its residual.
 *
 * The Newton block is one node.  Edges: a same-pass read (a participant
 * reads, at the current trial or the projected stage, a block unknown or what
 * another participant writes in the refresh), and a prior-pass read (the
 * block reads, frozen through the inner solve, what a participant writes in
 * the refresh: a field or a generated domain the residual reads, frozen rows
 * of a kernel in the block, affine constraints).  Prior-step reads (accepted
 * commit, accepted history) make no edge.  The component of the block in the
 * same-pass and prior-pass graph regenerates the block when one of its
 * prior-pass writers needs a fresh residual and selects the loop (Inferred,
 * or RequiredWhenPresent and present).
 *
 * The outer fixed point runs when a present RequiredWhenPresent declaration
 * exists (today's selection envelope) or the component regenerates the block
 * through an Inferred writer, and a transient solve is not per-step-only.
 * Without the outer fixed point a transient solve refreshes within the solve
 * unless an extension writes frozen rows (a monolithic extension), and then
 * reassembles the residual after an accepted-state refresh; line-search
 * trials synchronize as synchronizeTransientLineSearchTrials decides.  The
 * fixed-point options are outerFixedPointPreset with `enabled` set, and the
 * field criteria are the declared ones.
 */
[[nodiscard]] GeneratedStatePlan compileGeneratedStateSchedule(
    const GeneratedStateScheduleInputs& inputs,
    const std::vector<DeclaredGeneratedState>& declarations,
    const std::vector<GeneratedRead>& residual_reads);

/// Whether two plans select the same schedule: the selection, the
/// fixed-point options and the field criteria.
[[nodiscard]] bool sameGeneratedStateSchedule(const GeneratedStatePlan& a,
                                              const GeneratedStatePlan& b);

/**
 * @brief The checks before the first collective of a solve, on the
 *        declarations of its registered participants.
 *
 * Recompiles the schedule from @p declarations, which may name generated
 * domains the plan compiled before the cut geometry was installed did not
 * know, and requires it to select the schedule of @p plan.  Refuses every
 * finding of checkGeneratedStateDeclarations (decision D66 item 1).  A local
 * failure is agreed on, so the ranks stop together.  Then requires every
 * rank of the system's communicator to hold the same plan fingerprint.
 */
void requireGeneratedStateSchedule(
    const systems::FESystem& system,
    const GeneratedStateScheduleInputs& inputs,
    const std::vector<DeclaredGeneratedState>& declarations,
    const std::vector<GeneratedRead>& residual_reads,
    const std::vector<GeneratedResource>& generated_values,
    const GeneratedStatePlan& plan);

} // namespace timestepping
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/GeneratedState.h"

#include "Analysis/FormulationRecord.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Dofs/DofHandler.h"
#include "Forms/FormExpr.h"
#include "Systems/FESystem.h"
#include "Systems/FieldBinding.h"
#include "Systems/SystemCollectives.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace timestepping {

namespace {

// The driver's environment flag reader, unchanged.
bool parseBoolEnv(const char* name, bool default_value)
{
  const char* env = std::getenv(name);
  if (!env) {
    return default_value;
  }
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (v == "1" || v == "true" || v == "on" || v == "yes") {
    return true;
  }
  if (v == "0" || v == "false" || v == "off" || v == "no") {
    return false;
  }
  return default_value;
}

} // namespace

// Moved from Application/Core/ApplicationDriver.cpp with bodies unchanged
// (stateSyncPointName there).

const char* generatedStateSyncPointName(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point) noexcept
{
  using Point = svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  switch (point) {
  case Point::OuterFixedPointState:
    return "outer_fixed_point";
  case Point::ProjectedOuterFixedPointState:
    return "projected_outer_fixed_point";
  case Point::EndpointCandidateState:
    return "endpoint_candidate";
  case Point::ProjectedEndpointCandidateState:
    return "projected_endpoint_candidate";
  case Point::AcceptedNonlinearState:
    return "accepted";
  case Point::ResidualAssembly:
    return "residual";
  case Point::JacobianAssembly:
    return "jacobian";
  case Point::JacobianAndResidualAssembly:
    return "jacobian_and_residual";
  case Point::LineSearchTrialResidual:
    return "line_search_trial";
  case Point::RestoredNonlinearState:
    return "restored";
  case Point::RestoredOuterFixedPointState:
    return "restored_outer_fixed_point";
  case Point::RestoredProjectedOuterFixedPointState:
    return "restored_projected_outer_fixed_point";
  case Point::RestoredTimeStepState:
    return "restored_time_step";
  case Point::RestoredProjectedTimeStepState:
    return "restored_projected_time_step";
  case Point::FinalResidualAssembly:
    return "final_residual";
  }
  return "unknown";
}

bool restoresAcceptedTimeStepGeneratedState(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
    noexcept
{
  using Point =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  return point == Point::RestoredTimeStepState ||
         point == Point::RestoredProjectedTimeStepState;
}

bool refreshesFrozenLevelSetExtensionAtStateSync(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point,
    bool use_external_state_fixed_point) noexcept
{
  using Point =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  return use_external_state_fixed_point ||
         point == Point::RestoredProjectedTimeStepState;
}

bool synchronizeTransientLineSearchTrials(
    bool residual_defining_state_changes)
{
  // R(u, G(u)) is the nonlinear residual when generated cuts, projected
  // curvature, velocity extension, or state-dependent constraints are part of
  // the problem definition.  Backtracking must evaluate that deterministic
  // residual at each trial.  Freezing G at the previous iterate instead
  // globalizes a different Picard surrogate and can accept a step that
  // increases the actual residual.  Keep an explicit opt-out for controlled
  // legacy comparisons.
  return parseBoolEnv("SVMP_SYNC_LINE_SEARCH_TRIALS",
                      residual_defining_state_changes);
}

// ---------------------------------------------------------------------------
// Selection of the generated-state schedule
// ---------------------------------------------------------------------------

// Moved from Application/Core/ApplicationDriver.cpp (runSteady and
// runTransient) with the predicates unchanged.

bool perStepOnlyGeneratedState()
{
  return parseBoolEnv("SVMP_CUT_REFRESH_PER_STEP_ONLY", false);
}

GeneratedStateSelection legacyGeneratedStateSelection(
    const GeneratedStateRequests& requests,
    GeneratedStateSolve solve,
    bool per_step_only)
{
  GeneratedStateSelection selection;
  selection.policy.solve = solve;
  selection.generated_state =
      requests.cut_geometry || requests.maintenance || requests.extension;
  if (solve == GeneratedStateSolve::Steady) {
    // Generated geometry uses an outer Picard loop so each inner Newton
    // residual and Jacobian share exactly one frozen G.
    const bool use_steady_external_state_fixed_point = selection.generated_state;
    selection.policy.outer_fixed_point = use_steady_external_state_fixed_point;
    selection.accepted_state_sync_invalidates_residual =
        selection.generated_state && !use_steady_external_state_fixed_point;
    selection.jacobian_check_refreshes_geometry_within_solve =
        !use_steady_external_state_fixed_point;
    selection.jacobian_check_frozen_algebraic_extension = false;
    return selection;
  }
  // Generated cut rules, active-set constraints, and the wet-extension map
  // form one residual-defining package G(u).  The assembled tangent
  // differentiates R(u,G) at fixed G; it does not contain every shape,
  // projection, or active-set derivative.  Solve the consistent nested
  // problem with outer regeneration and inner frozen-geometry Newton solves.
  const bool use_transient_external_state_fixed_point =
      selection.generated_state && !per_step_only;
  const bool refresh_generated_geometry_within_solve =
      selection.generated_state && !use_transient_external_state_fixed_point &&
      !requests.monolithic_extension;
  // When generated data is refreshed at accepted iterates, reassemble the
  // residual after that refresh before deciding that Newton has converged.
  selection.accepted_state_sync_invalidates_residual =
      refresh_generated_geometry_within_solve && selection.generated_state;
  selection.policy.outer_fixed_point = use_transient_external_state_fixed_point;
  selection.policy.refresh_within_solve = refresh_generated_geometry_within_solve;
  selection.policy.synchronize_line_search_trials =
      synchronizeTransientLineSearchTrials(
          selection.accepted_state_sync_invalidates_residual);
  selection.policy.per_step_only = per_step_only;
  selection.track_cut_topology = requests.cut_geometry;
  selection.jacobian_check_refreshes_geometry_within_solve =
      refresh_generated_geometry_within_solve;
  selection.jacobian_check_frozen_algebraic_extension =
      requests.monolithic_extension;
  return selection;
}

// ---------------------------------------------------------------------------
// Outer fixed-point preset
// ---------------------------------------------------------------------------

// Moved from Application/Core/ApplicationDriver.cpp (runSteady, and the
// outer fixed-point block of runTransient) with the order of the overrides
// unchanged.

NewtonOptions::ExternalStateFixedPointOptions outerFixedPointPreset(
    GeneratedStateSolve solve,
    const OuterFixedPointOverrides& overrides)
{
  NewtonOptions::ExternalStateFixedPointOptions options;
  if (solve == GeneratedStateSolve::Steady) {
    options.max_iterations = kSteadyOuterFixedPointMaxPasses;
    return options;
  }
  // At most kTransientOuterFixedPointMaxPasses refreshes; the first
  // kTransientOuterFixedPointPlainPasses are plain and the updates from then
  // on use the safeguarded delta-squared (Aitken) relaxation.  A step that
  // converges within the plain passes is bitwise identical to the plain loop
  // with that cap; only steps that would have hit it continue, relaxed.
  options.max_iterations = kTransientOuterFixedPointMaxPasses;
  options.dynamic_relaxation.enabled = true;
  options.dynamic_relaxation.start_iteration =
      kTransientOuterFixedPointPlainPasses - 1;
  if (overrides.max_passes.has_value()) {
    options.max_iterations = std::max(1, *overrides.max_passes);
  }
  // A cut-topology change during a step is a normal event for a moving
  // interface: each outer fixed-point refresh may regenerate a new topology,
  // and the solve continues on it.  The number of such epoch changes is
  // therefore bounded by the outer iteration limit, which is the default
  // (0 restores the stop-and-reject behavior).
  options.max_discontinuity_restarts = options.max_iterations;
  if (overrides.topology_restarts.has_value() && *overrides.topology_restarts >= 0) {
    options.max_discontinuity_restarts = *overrides.topology_restarts;
  }
  if (overrides.relaxation_start_pass.has_value()) {
    options.dynamic_relaxation.enabled = *overrides.relaxation_start_pass > 0;
    options.dynamic_relaxation.start_iteration =
        std::max(0, *overrides.relaxation_start_pass - 1);
  }
  // Exit from a persistent two-state constraint cycle (see NewtonOptions::
  // ExternalStateFixedPointOptions::constraint_cycle_exit_start_iteration):
  // from the first refresh beyond the plain passes, so a step that converges
  // within them is unaffected.
  int exit_pass = kTransientOuterFixedPointPlainPasses + 1;
  if (overrides.cycle_exit_pass.has_value()) {
    exit_pass = *overrides.cycle_exit_pass;
  }
  options.constraint_cycle_exit_start_iteration = exit_pass > 0 ? exit_pass - 1 : -1;
  return options;
}

// ---------------------------------------------------------------------------
// Generated-state registry
// ---------------------------------------------------------------------------

const char* generatedStateRoleName(GeneratedStateRole role) noexcept
{
  switch (role) {
  case GeneratedStateRole::Geometry:
    return "geometry";
  case GeneratedStateRole::DerivedField:
    return "derived_field";
  case GeneratedStateRole::Extension:
    return "extension";
  case GeneratedStateRole::Reference:
    return "reference";
  }
  return "unknown";
}

FunctionGeneratedStateParticipant::FunctionGeneratedStateParticipant(
    std::string name,
    GeneratedStateRole role,
    Callbacks callbacks)
    : name_(std::move(name)), role_(role), callbacks_(std::move(callbacks))
{
}

void FunctionGeneratedStateParticipant::prepare(const GeneratedStateSync& sync)
{
  if (callbacks_.prepare) {
    callbacks_.prepare(sync);
  }
}

void FunctionGeneratedStateParticipant::synchronize(
    const systems::SystemStateView& state,
    GeneratedStateSync& sync)
{
  if (callbacks_.synchronize) {
    callbacks_.synchronize(state, sync);
  }
}

void FunctionGeneratedStateParticipant::invalidate()
{
  if (callbacks_.invalidate) {
    callbacks_.invalidate();
  }
}

GeneratedStateDescriptor FunctionGeneratedStateParticipant::describe(
    GeneratedStatePurpose purpose) const
{
  if (callbacks_.describe) {
    return callbacks_.describe(purpose);
  }
  return IGeneratedStateParticipant::describe(purpose);
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

const char* generatedStatePurposeName(GeneratedStatePurpose purpose) noexcept
{
  switch (purpose) {
  case GeneratedStatePurpose::SteadySolve:
    return "steady_solve";
  case GeneratedStatePurpose::TransientAttempt:
    return "transient_attempt";
  case GeneratedStatePurpose::GeneralizedAlphaStage:
    return "generalized_alpha_stage";
  case GeneratedStatePurpose::EndpointCandidate:
    return "endpoint_candidate";
  case GeneratedStatePurpose::OuterRestore:
    return "outer_restore";
  case GeneratedStatePurpose::TimeStepRestore:
    return "time_step_restore";
  case GeneratedStatePurpose::CertificateProbe:
    return "certificate_probe";
  case GeneratedStatePurpose::Initialization:
    return "initialization";
  case GeneratedStatePurpose::AcceptedStep:
    return "accepted_step";
  }
  return "unknown";
}

bool generatedStatePurposeProjects(GeneratedStatePurpose purpose) noexcept
{
  switch (purpose) {
  case GeneratedStatePurpose::CertificateProbe:
  case GeneratedStatePurpose::Initialization:
  case GeneratedStatePurpose::AcceptedStep:
    return false;
  default:
    return true;
  }
}

GeneratedStateDescriptor IGeneratedStateParticipant::describe(
    GeneratedStatePurpose /*purpose*/) const
{
  GeneratedStateDescriptor descriptor;
  descriptor.present = false;
  return descriptor;
}

GeneratedStateRegistry::GeneratedStateRegistry(GeneratedStatePolicy policy)
    : policy_(policy)
{
}

void GeneratedStateRegistry::add(
    std::shared_ptr<IGeneratedStateParticipant> participant)
{
  if (!participant) {
    throw std::invalid_argument(
        "GeneratedStateRegistry::add: null participant");
  }
  const auto role = participant->role();
  const auto position = std::find_if(
      participants_.begin(), participants_.end(), [role](const auto& entry) {
        return static_cast<int>(entry->role()) > static_cast<int>(role);
      });
  participants_.insert(position, std::move(participant));
}

bool GeneratedStateRegistry::synchronizes(
    NewtonOptions::StateSynchronizationPoint point) const noexcept
{
  using Point = NewtonOptions::StateSynchronizationPoint;
  if (!policy_.outer_fixed_point) {
    return true;
  }
  const bool outer = point == Point::OuterFixedPointState ||
                     point == Point::ProjectedOuterFixedPointState ||
                     point == Point::RestoredOuterFixedPointState ||
                     point == Point::RestoredProjectedOuterFixedPointState;
  if (policy_.solve == GeneratedStateSolve::Steady) {
    return outer;
  }
  return outer || point == Point::EndpointCandidateState ||
         point == Point::ProjectedEndpointCandidateState ||
         point == Point::RestoredTimeStepState ||
         point == Point::RestoredProjectedTimeStepState;
}

bool GeneratedStateRegistry::constraintConstructionPass(
    NewtonOptions::StateSynchronizationPoint point) const noexcept
{
  using Point = NewtonOptions::StateSynchronizationPoint;
  const bool outer_pass =
      policy_.outer_fixed_point &&
      (point == Point::OuterFixedPointState ||
       point == Point::RestoredOuterFixedPointState);
  if (policy_.solve == GeneratedStateSolve::Steady) {
    return outer_pass;
  }
  return outer_pass || point == Point::EndpointCandidateState ||
         point == Point::RestoredTimeStepState;
}

void GeneratedStateRegistry::synchronize(
    const systems::SystemStateView& state,
    NewtonOptions::StateSynchronizationPoint point) const
{
  using Point = NewtonOptions::StateSynchronizationPoint;
  // 1. point filter
  if (!synchronizes(point)) {
    return;
  }
  GeneratedStateSync sync;
  sync.point = point;
  sync.restoring_generated_state =
      point == Point::RestoredOuterFixedPointState ||
      point == Point::RestoredTimeStepState;
  sync.restoring_time_step_state =
      restoresAcceptedTimeStepGeneratedState(point);
  // 2. rank-local preparation
  for (const auto& participant : participants_) {
    participant->prepare(sync);
  }
  // 3. solve gates of a transient solve without the outer fixed point
  if (policy_.solve == GeneratedStateSolve::Transient &&
      !policy_.outer_fixed_point) {
    // The Jacobian may remain a refreshed-frozen/quasi-Newton tangent, but
    // line search globalizes the actual residual R(u,G(u)).  An explicit
    // environment override can freeze trials for a controlled legacy
    // comparison; it is not the production default when G changes the
    // residual.
    if (point == Point::LineSearchTrialResidual &&
        !policy_.synchronize_line_search_trials) {
      return;
    }
    if (!sync.restoring_time_step_state && !policy_.refresh_within_solve) {
      return;
    }
    // Refresh-cadence experiment knob: freeze the cut geometry (and the
    // φ-derived curvature/advection data) across ALL within-solve sync
    // points, so the integration domains move once per accepted STEP
    // (before_physics_solve + accepted_step callbacks) instead of after
    // every accepted Newton iterate. CAUTION: this changes Newton's
    // fixed point — the converged quadrature lags the converged φ by
    // one solve.
    if (!sync.restoring_time_step_state && policy_.per_step_only) {
      return;
    }
  }
  // 4. geometry
  for (const auto& participant : participants_) {
    if (participant->role() == GeneratedStateRole::Geometry) {
      participant->synchronize(state, sync);
    }
  }
  // 5. constraint-construction pass
  if (constraintConstructionPass(point)) {
    return;
  }
  // 6. derived fields, extensions and references
  for (const auto& participant : participants_) {
    if (participant->role() != GeneratedStateRole::Geometry) {
      participant->synchronize(state, sync);
    }
  }
}

void GeneratedStateRegistry::invalidate() const
{
  for (const auto& participant : participants_) {
    participant->invalidate();
  }
}

void GeneratedStateRegistry::bind(NewtonOptions& options) const
{
  auto self = shared_from_this();
  options.synchronize_state =
      [self](const systems::SystemStateView& state,
             NewtonOptions::StateSynchronizationPoint point) {
        self->synchronize(state, point);
      };
  std::shared_ptr<IGeneratedStateParticipant> provider;
  for (const auto& participant : participants_) {
    if (!participant->providesEpochs()) {
      continue;
    }
    if (provider) {
      throw std::logic_error(
          "GeneratedStateRegistry::bind: more than one participant provides "
          "generated-state epochs");
    }
    provider = participant;
  }
  if (!provider || policy_.solve != GeneratedStateSolve::Transient ||
      !policy_.outer_fixed_point) {
    return;
  }
  options.external_state_discontinuity =
      [provider](NewtonOptions::StateSynchronizationPoint point) {
        return provider->epochDiscontinuity(point);
      };
  options.acknowledge_external_state_discontinuity =
      [provider](NewtonOptions::StateSynchronizationPoint point) {
        provider->acknowledgeEpochDiscontinuity(point);
      };
  options.external_state_epoch_revisited =
      [provider](NewtonOptions::StateSynchronizationPoint point) {
        return provider->epochRevisited(point);
      };
  options.acknowledge_external_state_frozen_epoch = [provider]() {
    provider->acknowledgeFrozenEpoch();
  };
}

// ---------------------------------------------------------------------------
// Shadow fields
// ---------------------------------------------------------------------------

namespace {

/// One shadow field's copy: the source's field DOFs, their indices in the
/// state vector and the shadow's coefficient count.
struct ShadowCopy {
  systems::ShadowField field{};
  std::vector<GlobalIndex> field_dofs{};
  std::vector<GlobalIndex> state_dofs{};
  GlobalIndex coefficient_count{0};
};

/// The values of @p state at @p indices; @p vector, when set, is read
/// through its assembly view, as assembly reads it (ghosted and permuted
/// layouts resolve as for the unknowns).
void readState(std::span<const Real> values,
               const backends::GenericVector* vector,
               std::span<const GlobalIndex> indices,
               std::span<Real> out,
               std::string_view what)
{
  if (vector != nullptr) {
    const auto view = const_cast<backends::GenericVector*>(vector)->createAssemblyView();
    if (!view) {
      throw std::runtime_error("ShadowFieldParticipant: cannot read the " + std::string(what));
    }
    view->getVectorEntries(indices, out);
    return;
  }
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const auto index = static_cast<std::size_t>(indices[i]);
    if (index >= values.size()) {
      throw std::runtime_error("ShadowFieldParticipant: the " + std::string(what) +
                               " is smaller than the FE layout");
    }
    out[i] = values[index];
  }
}

class ShadowFieldParticipant final : public IGeneratedStateParticipant {
public:
  explicit ShadowFieldParticipant(systems::FESystem& system) : system_(&system)
  {
    std::exception_ptr failure;
    try {
      for (const auto& field : systems::shadowFields(system)) {
        copies_.push_back(prepare(field));
      }
    } catch (...) {
      failure = std::current_exception();
    }
    (void)systems::agreeOnLocalPhase(system, failure, /*local_available=*/true,
                                     "shadow_field_layout", "ShadowFieldParticipant");
  }

  [[nodiscard]] std::string_view name() const noexcept override { return "shadow_fields"; }
  [[nodiscard]] GeneratedStateRole role() const noexcept override { return GeneratedStateRole::Reference; }

  // A lagged copy reads its source as the synchronization left it and a
  // previous(u, k) copy the accepted value k steps back; both are refreshed
  // by the selected schedule and never select it.
  [[nodiscard]] GeneratedStateDescriptor describe(GeneratedStatePurpose purpose) const override
  {
    GeneratedStateDescriptor descriptor;
    descriptor.enablement = OuterEnablement::FollowsSchedule;
    descriptor.present = !copies_.empty();
    const bool projects = generatedStatePurposeProjects(purpose);
    for (const auto& copy : copies_) {
      GeneratedResource source{.kind = ResourceKind::Field, .field = copy.field.source};
      if (copy.field.kind == systems::ShadowField::Kind::Previous) {
        source.steps_back = copy.field.steps_back;
        descriptor.reads.push_back({.what = source, .version = StateVersion::AcceptedHistory});
      } else {
        descriptor.reads.push_back(
            {.what = source,
             .version = projects ? StateVersion::ProjectedStage : StateVersion::CurrentTrial});
      }
      const GeneratedResource shadow{.kind = ResourceKind::Field, .field = copy.field.shadow};
      const auto write = [&](WritePhase phase) {
        descriptor.writes.push_back({.what = shadow,
                                     .phase = phase,
                                     .residual_defining = true,
                                     .requires_fresh_residual = true});
      };
      if (projects) {
        write(WritePhase::ProjectedRefresh);
      }
      if (purpose == GeneratedStatePurpose::TransientAttempt) {
        write(WritePhase::AttemptEntry);
      }
    }
    return descriptor;
  }

  void synchronize(const systems::SystemStateView& state, GeneratedStateSync& /*sync*/) override
  {
    for (const auto& copy : copies_) {
      std::span<const Real> values = state.u;
      const backends::GenericVector* vector = state.u_vector;
      if (copy.field.kind == systems::ShadowField::Kind::Previous) {
        values = historyValues(state, copy.field.steps_back);
        vector = copy.field.steps_back == 1 ? state.u_prev_vector
                 : copy.field.steps_back == 2 ? state.u_prev2_vector
                                              : nullptr;
        if (vector == nullptr && values.empty()) {
          throw std::runtime_error("ShadowFieldParticipant: the state holds no value " +
                                   std::to_string(copy.field.steps_back) + " steps back for '" +
                                   system_->fieldRecord(copy.field.source).name + "'");
        }
      }
      std::vector<Real> read(copy.state_dofs.size(), Real(0.0));
      readState(values, vector, copy.state_dofs, read, "state of '" + system_->fieldRecord(copy.field.source).name + "'");
      std::vector<Real> coefficients(static_cast<std::size_t>(copy.coefficient_count), Real(0.0));
      for (std::size_t i = 0; i < copy.field_dofs.size(); ++i) {
        coefficients[static_cast<std::size_t>(copy.field_dofs[i])] = read[i];
      }
      system_->setPrescribedFieldCoefficients(copy.field.shadow, coefficients);
    }
  }

private:
  [[nodiscard]] ShadowCopy prepare(const systems::ShadowField& field) const
  {
    // A copy coefficient by coefficient needs the source's DOF map.
    const auto& source_map = system_->fieldDofHandler(field.source).getDofMap();
    const auto& shadow_map = system_->fieldDofHandler(field.shadow).getDofMap();
    const auto source_offsets = source_map.getOffsets();
    const auto shadow_offsets = shadow_map.getOffsets();
    const auto source_indices = source_map.getDofIndices();
    const auto shadow_indices = shadow_map.getDofIndices();
    if (source_map.getNumDofs() != shadow_map.getNumDofs() ||
        !std::equal(source_offsets.begin(), source_offsets.end(), shadow_offsets.begin(), shadow_offsets.end()) ||
        !std::equal(source_indices.begin(), source_indices.end(), shadow_indices.begin(), shadow_indices.end())) {
      throw std::runtime_error("ShadowFieldParticipant: the shadow field '" + system_->fieldRecord(field.shadow).name +
                               "' does not share the DOF map of '" + system_->fieldRecord(field.source).name + "'");
    }
    ShadowCopy copy;
    copy.field = field;
    copy.coefficient_count = shadow_map.getNumDofs();
    copy.field_dofs.assign(source_indices.begin(), source_indices.end());
    std::sort(copy.field_dofs.begin(), copy.field_dofs.end());
    copy.field_dofs.erase(std::unique(copy.field_dofs.begin(), copy.field_dofs.end()), copy.field_dofs.end());
    const auto offset = system_->fieldDofOffset(field.source);
    copy.state_dofs.reserve(copy.field_dofs.size());
    for (const auto dof : copy.field_dofs) {
      if (dof < 0 || dof >= copy.coefficient_count) {
        throw std::runtime_error("ShadowFieldParticipant: a DOF of '" + system_->fieldRecord(field.source).name +
                                 "' lies outside its field layout");
      }
      copy.state_dofs.push_back(offset + dof);
    }
    return copy;
  }

  [[nodiscard]] static std::span<const Real> historyValues(const systems::SystemStateView& state, int k)
  {
    if (k >= 1 && static_cast<std::size_t>(k) <= state.u_history.size()) {
      return state.u_history[static_cast<std::size_t>(k - 1)];
    }
    if (k == 1) {
      return state.u_prev;
    }
    if (k == 2) {
      return state.u_prev2;
    }
    return {};
  }

  systems::FESystem* system_{nullptr};
  std::vector<ShadowCopy> copies_{};
};

} // namespace

std::shared_ptr<IGeneratedStateParticipant> makeShadowFieldParticipant(systems::FESystem& system)
{
  return std::make_shared<ShadowFieldParticipant>(system);
}

// ---------------------------------------------------------------------------
// Declarations without a participant, residual reads and the checks
// ---------------------------------------------------------------------------

GeneratedStateDescriptor maintenanceGeneratedStateDescriptor(
    const std::vector<FieldId>& level_set_fields)
{
  GeneratedStateDescriptor descriptor;
  descriptor.enablement = OuterEnablement::RequiredWhenPresent;
  descriptor.present = !level_set_fields.empty();
  for (const auto field : level_set_fields) {
    const GeneratedResource phi{.kind = ResourceKind::Field, .field = field};
    descriptor.reads.push_back({.what = phi, .version = StateVersion::CurrentTrial});
    descriptor.reads.push_back({.what = phi, .version = StateVersion::AcceptedEndpoint});
    descriptor.writes.push_back({.what = phi,
                                 .phase = WritePhase::AcceptedCommit,
                                 .residual_defining = false,
                                 .requires_fresh_residual = false,
                                 .writes_state_vector = true});
  }
  return descriptor;
}

GeneratedStateDescriptor movingDomainGeneratedStateDescriptor(FieldId mesh_displacement)
{
  GeneratedStateDescriptor descriptor;
  descriptor.enablement = OuterEnablement::Never;
  descriptor.present = mesh_displacement != INVALID_FIELD_ID;
  if (descriptor.present) {
    descriptor.reads.push_back(
        {.what = {.kind = ResourceKind::Field, .field = mesh_displacement},
         .version = StateVersion::CurrentTrial});
    descriptor.writes.push_back({.what = {.kind = ResourceKind::TrialCoordinates},
                                 .phase = WritePhase::InnerTransaction,
                                 .residual_defining = true,
                                 .requires_fresh_residual = false});
  }
  return descriptor;
}

namespace {

void appendRead(std::vector<GeneratedRead>& reads, GeneratedRead read)
{
  for (const auto& existing : reads) {
    if (existing.what == read.what && existing.version == read.version) {
      return;
    }
  }
  reads.push_back(std::move(read));
}

GeneratedRead generatedDomainRead(int marker)
{
  return {.what = {.kind = ResourceKind::GeneratedDomain, .marker = marker},
          .version = StateVersion::FrozenPass};
}

void collectResidualReads(const systems::FESystem& system,
                          const forms::FormExprNode& node,
                          std::vector<GeneratedRead>& reads)
{
  using FT = forms::FormExprType;
  switch (node.type()) {
  case FT::DiscreteField:
  case FT::StateField: {
    // A DiscreteField of an unknown is a frozen iterate and a StateField of
    // an unknown is the block itself: neither is generated state.
    const auto field = node.fieldId();
    if (field && *field != INVALID_FIELD_ID &&
        system.fieldRecord(*field).source_kind == systems::FieldSourceKind::PrescribedData) {
      appendRead(reads, {.what = {.kind = ResourceKind::Field, .field = *field},
                         .version = StateVersion::FrozenPass});
    }
    break;
  }
  case FT::PreviousSolutionRef: {
    const auto k = node.historyIndex();
    appendRead(reads, {.what = {.kind = ResourceKind::Field,
                                .field = node.fieldId().value_or(INVALID_FIELD_ID),
                                .steps_back = k.value_or(0)},
                       .version = StateVersion::AcceptedHistory});
    break;
  }
  case FT::CutVolumeIntegral: {
    const auto marker = node.interfaceMarker();
    if (marker && *marker >= 0) {
      appendRead(reads, generatedDomainRead(*marker));
    }
    break;
  }
  case FT::InterfaceIntegral:
  case FT::InteriorFaceIntegral: {
    const auto* exterior = node.exteriorBoundaryMeasure();
    if (exterior != nullptr && exterior->isGeneratedActiveSubset()) {
      appendRead(reads, generatedDomainRead(exterior->generatedActiveBoundaryMarker()));
      break;
    }
    const auto marker = node.interfaceMarker();
    if (marker && *marker >= 0 &&
        system.isGeneratedEmbeddedInterfaceMarkerRegistered(static_cast<systems::InterfaceId>(*marker))) {
      appendRead(reads, generatedDomainRead(*marker));
    }
    break;
  }
  case FT::BoundaryIntegral: {
    const auto* exterior = node.exteriorBoundaryMeasure();
    if (exterior != nullptr && exterior->isGeneratedActiveSubset()) {
      appendRead(reads, generatedDomainRead(exterior->generatedActiveBoundaryMarker()));
    }
    break;
  }
  default:
    break;
  }
  for (const auto* child : node.children()) {
    if (child != nullptr) {
      collectResidualReads(system, *child, reads);
    }
  }
}

} // namespace

std::vector<GeneratedRead> residualGeneratedStateReads(const systems::FESystem& system)
{
  std::vector<GeneratedRead> reads;
  for (const auto& record : system.formulationRecords()) {
    if (!record.contributes_to_problem_analysis || !record.residual_expr) {
      continue;
    }
    collectResidualReads(system, *record.residual_expr, reads);
  }
  return reads;
}

std::vector<DeclaredGeneratedState> declaredGeneratedState(
    const GeneratedStateRegistry& registry,
    GeneratedStatePurpose purpose,
    std::vector<DeclaredGeneratedState> extra)
{
  std::vector<DeclaredGeneratedState> declarations;
  std::size_t position = 0;
  for (const auto& participant : registry.participants()) {
    declarations.push_back({.name = std::string(participant->name()),
                            .role = participant->role(),
                            .position = position++,
                            .registry_participant = true,
                            .descriptor = participant->describe(purpose)});
  }
  for (auto& declaration : extra) {
    declaration.position = position++;
    declaration.registry_participant = false;
    declarations.push_back(std::move(declaration));
  }
  return declarations;
}

const char* generatedStateFindingKindName(GeneratedStateFindingKind kind) noexcept
{
  switch (kind) {
  case GeneratedStateFindingKind::MissingProducer:
    return "missing_producer";
  case GeneratedStateFindingKind::ConflictingWriters:
    return "conflicting_writers";
  case GeneratedStateFindingKind::IncompatibleStageRequest:
    return "incompatible_stage_request";
  case GeneratedStateFindingKind::OrderInconsistency:
    return "order_inconsistency";
  }
  return "unknown";
}

namespace {

std::string describeResource(const GeneratedResource& resource)
{
  switch (resource.kind) {
  case ResourceKind::Field:
    return "field " + std::to_string(resource.field) +
           (resource.steps_back > 0 ? " (" + std::to_string(resource.steps_back) + " steps back)" : "");
  case ResourceKind::GeneratedDomain:
    return "generated domain " + std::to_string(resource.marker);
  case ResourceKind::AffineConstraints:
    return "affine constraints";
  case ResourceKind::FrozenRows:
    return "frozen rows of field " + std::to_string(resource.field) + " in operator '" +
           resource.operator_tag + "'";
  case ResourceKind::TrialCoordinates:
    return "trial coordinates";
  case ResourceKind::AuxiliaryInput:
    return "auxiliary input";
  }
  return "resource";
}

bool writtenByAny(const std::vector<DeclaredGeneratedState>& declarations,
                  const GeneratedResource& resource)
{
  for (const auto& declaration : declarations) {
    for (const auto& write : declaration.descriptor.writes) {
      if (write.what == resource) {
        return true;
      }
    }
  }
  return false;
}

} // namespace

std::vector<GeneratedStateFinding> checkGeneratedStateDeclarations(
    GeneratedStatePurpose purpose,
    const std::vector<DeclaredGeneratedState>& declarations,
    const std::vector<GeneratedRead>& residual_reads,
    const std::vector<GeneratedResource>& generated_values)
{
  std::vector<GeneratedStateFinding> findings;
  const std::string purpose_name = generatedStatePurposeName(purpose);
  // Missing producer.
  for (const auto& read : residual_reads) {
    if (read.version != StateVersion::FrozenPass) {
      continue;
    }
    const bool generated =
        read.what.kind == ResourceKind::GeneratedDomain ||
        std::find(generated_values.begin(), generated_values.end(), read.what) !=
            generated_values.end();
    if (generated && !writtenByAny(declarations, read.what)) {
      findings.push_back({.kind = GeneratedStateFindingKind::MissingProducer,
                          .participant = "residual",
                          .resource = read.what,
                          .message = "the residual reads " + describeResource(read.what) +
                                     ", which no participant refreshes in the " + purpose_name});
    }
  }
  // Conflicting writers.
  for (std::size_t i = 0; i < declarations.size(); ++i) {
    const auto& writes_i = declarations[i].descriptor.writes;
    for (std::size_t a = 0; a < writes_i.size(); ++a) {
      for (std::size_t j = i; j < declarations.size(); ++j) {
        const auto& writes_j = declarations[j].descriptor.writes;
        for (std::size_t b = (j == i ? a + 1 : 0); b < writes_j.size(); ++b) {
          if (writes_i[a].what == writes_j[b].what && writes_i[a].phase == writes_j[b].phase) {
            findings.push_back({.kind = GeneratedStateFindingKind::ConflictingWriters,
                                .participant = declarations[j].name,
                                .resource = writes_j[b].what,
                                .message = "'" + declarations[i].name + "' and '" +
                                           declarations[j].name + "' both write " +
                                           describeResource(writes_j[b].what) +
                                           " in one phase of the " + purpose_name});
          }
        }
      }
    }
  }
  // Incompatible stage requests.
  const bool projects = generatedStatePurposeProjects(purpose);
  for (const auto& declaration : declarations) {
    for (const auto& read : declaration.descriptor.reads) {
      if (read.version == StateVersion::ProjectedStage && !projects) {
        findings.push_back({.kind = GeneratedStateFindingKind::IncompatibleStageRequest,
                            .participant = declaration.name,
                            .resource = read.what,
                            .message = "'" + declaration.name + "' reads the projected " +
                                       describeResource(read.what) + ", but the " + purpose_name +
                                       " has no projection step"});
      } else if (read.version == StateVersion::FrozenPass &&
                 !writtenByAny(declarations, read.what)) {
        findings.push_back({.kind = GeneratedStateFindingKind::IncompatibleStageRequest,
                            .participant = declaration.name,
                            .resource = read.what,
                            .message = "'" + declaration.name + "' reads the frozen " +
                                       describeResource(read.what) + ", which nothing writes in the " +
                                       purpose_name});
      }
    }
  }
  // Order consistency: within the raw and the projected pass, a writer runs
  // before every participant that reads what it writes.
  for (const auto& reader : declarations) {
    if (!reader.registry_participant) {
      continue;
    }
    for (const auto& read : reader.descriptor.reads) {
      WritePhase pass{};
      if (read.version == StateVersion::CurrentTrial) {
        pass = WritePhase::ConstraintConstruction;
      } else if (read.version == StateVersion::ProjectedStage) {
        pass = WritePhase::ProjectedRefresh;
      } else {
        continue;
      }
      for (const auto& writer : declarations) {
        if (!writer.registry_participant || writer.position <= reader.position) {
          continue;
        }
        for (const auto& write : writer.descriptor.writes) {
          if (write.phase == pass && write.what == read.what) {
            findings.push_back({.kind = GeneratedStateFindingKind::OrderInconsistency,
                                .participant = reader.name,
                                .resource = read.what,
                                .message = "'" + reader.name + "' reads " + describeResource(read.what) +
                                           " before '" + writer.name + "' writes it in the same pass"});
          }
        }
      }
    }
  }
  return findings;
}

// ---------------------------------------------------------------------------
// Schedule compiler
// ---------------------------------------------------------------------------

namespace {

constexpr std::size_t kBlockNode = 0;

bool refreshPhase(WritePhase phase) noexcept
{
  return phase == WritePhase::ConstraintConstruction || phase == WritePhase::ProjectedRefresh;
}

bool sameRefreshVersion(StateVersion version) noexcept
{
  return version == StateVersion::CurrentTrial || version == StateVersion::ProjectedStage;
}

// The strongly connected component of every node (Tarjan, iterative).
std::vector<std::size_t> stronglyConnectedComponents(
    const std::vector<std::vector<std::size_t>>& adjacency)
{
  const std::size_t n = adjacency.size();
  constexpr std::size_t kUnvisited = static_cast<std::size_t>(-1);
  std::vector<std::size_t> index(n, kUnvisited), low(n, 0), component(n, kUnvisited);
  std::vector<bool> on_stack(n, false);
  std::vector<std::size_t> stack;
  std::size_t next_index = 0;
  std::size_t next_component = 0;
  for (std::size_t root = 0; root < n; ++root) {
    if (index[root] != kUnvisited) {
      continue;
    }
    // (node, next edge position)
    std::vector<std::pair<std::size_t, std::size_t>> work{{root, 0}};
    index[root] = low[root] = next_index++;
    stack.push_back(root);
    on_stack[root] = true;
    while (!work.empty()) {
      auto& [node, edge] = work.back();
      if (edge < adjacency[node].size()) {
        const auto next = adjacency[node][edge++];
        if (index[next] == kUnvisited) {
          index[next] = low[next] = next_index++;
          stack.push_back(next);
          on_stack[next] = true;
          work.emplace_back(next, 0);
        } else if (on_stack[next]) {
          low[node] = std::min(low[node], index[next]);
        }
        continue;
      }
      if (low[node] == index[node]) {
        std::size_t member = 0;
        do {
          member = stack.back();
          stack.pop_back();
          on_stack[member] = false;
          component[member] = next_component;
        } while (member != node);
        ++next_component;
      }
      const auto finished = node;
      work.pop_back();
      if (!work.empty()) {
        auto& parent = work.back().first;
        low[parent] = std::min(low[parent], low[finished]);
      }
    }
  }
  return component;
}

class Fnv1a {
public:
  void add(std::string_view text)
  {
    for (const unsigned char c : text) {
      hash_ = (hash_ ^ c) * 1099511628211ull;
    }
    add(std::uint64_t{text.size()});
  }
  void add(std::uint64_t value)
  {
    for (int i = 0; i < 8; ++i) {
      hash_ = (hash_ ^ ((value >> (8 * i)) & 0xffu)) * 1099511628211ull;
    }
  }
  void add(std::int64_t value) { add(static_cast<std::uint64_t>(value)); }
  void add(int value) { add(static_cast<std::int64_t>(value)); }
  void add(bool value) { add(std::uint64_t{value ? 1u : 0u}); }
  void add(double value)
  {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    add(bits);
  }
  [[nodiscard]] std::uint64_t value() const noexcept { return hash_; }

private:
  std::uint64_t hash_{1469598103934665603ull};
};

void addResource(Fnv1a& hash, const GeneratedResource& resource)
{
  hash.add(static_cast<int>(resource.kind));
  hash.add(static_cast<std::int64_t>(resource.field));
  hash.add(resource.marker);
  hash.add(std::string_view(resource.operator_tag));
  hash.add(resource.steps_back);
}

std::string joined(const std::vector<std::string>& names)
{
  std::string out;
  for (const auto& name : names) {
    out += (out.empty() ? "" : ", ") + name;
  }
  return out;
}

} // namespace

GeneratedStatePlan compileGeneratedStateSchedule(
    const GeneratedStateScheduleInputs& inputs,
    const std::vector<DeclaredGeneratedState>& declarations,
    const std::vector<GeneratedRead>& residual_reads)
{
  GeneratedStatePlan plan;
  const auto block_field = [&](FieldId field) {
    return std::find(inputs.block_fields.begin(), inputs.block_fields.end(), field) !=
           inputs.block_fields.end();
  };
  const auto read_by_residual = [&](const GeneratedResource& resource) {
    return std::any_of(residual_reads.begin(), residual_reads.end(), [&](const GeneratedRead& read) {
      return read.version == StateVersion::FrozenPass && read.what == resource;
    });
  };

  // Nodes: the block, then one per declaration.  A declaration that
  // generates nothing makes no edge.
  std::vector<std::vector<std::size_t>> adjacency(declarations.size() + 1);
  std::vector<bool> fresh_prior_writer(declarations.size(), false);
  for (std::size_t i = 0; i < declarations.size(); ++i) {
    const auto& descriptor = declarations[i].descriptor;
    if (!descriptor.present) {
      continue;
    }
    // Prior pass: the block reads, frozen through the inner solve, what the
    // refresh writes.
    for (const auto& write : descriptor.writes) {
      if (!refreshPhase(write.phase) || !write.residual_defining) {
        continue;
      }
      const bool read_by_block =
          write.what.kind == ResourceKind::FrozenRows ||
          write.what.kind == ResourceKind::AffineConstraints ||
          (write.what.kind == ResourceKind::GeneratedDomain && read_by_residual(write.what)) ||
          (write.what.kind == ResourceKind::Field && !block_field(write.what.field) &&
           read_by_residual(write.what));
      if (read_by_block) {
        adjacency[i + 1].push_back(kBlockNode);
        fresh_prior_writer[i] = fresh_prior_writer[i] || write.requires_fresh_residual;
      }
    }
    // Same pass: the participant reads the block or an earlier write of the
    // refresh.
    for (const auto& read : descriptor.reads) {
      if (!sameRefreshVersion(read.version)) {
        continue;
      }
      if (read.what.kind == ResourceKind::Field && block_field(read.what.field)) {
        adjacency[kBlockNode].push_back(i + 1);
      }
      for (std::size_t j = 0; j < declarations.size(); ++j) {
        if (j == i || !declarations[j].descriptor.present) {
          continue;
        }
        for (const auto& write : declarations[j].descriptor.writes) {
          if (refreshPhase(write.phase) && write.what == read.what) {
            adjacency[j + 1].push_back(i + 1);
          }
        }
      }
    }
  }
  const auto component = stronglyConnectedComponents(adjacency);

  bool envelope_selects = false;
  bool inferred_selects = false;
  bool monolithic_extension = false;
  bool geometry_present = false;
  std::vector<std::string> inferred;
  std::vector<std::string> envelope;
  for (std::size_t i = 0; i < declarations.size(); ++i) {
    const auto& declaration = declarations[i];
    const auto& descriptor = declaration.descriptor;
    const bool in_block_component = component[i + 1] == component[kBlockNode];
    if (in_block_component) {
      plan.component.push_back(declaration.name);
    }
    if (!descriptor.present) {
      continue;
    }
    const bool required = descriptor.enablement == OuterEnablement::RequiredWhenPresent;
    const bool selecting = required || descriptor.enablement == OuterEnablement::Inferred;
    envelope_selects = envelope_selects || required;
    if (in_block_component && fresh_prior_writer[i] && selecting) {
      inferred.push_back(declaration.name);
      inferred_selects =
          inferred_selects || descriptor.enablement == OuterEnablement::Inferred;
    } else if (required) {
      envelope.push_back(declaration.name);
    }
    for (const auto& write : descriptor.writes) {
      monolithic_extension = monolithic_extension || write.what.kind == ResourceKind::FrozenRows;
    }
    geometry_present = geometry_present || (declaration.registry_participant &&
                                            declaration.role == GeneratedStateRole::Geometry);
  }

  // The selection, as legacyGeneratedStateSelection forms it from requests.
  auto& selection = plan.selection;
  selection.policy.solve = inputs.solve;
  selection.generated_state = envelope_selects || inferred_selects;
  bool outer = envelope_selects || inferred_selects;
  if (inputs.solve == GeneratedStateSolve::Steady) {
    selection.policy.outer_fixed_point = outer;
    selection.accepted_state_sync_invalidates_residual = selection.generated_state && !outer;
    selection.jacobian_check_refreshes_geometry_within_solve = !outer;
    selection.jacobian_check_frozen_algebraic_extension = false;
  } else {
    outer = outer && !inputs.per_step_only;
    const bool refresh_within_solve =
        selection.generated_state && !outer && !monolithic_extension;
    selection.policy.outer_fixed_point = outer;
    selection.policy.refresh_within_solve = refresh_within_solve;
    selection.accepted_state_sync_invalidates_residual =
        refresh_within_solve && selection.generated_state;
    selection.policy.synchronize_line_search_trials =
        synchronizeTransientLineSearchTrials(selection.accepted_state_sync_invalidates_residual);
    selection.policy.per_step_only = inputs.per_step_only;
    selection.track_cut_topology = geometry_present;
    selection.jacobian_check_refreshes_geometry_within_solve = refresh_within_solve;
    selection.jacobian_check_frozen_algebraic_extension = monolithic_extension;
  }
  if (outer) {
    if (!inferred.empty()) {
      plan.reasons.push_back("inferred(" + joined(inferred) + ")");
    }
    if (!envelope.empty()) {
      plan.reasons.push_back("envelope(" + joined(envelope) + ")");
    }
  }
  plan.fixed_point = outerFixedPointPreset(inputs.solve, inputs.overrides);
  plan.fixed_point.enabled = outer;

  for (const auto& declaration : declarations) {
    if (!declaration.registry_participant) {
      continue;
    }
    if (declaration.role == GeneratedStateRole::Geometry) {
      plan.constraint_construction_order.push_back(declaration.name);
    }
    plan.projected_refresh_order.push_back(declaration.name);
  }

  Fnv1a hash;
  hash.add(static_cast<int>(inputs.purpose));
  hash.add(static_cast<int>(inputs.solve));
  for (const auto& declaration : declarations) {
    hash.add(std::string_view(declaration.name));
    hash.add(static_cast<int>(declaration.role));
    hash.add(static_cast<std::uint64_t>(declaration.position));
    hash.add(declaration.registry_participant);
    hash.add(declaration.descriptor.present);
    hash.add(static_cast<int>(declaration.descriptor.enablement));
    for (const auto& read : declaration.descriptor.reads) {
      addResource(hash, read.what);
      hash.add(static_cast<int>(read.version));
    }
    for (const auto& write : declaration.descriptor.writes) {
      addResource(hash, write.what);
      hash.add(static_cast<int>(write.phase));
      hash.add(write.residual_defining);
      hash.add(write.requires_fresh_residual);
      hash.add(write.writes_state_vector);
    }
  }
  hash.add(selection.policy.outer_fixed_point);
  hash.add(selection.policy.refresh_within_solve);
  hash.add(selection.policy.synchronize_line_search_trials);
  hash.add(selection.policy.per_step_only);
  hash.add(selection.accepted_state_sync_invalidates_residual);
  hash.add(selection.track_cut_topology);
  hash.add(plan.fixed_point.max_iterations);
  hash.add(plan.fixed_point.max_discontinuity_restarts);
  hash.add(plan.fixed_point.constraint_cycle_exit_start_iteration);
  hash.add(plan.fixed_point.dynamic_relaxation.enabled);
  hash.add(plan.fixed_point.dynamic_relaxation.start_iteration);
  plan.fingerprint = hash.value();
  return plan;
}

std::vector<std::string> generatedStateScheduleMismatches(
    const GeneratedStateSelection& legacy,
    const NewtonOptions::ExternalStateFixedPointOptions& legacy_fixed_point,
    const GeneratedStatePlan& plan)
{
  std::vector<std::string> mismatches;
  const auto compare = [&](const char* name, auto a, auto b) {
    if (!(a == b)) {
      mismatches.emplace_back(name);
    }
  };
  const auto& s = plan.selection;
  compare("solve", legacy.policy.solve, s.policy.solve);
  compare("outer_fixed_point", legacy.policy.outer_fixed_point, s.policy.outer_fixed_point);
  compare("refresh_within_solve", legacy.policy.refresh_within_solve, s.policy.refresh_within_solve);
  compare("synchronize_line_search_trials", legacy.policy.synchronize_line_search_trials,
          s.policy.synchronize_line_search_trials);
  compare("per_step_only", legacy.policy.per_step_only, s.policy.per_step_only);
  compare("generated_state", legacy.generated_state, s.generated_state);
  compare("accepted_state_sync_invalidates_residual", legacy.accepted_state_sync_invalidates_residual,
          s.accepted_state_sync_invalidates_residual);
  compare("track_cut_topology", legacy.track_cut_topology, s.track_cut_topology);
  compare("jacobian_check_refreshes_geometry_within_solve",
          legacy.jacobian_check_refreshes_geometry_within_solve,
          s.jacobian_check_refreshes_geometry_within_solve);
  compare("jacobian_check_frozen_algebraic_extension", legacy.jacobian_check_frozen_algebraic_extension,
          s.jacobian_check_frozen_algebraic_extension);
  const auto& a = legacy_fixed_point;
  const auto& b = plan.fixed_point;
  compare("fixed_point.enabled", a.enabled, b.enabled);
  compare("fixed_point.max_iterations", a.max_iterations, b.max_iterations);
  compare("fixed_point.max_discontinuity_restarts", a.max_discontinuity_restarts,
          b.max_discontinuity_restarts);
  compare("fixed_point.constraint_cycle_exit_start_iteration", a.constraint_cycle_exit_start_iteration,
          b.constraint_cycle_exit_start_iteration);
  compare("fixed_point.dynamic_relaxation.enabled", a.dynamic_relaxation.enabled,
          b.dynamic_relaxation.enabled);
  compare("fixed_point.dynamic_relaxation.start_iteration", a.dynamic_relaxation.start_iteration,
          b.dynamic_relaxation.start_iteration);
  compare("fixed_point.dynamic_relaxation.initial_factor", a.dynamic_relaxation.initial_factor,
          b.dynamic_relaxation.initial_factor);
  compare("fixed_point.dynamic_relaxation.minimum_factor", a.dynamic_relaxation.minimum_factor,
          b.dynamic_relaxation.minimum_factor);
  compare("fixed_point.dynamic_relaxation.maximum_factor", a.dynamic_relaxation.maximum_factor,
          b.dynamic_relaxation.maximum_factor);
  compare("fixed_point.dynamic_relaxation.denominator_relative_tolerance",
          a.dynamic_relaxation.denominator_relative_tolerance,
          b.dynamic_relaxation.denominator_relative_tolerance);
  compare("fixed_point.residual_first_on_predicted_certificate",
          a.residual_first_on_predicted_certificate, b.residual_first_on_predicted_certificate);
  return mismatches;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/GeneratedState.h"

#include "Backends/Interfaces/GenericVector.h"
#include "Dofs/DofHandler.h"
#include "Systems/FESystem.h"
#include "Systems/FieldBinding.h"
#include "Systems/SystemCollectives.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
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

} // namespace timestepping
} // namespace FE
} // namespace svmp

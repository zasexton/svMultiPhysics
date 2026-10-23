/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/MeshNormalConstraintStageMeasurements.h"

#include "Assembly/TimeIntegrationContext.h"
#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Forms/BoundaryFunctional.h"
#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"
#include "Systems/BoundaryReductionService.h"
#include "Systems/MeshBoundaryRecordSupport.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp::FE::systems {

// The record code was in FESystem.cpp and then in the Navier-Stokes
// free-surface physics (FreeSurface/FittedALEStageRecords.cpp); it keeps the
// names it used there.
using namespace record_support;

namespace {

[[nodiscard]] bool meshNormalDeclarationsStructurallyEqual(
    const MeshNormalBoundaryConstraintDeclaration& lhs,
    const MeshNormalBoundaryConstraintDeclaration& rhs)
{
    MeshBoundaryConsensusTokens lhs_tokens;
    MeshBoundaryConsensusTokens rhs_tokens;
    bool lhs_contains_opaque_callable = false;
    bool rhs_contains_opaque_callable = false;
    appendMeshNormalDeclaration(
        lhs_tokens, lhs, lhs_contains_opaque_callable);
    appendMeshNormalDeclaration(
        rhs_tokens, rhs, rhs_contains_opaque_callable);
    return lhs_contains_opaque_callable == rhs_contains_opaque_callable &&
           lhs_tokens == rhs_tokens;
}

void appendFittedALENormalMeasurementDeclaration(
    MeshBoundaryConsensusTokens& tokens,
    const FittedALENormalMeasurementDeclaration& declaration,
    bool& contains_opaque_callable)
{
    appendMeshBoundaryIntegral(
        tokens, declaration.key.mesh_displacement_field);
    appendMeshBoundaryIntegral(tokens, declaration.key.boundary_marker);
    appendMeshBoundaryIntegral(
        tokens, declaration.related_velocity_field);
    appendMeshNormalDeclaration(
        tokens,
        declaration.normal_constraint,
        contains_opaque_callable);
    appendMeshBoundaryIntegral(tokens, declaration.tangential_policy);
    appendMeshBoundaryToken(
        tokens,
        declaration.prescribed_tangential_mesh_velocity.has_value()
            ? 1u
            : 0u);
    if (declaration.prescribed_tangential_mesh_velocity.has_value()) {
        for (const auto value :
             *declaration.prescribed_tangential_mesh_velocity) {
            appendMeshBoundaryToken(
                tokens, generatedBoundaryTraceRealBits(value));
        }
    }
    appendMeshBoundaryString(
        tokens, declaration.mesh_normal_integral_functional);
    appendMeshBoundaryString(
        tokens, declaration.fluid_normal_integral_functional);
    appendMeshBoundaryString(
        tokens, declaration.normal_gap_squared_integral_functional);
    appendMeshBoundaryString(
        tokens, declaration.mesh_velocity_squared_integral_functional);
    appendMeshBoundaryString(
        tokens, declaration.mesh_normal_squared_integral_functional);
    appendMeshBoundaryString(
        tokens, declaration.mesh_tangential_squared_integral_functional);
    appendMeshBoundaryString(
        tokens,
        declaration.tangential_target_gap_squared_integral_functional);
}

void appendFittedALENormalOperatorStageRawValue(
    MeshBoundaryConsensusTokens& tokens,
    const FittedALENormalOperatorStageRawValue& raw)
{
    appendMeshBoundaryIntegral(
        tokens, raw.key.mesh_displacement_field);
    appendMeshBoundaryIntegral(tokens, raw.key.boundary_marker);
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(raw.A));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(raw.Wn));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(raw.Un));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(raw.gap_sq));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(raw.mesh_velocity_sq));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(raw.mesh_normal_sq));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(raw.mesh_tangential_sq));
    appendMeshBoundaryToken(
        tokens, raw.tangential_target_gap_sq.has_value() ? 1u : 0u);
    if (raw.tangential_target_gap_sq.has_value()) {
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                *raw.tangential_target_gap_sq));
    }
    // Both geometry stamps are rank-local provenance and must never
    // participate in a cross-rank equality claim.
}

void appendFittedALENormalOperatorStageRecord(
    MeshBoundaryConsensusTokens& tokens,
    const FittedALENormalOperatorStageHistoryRecord& record,
    bool& contains_opaque_callable)
{
    appendFittedALENormalMeasurementDeclaration(
        tokens, record.declaration, contains_opaque_callable);
    appendOperatorStageMeasurementMetadata(tokens, record.stage);
    appendFittedALENormalOperatorStageRawValue(tokens, record.raw);
}

} // namespace

MeshNormalConstraintStageMeasurements::MeshNormalConstraintStageMeasurements(FESystem& system)
    : system_(&system),
      fitted_ale_normal_measurement_declarations_(
          system.attachment<MeshNormalConstraintStageMeasurementStore>().declarations),
      pending_fitted_ale_normal_measurements_(
          system.attachment<MeshNormalConstraintStageMeasurementStore>().pending),
      fitted_ale_normal_measurement_history_(
          system.attachment<MeshNormalConstraintStageMeasurementStore>().history),
      fitted_ale_normal_measurement_declarations_frozen_(
          system.attachment<MeshNormalConstraintStageMeasurementStore>().declarations_frozen),
      fitted_ale_normal_measurement_transaction_active_(
          system.attachment<MeshNormalConstraintStageMeasurementStore>().transaction_active),
      field_registry_{&system},
      mesh_normal_boundary_constraints_{&system},
      mesh_normal_boundary_constraint_history_{&system},
      mesh_tangential_boundary_policies_{&system},
      mesh_tangential_boundary_policy_history_{&system}
{
}

ConstMeshNormalConstraintStageMeasurements::ConstMeshNormalConstraintStageMeasurements(
    const FESystem& system) noexcept
    : store_(system.findAttachment<MeshNormalConstraintStageMeasurementStore>())
{
}

std::span<const FittedALENormalMeasurementDeclaration>
ConstMeshNormalConstraintStageMeasurements::fittedALENormalOperatorStageMeasurementDeclarations()
    const noexcept
{
  if (store_ == nullptr) {
    return {};
  }
  return store_->declarations;
}

std::span<const FittedALENormalOperatorStageHistoryRecord>
ConstMeshNormalConstraintStageMeasurements::pendingFittedALENormalOperatorStageMeasurements()
    const noexcept
{
  if (store_ == nullptr) {
    return {};
  }
  return store_->pending;
}

std::span<const FittedALENormalOperatorStageHistoryRecord>
ConstMeshNormalConstraintStageMeasurements::fittedALENormalOperatorStageMeasurementHistory()
    const noexcept
{
  if (store_ == nullptr) {
    return {};
  }
  return store_->history;
}

MeshNormalConstraintStageMeasurements meshNormalConstraintStageMeasurements(FESystem& system)
{
  return MeshNormalConstraintStageMeasurements(system);
}

ConstMeshNormalConstraintStageMeasurements meshNormalConstraintStageMeasurements(const FESystem& system) noexcept
{
  return ConstMeshNormalConstraintStageMeasurements(system);
}

void MeshNormalConstraintStageMeasurements::registerFittedALENormalOperatorStageMeasurement(
    FieldId mesh_displacement_field,
    int boundary_marker,
    std::optional<std::array<Real, 3>>
        prescribed_tangential_mesh_velocity)
{
    FE_THROW_IF(
        meshBoundaryHistoryTransactionActive() ||
            !mesh_normal_boundary_constraint_history_.empty() ||
            !mesh_tangential_boundary_policy_history_.empty() ||
            fitted_ale_normal_measurement_declarations_frozen_ ||
            fitted_ale_normal_measurement_transaction_active_ ||
            !pending_fitted_ale_normal_measurements_.empty() ||
            !fitted_ale_normal_measurement_history_.empty(),
        InvalidArgumentException,
        "FESystem::registerFittedALENormalOperatorStageMeasurement: "
        "declarations are frozen after staging or any accepted mesh-"
        "boundary history begins");

    // Complete all declaration/consumer validation before mutating either
    // the reduction service or the measurement declaration table.
    const auto& constraint = validatedFittedALENormalConstraint(
        mesh_displacement_field, boundary_marker);
    const auto tangential_declaration = std::find_if(
        mesh_tangential_boundary_policies_.begin(),
        mesh_tangential_boundary_policies_.end(),
        [&](const auto& candidate) {
            return candidate.mesh_displacement_field ==
                       mesh_displacement_field &&
                   candidate.boundary_marker == boundary_marker;
        });
    FE_THROW_IF(
        tangential_declaration ==
                mesh_tangential_boundary_policies_.end() ||
            !tangential_declaration->consumer_bound ||
            tangential_declaration->consumer_operator_tag !=
                constraint.consumer_binding->operator_tag ||
            tangential_declaration->consumer_source.find_first_not_of(
                " \t\r\n") == std::string::npos,
        InvalidArgumentException,
        "FESystem::registerFittedALENormalOperatorStageMeasurement: "
        "the matching tangential policy requires an exact consumer binding "
        "on the fitted normal operator");
    const bool has_prescribed_target =
        prescribed_tangential_mesh_velocity.has_value();
    const bool requires_prescribed_target =
        tangential_declaration->policy ==
        MeshTangentialBoundaryPolicy::Prescribed;
    FE_THROW_IF(
        has_prescribed_target != requires_prescribed_target,
        InvalidArgumentException,
        "FESystem::registerFittedALENormalOperatorStageMeasurement: "
        "a prescribed tangential policy requires exactly one velocity "
        "target and untargeted policies must not invent one");
    const FittedALENormalMeasurementKey key{
        .mesh_displacement_field = mesh_displacement_field,
        .boundary_marker = boundary_marker,
    };
    const std::string name_base =
        "__svmp_fitted_ale_normal_operator_stage_d" +
        std::to_string(mesh_displacement_field) + "_b" +
        std::to_string(boundary_marker);
    FittedALENormalMeasurementDeclaration measurement{
        .key = key,
        .related_velocity_field = constraint.related_velocity_field,
        .normal_constraint = constraint,
        .tangential_policy = tangential_declaration->policy,
        .prescribed_tangential_mesh_velocity =
            prescribed_tangential_mesh_velocity,
        .mesh_normal_integral_functional =
            name_base + "_integral_mesh_normal",
        .fluid_normal_integral_functional =
            name_base + "_integral_fluid_normal",
        .normal_gap_squared_integral_functional =
            name_base + "_integral_normal_gap_squared",
        .mesh_velocity_squared_integral_functional =
            name_base + "_integral_mesh_velocity_squared",
        .mesh_normal_squared_integral_functional =
            name_base + "_integral_mesh_normal_squared",
        .mesh_tangential_squared_integral_functional =
            name_base + "_integral_mesh_tangential_squared",
        .tangential_target_gap_squared_integral_functional =
            requires_prescribed_target
                ? name_base +
                      "_integral_tangential_target_gap_squared"
                : std::string{},
    };

    const auto existing = std::find_if(
        fitted_ale_normal_measurement_declarations_.begin(),
        fitted_ale_normal_measurement_declarations_.end(),
        [&](const auto& candidate) { return candidate.key == key; });
    if (existing !=
        fitted_ale_normal_measurement_declarations_.end()) {
        FE_THROW_IF(
            existing->related_velocity_field !=
                    measurement.related_velocity_field ||
                !meshNormalDeclarationsStructurallyEqual(
                    existing->normal_constraint,
                    measurement.normal_constraint) ||
                existing->tangential_policy !=
                    measurement.tangential_policy ||
                existing->prescribed_tangential_mesh_velocity !=
                    measurement.prescribed_tangential_mesh_velocity ||
                existing->mesh_normal_integral_functional !=
                    measurement.mesh_normal_integral_functional ||
                existing->fluid_normal_integral_functional !=
                    measurement.fluid_normal_integral_functional ||
                existing->normal_gap_squared_integral_functional !=
                    measurement.normal_gap_squared_integral_functional ||
                existing->mesh_velocity_squared_integral_functional !=
                    measurement.mesh_velocity_squared_integral_functional ||
                existing->mesh_normal_squared_integral_functional !=
                    measurement.mesh_normal_squared_integral_functional ||
                existing->mesh_tangential_squared_integral_functional !=
                    measurement.mesh_tangential_squared_integral_functional ||
                existing
                        ->tangential_target_gap_squared_integral_functional !=
                    measurement
                        .tangential_target_gap_squared_integral_functional,
            InvalidArgumentException,
            "FESystem::registerFittedALENormalOperatorStageMeasurement: "
            "measurement key already has a conflicting declaration");
        return;
    }

    const auto& displacement = field_registry_.get(
        constraint.mesh_displacement_field);
    if (prescribed_tangential_mesh_velocity.has_value()) {
        for (int component = 0; component < 3; ++component) {
            const auto value =
                (*prescribed_tangential_mesh_velocity)[
                    static_cast<std::size_t>(component)];
            FE_THROW_IF(
                !std::isfinite(value) ||
                    (component >= displacement.components &&
                     value != Real{0.0}),
                InvalidArgumentException,
                "FESystem::registerFittedALENormalOperatorStageMeasurement: "
                "the prescribed tangential target must be finite and lie "
                "in the mesh dimension");
        }
    }
    const auto expected_tangential_enforcement =
        tangential_declaration->policy ==
                MeshTangentialBoundaryPolicy::Prescribed
            ? analysis::EnforcementKind::WeakPenalty
            : analysis::EnforcementKind::WeakConsistent;
    const auto tangential_descriptor_count =
        countOperatorBoundBoundaryDescriptors(
            boundaryConditionDescriptors(),
            boundaryConditionDescriptorOperatorTags(),
            tangential_declaration->consumer_operator_tag,
            [&](const auto& descriptor) {
                return descriptor.primary_variable ==
                           analysis::VariableKey::field(
                               mesh_displacement_field) &&
                       descriptor.component == -1 &&
                       descriptor.domain ==
                           analysis::DomainKind::Boundary &&
                       descriptor.boundary_marker == boundary_marker &&
                       descriptor.interface_marker == -1 &&
                       descriptor.trace_kind ==
                           analysis::TraceKind::TangentialComponent &&
                       descriptor.enforcement_kind ==
                           expected_tangential_enforcement &&
                       descriptor.related_variables.empty() &&
                       descriptor.source ==
                           tangential_declaration->consumer_source;
            });
    FE_THROW_IF(
        tangential_descriptor_count != 1u,
        InvalidArgumentException,
        "FESystem::registerFittedALENormalOperatorStageMeasurement: "
        "the fitted tangential policy requires exactly one bound boundary "
        "descriptor");
    const auto displacement_state = forms::StateField(
        constraint.mesh_displacement_field,
        *displacement.space,
        "fitted_ale_mesh_displacement");
    const auto current_normal = forms::currentNormal();
    const auto mesh_normal = forms::normalTrace(
        displacement_state.dt(), current_normal);
    const auto mesh_velocity = displacement_state.dt();
    const auto tangential_projector =
        forms::FormExpr::identity(displacement.components) -
        forms::outer(current_normal, current_normal);
    const auto mesh_tangential_velocity =
        tangential_projector * mesh_velocity;
    std::optional<forms::FormExpr> tangential_target_gap;
    if (prescribed_tangential_mesh_velocity.has_value()) {
        std::vector<forms::FormExpr> target_components;
        target_components.reserve(
            static_cast<std::size_t>(displacement.components));
        for (int component = 0;
             component < displacement.components;
             ++component) {
            target_components.push_back(forms::FormExpr::constant(
                (*prescribed_tangential_mesh_velocity)[
                    static_cast<std::size_t>(component)]));
        }
        const auto target =
            forms::FormExpr::asVector(std::move(target_components));
        tangential_target_gap.emplace(
            tangential_projector * (mesh_velocity - target));
    }
    // Preserve the exact consumer-bound expression rather than rebuilding an
    // equivalent fluid trace from field metadata.
    const auto fluid_normal = constraint.target_expression;
    const auto normal_gap = mesh_normal - fluid_normal;

    forms::BoundaryFunctional mesh_functional{
        .integrand = mesh_normal,
        .boundary_marker = boundary_marker,
        .name = measurement.mesh_normal_integral_functional,
        .reduction = forms::BoundaryFunctional::Reduction::Sum,
    };
    forms::BoundaryFunctional fluid_functional{
        .integrand = fluid_normal,
        .boundary_marker = boundary_marker,
        .name = measurement.fluid_normal_integral_functional,
        .reduction = forms::BoundaryFunctional::Reduction::Sum,
    };
    forms::BoundaryFunctional gap_functional{
        .integrand = normal_gap * normal_gap,
        .boundary_marker = boundary_marker,
        .name = measurement.normal_gap_squared_integral_functional,
        .reduction = forms::BoundaryFunctional::Reduction::Sum,
    };
    forms::BoundaryFunctional mesh_velocity_squared_functional{
        .integrand = forms::inner(mesh_velocity, mesh_velocity),
        .boundary_marker = boundary_marker,
        .name = measurement.mesh_velocity_squared_integral_functional,
        .reduction = forms::BoundaryFunctional::Reduction::Sum,
    };
    forms::BoundaryFunctional mesh_normal_squared_functional{
        .integrand = mesh_normal * mesh_normal,
        .boundary_marker = boundary_marker,
        .name = measurement.mesh_normal_squared_integral_functional,
        .reduction = forms::BoundaryFunctional::Reduction::Sum,
    };
    forms::BoundaryFunctional mesh_tangential_squared_functional{
        .integrand = forms::inner(
            mesh_tangential_velocity, mesh_tangential_velocity),
        .boundary_marker = boundary_marker,
        .name = measurement.mesh_tangential_squared_integral_functional,
        .reduction = forms::BoundaryFunctional::Reduction::Sum,
    };
    std::optional<forms::BoundaryFunctional>
        tangential_target_gap_squared_functional;
    if (tangential_target_gap.has_value()) {
        tangential_target_gap_squared_functional.emplace(
            forms::BoundaryFunctional{
                .integrand = forms::inner(
                    *tangential_target_gap,
                    *tangential_target_gap),
                .boundary_marker = boundary_marker,
                .name = measurement
                    .tangential_target_gap_squared_integral_functional,
                .reduction =
                    forms::BoundaryFunctional::Reduction::Sum,
            });
    }

    // The displacement must remain primary: dt(StateField(d)) consumes its
    // exact history alias.  The velocity is a current-only secondary field.
    auto& service = boundaryReductionService(mesh_displacement_field);
    service.addBoundaryFunctional(std::move(mesh_functional));
    service.addBoundaryFunctional(std::move(fluid_functional));
    service.addBoundaryFunctional(std::move(gap_functional));
    service.addBoundaryFunctional(
        std::move(mesh_velocity_squared_functional));
    service.addBoundaryFunctional(
        std::move(mesh_normal_squared_functional));
    service.addBoundaryFunctional(
        std::move(mesh_tangential_squared_functional));
    if (tangential_target_gap_squared_functional.has_value()) {
        service.addBoundaryFunctional(
            std::move(*tangential_target_gap_squared_functional));
    }
    bindSecondaryFields(
        service,
        mesh_displacement_field,
        std::vector<FieldId>{
            mesh_displacement_field,
            constraint.related_velocity_field});

    // Publish the declaration only after every reduction and binding has
    // been installed.  A retry after an allocation failure is deterministic
    // because BoundaryReductionService registration is idempotent.
    fitted_ale_normal_measurement_declarations_.push_back(
        std::move(measurement));
    std::sort(
        fitted_ale_normal_measurement_declarations_.begin(),
        fitted_ale_normal_measurement_declarations_.end(),
        [](const auto& lhs, const auto& rhs) {
            return std::tie(lhs.key.mesh_displacement_field,
                            lhs.key.boundary_marker) <
                   std::tie(rhs.key.mesh_displacement_field,
                            rhs.key.boundary_marker);
        });
}

void MeshNormalConstraintStageMeasurements::stageFittedALENormalOperatorStageMeasurements(
    const SystemStateView& state,
    OperatorStageMeasurementMetadata metadata)
{
    const auto collective = meshBoundaryCollectiveContext(*system_);
    MeshBoundaryConsensusTokens local_tokens;
    MeshBoundaryConsensusTokens minimum_tokens;
    MeshBoundaryConsensusTokens maximum_tokens;
    std::exception_ptr local_exception;
    bool contains_opaque_callable = false;

    try {
        std::sort(
            metadata.derivative_fields.begin(),
            metadata.derivative_fields.end());
        metadata.derivative_fields.erase(
            std::unique(metadata.derivative_fields.begin(),
                        metadata.derivative_fields.end()),
            metadata.derivative_fields.end());
        FE_THROW_IF(
            !isSetup() || !dofHandler().isFinalized(),
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "system setup must be complete");
        FE_THROW_IF(
            fitted_ale_normal_measurement_transaction_active_ ||
                !pending_fitted_ale_normal_measurements_.empty(),
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: a "
            "candidate measurement group is already active or pending");
        const bool is_endpoint_scheme =
            metadata.scheme_name == "BackwardEuler" ||
            metadata.scheme_name == "DG0";
        const bool is_generalized_alpha =
            metadata.scheme_name == "GeneralizedAlphaFirstOrder";
        FE_THROW_IF(
            (!is_endpoint_scheme && !is_generalized_alpha) ||
                metadata.temporal_order != 1 ||
                metadata.prospective_accepted_step == 0u ||
                !std::isfinite(metadata.step_start_time) ||
                !std::isfinite(metadata.step_end_time) ||
                !std::isfinite(metadata.state_time) ||
                !std::isfinite(metadata.rate_time) ||
                !std::isfinite(metadata.dt) || metadata.dt <= Real{0.0} ||
                metadata.step_end_time < metadata.step_start_time ||
                metadata.state_revision == 0u ||
                metadata.rate_revision == 0u,
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "v1 requires BackwardEuler, DG0, or "
            "GeneralizedAlphaFirstOrder at temporal order one, a nonzero "
            "prospective step, finite times, positive dt, and nonzero "
            "state/rate revisions");
        FE_THROW_IF(
            metadata.generalized_alpha.has_value() &&
                (!std::isfinite(
                     metadata.generalized_alpha->alpha_f) ||
                 !std::isfinite(
                     metadata.generalized_alpha->alpha_m)),
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "generalized-alpha coordinates must be finite");
        FE_THROW_IF(
            !operatorStageTimesMatch(
                metadata.step_end_time,
                metadata.step_start_time + metadata.dt),
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "step end time is inconsistent with start time plus dt");
        if (is_endpoint_scheme) {
            FE_THROW_IF(
                metadata.generalized_alpha.has_value() ||
                    !operatorStageTimesMatch(
                        metadata.state_time,
                        metadata.step_end_time) ||
                    !operatorStageTimesMatch(
                        metadata.rate_time,
                        metadata.step_end_time),
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "BackwardEuler and DG0 require endpoint state/rate times and "
                "no generalized-alpha coordinates");
        } else {
            FE_THROW_IF(
                !metadata.generalized_alpha.has_value(),
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "GeneralizedAlphaFirstOrder requires alpha_f/alpha_m "
                "coordinates");
            const auto& alpha = *metadata.generalized_alpha;
            FE_THROW_IF(
                !operatorStageTimesMatch(
                    metadata.state_time,
                    metadata.step_start_time + alpha.alpha_f * metadata.dt) ||
                    !operatorStageTimesMatch(
                        metadata.rate_time,
                        metadata.step_start_time +
                            alpha.alpha_m * metadata.dt),
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "generalized-alpha state/rate times do not match their "
                "alpha-derived operator coordinates");
        }
        for (const auto field : metadata.derivative_fields) {
            FE_THROW_IF(
                !field_registry_.has(field),
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "derivative field metadata contains an unknown field");
        }

        const bool has_current_state =
            state.u_vector != nullptr || !state.u.empty();
        const bool has_exact_rate =
            state.u_prev_vector != nullptr || !state.u_prev.empty();
        FE_THROW_IF(
            !fitted_ale_normal_measurement_declarations_.empty() &&
                (!has_current_state || !has_exact_rate),
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "nonempty measurements require current stage state and an "
            "exact-rate alias in u_prev");

        const auto eligible_constraint_count =
            static_cast<std::size_t>(std::count_if(
                mesh_normal_boundary_constraints_.begin(),
                mesh_normal_boundary_constraints_.end(),
                [](const auto& declaration) {
                    return declaration.target_kind ==
                               MeshNormalBoundaryTargetKind::
                                   FluidNormalVelocity &&
                           declaration.consumer_binding.has_value();
                }));
        FE_THROW_IF(
            eligible_constraint_count !=
                fitted_ale_normal_measurement_declarations_.size(),
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "every consumer-bound fluid-normal mesh constraint requires "
            "exactly one fitted-ALE measurement declaration");

        for (std::size_t index = 0u;
             index < fitted_ale_normal_measurement_declarations_.size();
             ++index) {
            const auto& declaration =
                fitted_ale_normal_measurement_declarations_[index];
            if (index != 0u) {
                const auto& previous =
                    fitted_ale_normal_measurement_declarations_[index - 1u];
                FE_THROW_IF(
                    std::tie(declaration.key.mesh_displacement_field,
                             declaration.key.boundary_marker) <=
                        std::tie(previous.key.mesh_displacement_field,
                                 previous.key.boundary_marker),
                    InvalidArgumentException,
                    "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                    "measurement declarations are not in canonical key order");
            }
            const auto& constraint =
                validatedFittedALENormalConstraint(
                    declaration.key.mesh_displacement_field,
                    declaration.key.boundary_marker);
            const auto tangential = std::find_if(
                mesh_tangential_boundary_policies_.begin(),
                mesh_tangential_boundary_policies_.end(),
                [&](const auto& candidate) {
                    return candidate.mesh_displacement_field ==
                               declaration.key.mesh_displacement_field &&
                           candidate.boundary_marker ==
                               declaration.key.boundary_marker;
                });
            const bool has_prescribed_target =
                declaration.prescribed_tangential_mesh_velocity.has_value();
            FE_THROW_IF(
                declaration.related_velocity_field !=
                        constraint.related_velocity_field ||
                    !meshNormalDeclarationsStructurallyEqual(
                        declaration.normal_constraint,
                        constraint) ||
                    tangential ==
                        mesh_tangential_boundary_policies_.end() ||
                    !tangential->consumer_bound ||
                    tangential->consumer_operator_tag !=
                        constraint.consumer_binding->operator_tag ||
                    tangential->policy !=
                        declaration.tangential_policy ||
                    has_prescribed_target !=
                        (declaration.tangential_policy ==
                         MeshTangentialBoundaryPolicy::Prescribed) ||
                    has_prescribed_target !=
                        !declaration
                             .tangential_target_gap_squared_integral_functional
                             .empty() ||
                    !std::binary_search(
                        metadata.derivative_fields.begin(),
                        metadata.derivative_fields.end(),
                        declaration.key.mesh_displacement_field),
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "measurement no longer matches its exact constraint or the "
                "mesh displacement is not declared as a derivative field");
            const auto* service = boundaryReductionServiceIfPresent(
                declaration.key.mesh_displacement_field);
            FE_THROW_IF(
                service == nullptr ||
                    !service->hasFunctional(
                        declaration.mesh_normal_integral_functional) ||
                    !service->hasFunctional(
                        declaration.fluid_normal_integral_functional) ||
                    !service->hasFunctional(
                        declaration.normal_gap_squared_integral_functional) ||
                    !service->hasFunctional(
                        declaration.mesh_velocity_squared_integral_functional) ||
                    !service->hasFunctional(
                        declaration.mesh_normal_squared_integral_functional) ||
                    !service->hasFunctional(
                        declaration.mesh_tangential_squared_integral_functional) ||
                    (has_prescribed_target &&
                     !service->hasFunctional(
                         declaration
                             .tangential_target_gap_squared_integral_functional)),
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "registered reduction service is incomplete");
        }

        if (!fitted_ale_normal_measurement_declarations_.empty()) {
            FE_THROW_IF(
                !metadata.expected_stage_geometry.has_value() ||
                    !meshAccess().revisionTrackingAvailable(),
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "v1 requires an available expected rank-local stage "
                "geometry stamp");
            const auto& expected = *metadata.expected_stage_geometry;
            FE_THROW_IF(
                captureOperatorStageGeometry(meshAccess()) != expected,
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "live geometry does not match the captured operator-stage "
                "observation");
        }

        appendMeshBoundaryToken(
            local_tokens,
            static_cast<std::uint64_t>(
                fitted_ale_normal_measurement_declarations_.size()));
        appendMeshBoundaryToken(
            local_tokens,
            fitted_ale_normal_measurement_declarations_frozen_ ? 1u : 0u);
        for (const auto& declaration :
             fitted_ale_normal_measurement_declarations_) {
            appendFittedALENormalMeasurementDeclaration(
                local_tokens,
                declaration,
                contains_opaque_callable);
        }
        appendOperatorStageMeasurementMetadata(local_tokens, metadata);
        appendMeshBoundaryToken(
            local_tokens,
            static_cast<std::uint64_t>(
                fitted_ale_normal_measurement_history_.size()));
        FE_THROW_IF(
            collective.size > 1 && contains_opaque_callable,
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "distributed declarations may not contain opaque callbacks");
        FE_THROW_IF(
            local_tokens.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()),
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "consensus token sequence exceeds the MPI count range");
        minimum_tokens = local_tokens;
        maximum_tokens = local_tokens;
    } catch (...) {
        local_exception = std::current_exception();
    }

    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::stageFittedALENormalOperatorStageMeasurements/preflight")) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "another active communicator rank rejected stage provenance");
    }
    requireMeshBoundaryTokenConsensus(
        collective,
        local_tokens,
        minimum_tokens,
        maximum_tokens,
        "FESystem::stageFittedALENormalOperatorStageMeasurements/consensus");

    if (fitted_ale_normal_measurement_declarations_.empty()) {
        return;
    }

    // Once a collective stage transaction begins, its declaration schedule
    // remains fixed even if the candidate is later discarded.
    fitted_ale_normal_measurement_declarations_frozen_ = true;
    system_->freezeMeshNormalBoundaryConstraintDeclarations();

    OperatorStageGeometryMetadata stage_mesh_revision{};
    local_exception = nullptr;
    try {
        const auto& expected = *metadata.expected_stage_geometry;
        stage_mesh_revision =
            captureOperatorStageGeometry(meshAccess());
        FE_THROW_IF(
            stage_mesh_revision != expected,
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "geometry changed during stage consensus");
    } catch (...) {
        local_exception = std::current_exception();
    }
    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::stageFittedALENormalOperatorStageMeasurements/geometry")) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "another active communicator rank observed stale stage geometry");
    }

    fitted_ale_normal_measurement_transaction_active_ = true;
    struct MeasurementRoute {
        BoundaryReductionService* service{nullptr};
        const forms::BoundaryFunctional* measure_functional{nullptr};
        std::string_view mesh_normal_name{};
        std::string_view fluid_normal_name{};
        std::string_view gap_squared_name{};
        std::string_view mesh_velocity_squared_name{};
        std::string_view mesh_normal_squared_name{};
        std::string_view mesh_tangential_squared_name{};
        std::string_view tangential_target_gap_squared_name{};
    };
    std::optional<assembly::TimeIntegrationContext> exact_rate_alias;
    std::optional<SystemStateView> measurement_state;
    std::vector<MeasurementRoute> measurement_routes;
    std::vector<FittedALENormalOperatorStageHistoryRecord> staged_records;
    local_exception = nullptr;
    try {
        exact_rate_alias.emplace();
        exact_rate_alias->integrator_name =
            "FittedALEExactOperatorStageRateAlias";
        exact_rate_alias->dt1 = assembly::TimeDerivativeStencil{
            .order = 1,
            .a = {Real{0.0}, Real{1.0}},
        };
        measurement_state.emplace(state);
        measurement_state->time = metadata.state_time;
        measurement_state->dt = metadata.dt;
        measurement_state->effective_dt = metadata.dt;
        measurement_state->u_prev2 = {};
        measurement_state->u_prev2_vector = nullptr;
        measurement_state->u_history = {};
        measurement_state->dt_history = {};
        measurement_state->time_integration =
            &*exact_rate_alias;

        measurement_routes.reserve(
            fitted_ale_normal_measurement_declarations_.size());
        staged_records.reserve(
            fitted_ale_normal_measurement_declarations_.size());
        for (const auto& declaration :
             fitted_ale_normal_measurement_declarations_) {
            auto* service = boundaryReductionServiceIfPresent(
                declaration.key.mesh_displacement_field);
            FE_THROW_IF(
                service == nullptr,
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "registered reduction service disappeared during stage "
                "preparation");
            const auto& measure_functional = service->functionalDef(
                declaration.mesh_normal_integral_functional);
            measurement_routes.push_back(
                MeasurementRoute{
                    .service = service,
                    .measure_functional = &measure_functional,
                    .mesh_normal_name =
                        declaration.mesh_normal_integral_functional,
                    .fluid_normal_name =
                        declaration.fluid_normal_integral_functional,
                    .gap_squared_name =
                        declaration.normal_gap_squared_integral_functional,
                    .mesh_velocity_squared_name =
                        declaration.mesh_velocity_squared_integral_functional,
                    .mesh_normal_squared_name =
                        declaration.mesh_normal_squared_integral_functional,
                    .mesh_tangential_squared_name =
                        declaration.mesh_tangential_squared_integral_functional,
                    .tangential_target_gap_squared_name =
                        declaration
                            .tangential_target_gap_squared_integral_functional,
                });
            staged_records.push_back(
                FittedALENormalOperatorStageHistoryRecord{
                    .declaration = declaration,
                    .stage = metadata,
                    .raw =
                        FittedALENormalOperatorStageRawValue{
                            .key = declaration.key,
                            .tangential_target_gap_sq =
                                declaration
                                        .prescribed_tangential_mesh_velocity
                                        .has_value()
                                    ? std::optional<Real>{Real{0.0}}
                                    : std::nullopt,
                            .stage_mesh_revision = stage_mesh_revision,
                        },
                });
        }
    } catch (...) {
        local_exception = std::current_exception();
    }
    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::stageFittedALENormalOperatorStageMeasurements/prepare")) {
        fitted_ale_normal_measurement_transaction_active_ = false;
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "another active communicator rank could not prepare the fixed "
            "measurement schedule");
    }

    local_exception = nullptr;
    try {
        // No allocation, record copy, or local validation occurs between
        // these fixed-order collective reduction calls.  Scalar validation
        // is deferred until every rank has completed the complete schedule.
        for (std::size_t index = 0u;
             index < measurement_routes.size();
             ++index) {
            auto& route = measurement_routes[index];
            auto& raw = staged_records[index].raw;
            raw.A = route.service->boundaryMeasure(
                *route.measure_functional, *measurement_state);
            raw.Wn = route.service->evaluateFunctional(
                route.mesh_normal_name, *measurement_state);
            raw.Un = route.service->evaluateFunctional(
                route.fluid_normal_name, *measurement_state);
            raw.gap_sq = route.service->evaluateFunctional(
                route.gap_squared_name, *measurement_state);
            raw.mesh_velocity_sq = route.service->evaluateFunctional(
                route.mesh_velocity_squared_name, *measurement_state);
            raw.mesh_normal_sq = route.service->evaluateFunctional(
                route.mesh_normal_squared_name, *measurement_state);
            raw.mesh_tangential_sq = route.service->evaluateFunctional(
                route.mesh_tangential_squared_name, *measurement_state);
            if (!route.tangential_target_gap_squared_name.empty()) {
                raw.tangential_target_gap_sq =
                    route.service->evaluateFunctional(
                        route.tangential_target_gap_squared_name,
                        *measurement_state);
            }
        }
        for (auto& record : staged_records) {
            auto& raw = record.raw;
            FE_THROW_IF(
                !std::isfinite(raw.A) || raw.A <= Real{0.0} ||
                    !std::isfinite(raw.Wn) || !std::isfinite(raw.Un) ||
                    !std::isfinite(raw.gap_sq) ||
                    !std::isfinite(raw.mesh_velocity_sq) ||
                    !std::isfinite(raw.mesh_normal_sq) ||
                    !std::isfinite(raw.mesh_tangential_sq) ||
                    (raw.tangential_target_gap_sq.has_value() &&
                     !std::isfinite(*raw.tangential_target_gap_sq)),
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "boundary measure and raw moments must be finite and area "
                "must be positive");
            const Real squared_moment_roundoff =
                Real{128.0} * std::numeric_limits<Real>::epsilon() *
                std::max(
                    {Real{1.0},
                     std::abs(raw.A),
                     std::abs(raw.Wn),
                     std::abs(raw.Un),
                     std::abs(raw.gap_sq),
                     std::abs(raw.mesh_velocity_sq),
                     std::abs(raw.mesh_normal_sq),
                     std::abs(raw.mesh_tangential_sq),
                     std::abs(
                         raw.tangential_target_gap_sq.value_or(
                             Real{0.0}))});
            FE_THROW_IF(
                raw.gap_sq < -squared_moment_roundoff ||
                    raw.mesh_velocity_sq < -squared_moment_roundoff ||
                    raw.mesh_normal_sq < -squared_moment_roundoff ||
                    raw.mesh_tangential_sq < -squared_moment_roundoff ||
                    (raw.tangential_target_gap_sq.has_value() &&
                     *raw.tangential_target_gap_sq <
                         -squared_moment_roundoff),
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "an integrated squared velocity moment is negative beyond "
                "roundoff");
            if (raw.gap_sq < Real{0.0}) {
                raw.gap_sq = Real{0.0};
            }
            if (raw.mesh_velocity_sq < Real{0.0}) {
                raw.mesh_velocity_sq = Real{0.0};
            }
            if (raw.mesh_normal_sq < Real{0.0}) {
                raw.mesh_normal_sq = Real{0.0};
            }
            if (raw.mesh_tangential_sq < Real{0.0}) {
                raw.mesh_tangential_sq = Real{0.0};
            }
            if (raw.tangential_target_gap_sq.has_value() &&
                *raw.tangential_target_gap_sq < Real{0.0}) {
                raw.tangential_target_gap_sq = Real{0.0};
            }
            const Real projection_identity_residual =
                raw.mesh_velocity_sq - raw.mesh_normal_sq -
                raw.mesh_tangential_sq;
            const Real projection_identity_tolerance =
                Real{1024.0} * std::numeric_limits<Real>::epsilon() *
                std::max(
                    {Real{1.0},
                     raw.mesh_velocity_sq,
                     raw.mesh_normal_sq,
                     raw.mesh_tangential_sq});
            FE_THROW_IF(
                std::abs(projection_identity_residual) >
                    projection_identity_tolerance,
                InvalidArgumentException,
                "FESystem::stageFittedALENormalOperatorStageMeasurements: "
                "mesh normal and tangential projection moments do not "
                "reconstruct the full velocity norm");
        }
        FE_THROW_IF(
            captureOperatorStageGeometry(meshAccess()) !=
                stage_mesh_revision,
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "stage geometry changed during boundary reduction evaluation");
    } catch (...) {
        local_exception = std::current_exception();
    }

    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::stageFittedALENormalOperatorStageMeasurements/evaluate")) {
        fitted_ale_normal_measurement_transaction_active_ = false;
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "another active communicator rank rejected evaluated moments");
    }

    // Reserve accepted-history storage while the candidate is still
    // rollback-capable.  Accepted-step publication can then move records
    // without allocating.
    local_exception = nullptr;
    try {
        FE_THROW_IF(
            staged_records.size() >
                fitted_ale_normal_measurement_history_.max_size() -
                    fitted_ale_normal_measurement_history_.size(),
            InvalidArgumentException,
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "accepted history size would overflow");
        fitted_ale_normal_measurement_history_.reserve(
            fitted_ale_normal_measurement_history_.size() +
            staged_records.size());
    } catch (...) {
        local_exception = std::current_exception();
    }
    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::stageFittedALENormalOperatorStageMeasurements/reserve")) {
        fitted_ale_normal_measurement_transaction_active_ = false;
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::stageFittedALENormalOperatorStageMeasurements: "
            "another active communicator rank could not reserve accepted "
            "history storage");
    }

    pending_fitted_ale_normal_measurements_ =
        std::move(staged_records);
    fitted_ale_normal_measurement_transaction_active_ = false;
}

void MeshNormalConstraintStageMeasurements::discardPendingFittedALENormalOperatorStageMeasurements()
    noexcept
{
    pending_fitted_ale_normal_measurements_.clear();
    fitted_ale_normal_measurement_transaction_active_ = false;
}

void MeshNormalConstraintStageMeasurements::commitPendingFittedALENormalOperatorStageMeasurements(
    std::uint64_t accepted_step,
    Real accepted_time,
    Real dt)
{
    const auto collective = meshBoundaryCollectiveContext(*system_);
    MeshBoundaryConsensusTokens local_tokens;
    MeshBoundaryConsensusTokens minimum_tokens;
    MeshBoundaryConsensusTokens maximum_tokens;
    std::exception_ptr local_exception;
    bool contains_opaque_callable = false;
    bool identical_replay = false;
    const auto declaration_count =
        fitted_ale_normal_measurement_declarations_.size();

    try {
        FE_THROW_IF(
            fitted_ale_normal_measurement_transaction_active_,
            InvalidArgumentException,
            "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
            "another measurement transaction is active");
        appendMeshBoundaryToken(
            local_tokens,
            static_cast<std::uint64_t>(declaration_count));
        for (const auto& declaration :
             fitted_ale_normal_measurement_declarations_) {
            appendFittedALENormalMeasurementDeclaration(
                local_tokens,
                declaration,
                contains_opaque_callable);
        }
        appendMeshBoundaryToken(
            local_tokens,
            static_cast<std::uint64_t>(
                pending_fitted_ale_normal_measurements_.size()));

        FE_THROW_IF(
            declaration_count == 0u
                ? !pending_fitted_ale_normal_measurements_.empty()
                : pending_fitted_ale_normal_measurements_.size() !=
                      declaration_count,
            InvalidArgumentException,
            "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
            "every nonempty declaration table requires one complete pending "
            "group; only an empty declaration/pending pair is a no-op");

        if (!pending_fitted_ale_normal_measurements_.empty()) {
            FE_THROW_IF(
                !isSetup() || !dofHandler().isFinalized() ||
                    !std::isfinite(accepted_time) ||
                    !std::isfinite(dt) || dt <= Real{0.0},
                InvalidArgumentException,
                "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
                "pending group, setup, and accepted metadata are invalid");
            const auto& pending_stage =
                pending_fitted_ale_normal_measurements_.front().stage;
            FE_THROW_IF(
                accepted_step !=
                        pending_stage.prospective_accepted_step ||
                    accepted_time != pending_stage.step_end_time ||
                    dt != pending_stage.dt,
                InvalidArgumentException,
                "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
                "accepted step/time/dt do not match the pending prospective "
                "stage");
            for (std::size_t index = 0u; index < declaration_count;
                 ++index) {
                const auto& pending =
                    pending_fitted_ale_normal_measurements_[index];
                const auto& declaration =
                    fitted_ale_normal_measurement_declarations_[index];
                FE_THROW_IF(
                    pending.raw.key != declaration.key ||
                        pending.declaration.key != declaration.key ||
                        pending.declaration.related_velocity_field !=
                            declaration.related_velocity_field ||
                        pending.declaration.tangential_policy !=
                            declaration.tangential_policy ||
                        pending.declaration
                                .prescribed_tangential_mesh_velocity !=
                            declaration
                                .prescribed_tangential_mesh_velocity ||
                        pending.declaration
                                .mesh_normal_integral_functional !=
                            declaration.mesh_normal_integral_functional ||
                        pending.declaration
                                .fluid_normal_integral_functional !=
                            declaration.fluid_normal_integral_functional ||
                        pending.declaration
                                .normal_gap_squared_integral_functional !=
                            declaration
                                .normal_gap_squared_integral_functional ||
                        pending.declaration
                                .mesh_velocity_squared_integral_functional !=
                            declaration
                                .mesh_velocity_squared_integral_functional ||
                        pending.declaration
                                .mesh_normal_squared_integral_functional !=
                            declaration
                                .mesh_normal_squared_integral_functional ||
                        pending.declaration
                                .mesh_tangential_squared_integral_functional !=
                            declaration
                                .mesh_tangential_squared_integral_functional ||
                        pending.declaration
                                .tangential_target_gap_squared_integral_functional !=
                            declaration
                                .tangential_target_gap_squared_integral_functional ||
                        !meshNormalDeclarationsStructurallyEqual(
                            pending.declaration.normal_constraint,
                            declaration.normal_constraint) ||
                        pending.stage != pending_stage ||
                        !pending.stage.expected_stage_geometry.has_value() ||
                        pending.raw.stage_mesh_revision !=
                            *pending.stage.expected_stage_geometry,
                    InvalidArgumentException,
                    "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
                    "pending group is inconsistent with its declaration or "
                    "shared stage provenance");
                appendFittedALENormalOperatorStageRecord(
                    local_tokens,
                    pending,
                    contains_opaque_callable);
            }
            appendMeshBoundaryToken(local_tokens, accepted_step);
            appendMeshBoundaryToken(
                local_tokens,
                generatedBoundaryTraceRealBits(accepted_time));
            appendMeshBoundaryToken(
                local_tokens, generatedBoundaryTraceRealBits(dt));

            FE_THROW_IF(
                fitted_ale_normal_measurement_history_.size() %
                        declaration_count !=
                    0u,
                InvalidArgumentException,
                "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
                "accepted history contains an incomplete group");
            if (!fitted_ale_normal_measurement_history_.empty()) {
                const auto latest_begin =
                    fitted_ale_normal_measurement_history_.size() -
                    declaration_count;
                const auto& latest_stage =
                    fitted_ale_normal_measurement_history_[latest_begin]
                        .stage;
                FE_THROW_IF(
                    accepted_step <
                        latest_stage.prospective_accepted_step,
                    InvalidArgumentException,
                    "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
                    "accepted history step must be monotone");
                if (accepted_step ==
                    latest_stage.prospective_accepted_step) {
                    identical_replay = true;
                    for (std::size_t index = 0u;
                         index < declaration_count;
                         ++index) {
                        const auto& pending =
                            pending_fitted_ale_normal_measurements_[index];
                        const auto& latest =
                            fitted_ale_normal_measurement_history_[
                                latest_begin + index];
                        MeshBoundaryConsensusTokens pending_tokens;
                        MeshBoundaryConsensusTokens latest_tokens;
                        bool pending_opaque = false;
                        bool latest_opaque = false;
                        appendFittedALENormalOperatorStageRecord(
                            pending_tokens,
                            pending,
                            pending_opaque);
                        appendFittedALENormalOperatorStageRecord(
                            latest_tokens,
                            latest,
                            latest_opaque);
                        identical_replay =
                            identical_replay &&
                            pending_tokens == latest_tokens &&
                            pending_opaque == latest_opaque &&
                            pending.stage.expected_stage_geometry ==
                                latest.stage.expected_stage_geometry &&
                            pending.raw.stage_mesh_revision ==
                                latest.raw.stage_mesh_revision;
                    }
                    FE_THROW_IF(
                        !identical_replay,
                        InvalidArgumentException,
                        "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
                        "an accepted step cannot be replayed with different "
                        "operator-stage moments or provenance");
                } else {
                    FE_THROW_IF(
                        accepted_time < latest_stage.step_end_time,
                        InvalidArgumentException,
                        "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
                        "accepted history time must be monotone");
                }
                appendMeshBoundaryToken(
                    local_tokens,
                    static_cast<std::uint64_t>(
                        fitted_ale_normal_measurement_history_.size()));
                for (std::size_t index = 0u;
                     index < declaration_count;
                     ++index) {
                    appendFittedALENormalOperatorStageRecord(
                        local_tokens,
                        fitted_ale_normal_measurement_history_[
                            latest_begin + index],
                        contains_opaque_callable);
                }
            } else {
                appendMeshBoundaryToken(local_tokens, 0u);
            }
            FE_THROW_IF(
                !identical_replay &&
                    fitted_ale_normal_measurement_history_.capacity() <
                        fitted_ale_normal_measurement_history_.size() +
                            declaration_count,
                InvalidArgumentException,
                "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
                "staging did not reserve allocation-free accepted-history "
                "storage");
        }
        appendMeshBoundaryToken(
            local_tokens, identical_replay ? 1u : 0u);
        FE_THROW_IF(
            collective.size > 1 && contains_opaque_callable,
            InvalidArgumentException,
            "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
            "distributed history may not contain opaque callbacks");
        FE_THROW_IF(
            local_tokens.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()),
            InvalidArgumentException,
            "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
            "consensus token sequence exceeds the MPI count range");
        minimum_tokens = local_tokens;
        maximum_tokens = local_tokens;
    } catch (...) {
        local_exception = std::current_exception();
    }

    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::commitPendingFittedALENormalOperatorStageMeasurements/preflight")) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
            "another active communicator rank rejected accepted history");
    }
    requireMeshBoundaryTokenConsensus(
        collective,
        local_tokens,
        minimum_tokens,
        maximum_tokens,
        "FESystem::commitPendingFittedALENormalOperatorStageMeasurements/consensus");

    if (pending_fitted_ale_normal_measurements_.empty()) {
        return;
    }
    if (identical_replay) {
        pending_fitted_ale_normal_measurements_.clear();
        return;
    }

    static_assert(
        std::is_nothrow_move_constructible_v<
            FittedALENormalOperatorStageHistoryRecord>);
    static_assert(
        std::is_nothrow_move_assignable_v<
            FittedALENormalOperatorStageHistoryRecord>);
    const auto history_size =
        fitted_ale_normal_measurement_history_.size();
    std::size_t appended_count = 0u;
    local_exception = nullptr;
    fitted_ale_normal_measurement_transaction_active_ = true;
    try {
        for (auto& record :
             pending_fitted_ale_normal_measurements_) {
            fitted_ale_normal_measurement_history_.push_back(
                std::move(record));
            ++appended_count;
        }
    } catch (...) {
        local_exception = std::current_exception();
    }
    const bool any_failed = coordinateMeshBoundaryLocalFailure(
        collective,
        local_exception != nullptr,
        "FESystem::commitPendingFittedALENormalOperatorStageMeasurements/publish");
    if (any_failed) {
        for (std::size_t index = 0u; index < appended_count;
             ++index) {
            pending_fitted_ale_normal_measurements_[index] =
                std::move(
                    fitted_ale_normal_measurement_history_[
                        history_size + index]);
        }
        fitted_ale_normal_measurement_history_.resize(history_size);
        fitted_ale_normal_measurement_transaction_active_ = false;
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::commitPendingFittedALENormalOperatorStageMeasurements: "
            "another active communicator rank failed accepted publication");
    }

    pending_fitted_ale_normal_measurements_.clear();
    fitted_ale_normal_measurement_transaction_active_ = false;
    try {
        for (std::size_t index = history_size;
             index < fitted_ale_normal_measurement_history_.size();
             ++index) {
            const auto& record =
                fitted_ale_normal_measurement_history_[index];
            std::ostringstream message;
            message << std::setprecision(17)
                    << "FESystem: accepted fitted-ALE normal measurement"
                    << " semantics=operator_stage"
                    << " raw_moments_not_work_or_dissipation=true"
                    << " scheme='" << record.stage.scheme_name << "'"
                    << " accepted_step=" << accepted_step
                    << " attempt="
                    << record.stage.prospective_attempt
                    << " state_time=" << record.stage.state_time
                    << " rate_time=" << record.stage.rate_time
                    << " mesh_displacement_field="
                    << record.raw.key.mesh_displacement_field
                    << " boundary_marker="
                    << record.raw.key.boundary_marker
                    << " A=" << record.raw.A
                    << " Wn=" << record.raw.Wn
                    << " Un=" << record.raw.Un
                    << " gap_sq=" << record.raw.gap_sq
                    << " mesh_velocity_sq="
                    << record.raw.mesh_velocity_sq
                    << " mesh_normal_sq="
                    << record.raw.mesh_normal_sq
                    << " mesh_tangential_sq="
                    << record.raw.mesh_tangential_sq
                    << " tangential_target_gap_applicable="
                    << (record.raw.tangential_target_gap_sq.has_value()
                            ? "true"
                            : "false");
            if (record.raw.tangential_target_gap_sq.has_value()) {
                message << " tangential_target_gap_sq="
                        << *record.raw.tangential_target_gap_sq
                        << " tangential_target_gap_rms="
                        << std::sqrt(
                               *record.raw.tangential_target_gap_sq /
                               record.raw.A);
            }
            message
                    << " mesh_velocity_rms="
                    << std::sqrt(
                           record.raw.mesh_velocity_sq /
                           record.raw.A)
                    << " mesh_normal_rms="
                    << std::sqrt(
                           record.raw.mesh_normal_sq /
                           record.raw.A)
                    << " mesh_tangential_rms="
                    << std::sqrt(
                           record.raw.mesh_tangential_sq /
                           record.raw.A)
                    << " normal_gap_rms="
                    << std::sqrt(record.raw.gap_sq / record.raw.A)
                    << " projection_identity_residual="
                    << record.raw.mesh_velocity_sq -
                           record.raw.mesh_normal_sq -
                           record.raw.mesh_tangential_sq
                    << " stage_mesh_geometry_revision="
                    << record.raw.stage_mesh_revision.geometry_revision
                    << " stage_mesh_topology_revision="
                    << record.raw.stage_mesh_revision.topology_revision
                    << " stage_mesh_ownership_revision="
                    << record.raw.stage_mesh_revision.ownership_revision
                    << " stage_mesh_numbering_revision="
                    << record.raw.stage_mesh_revision.numbering_revision
                    << " stage_mesh_field_layout_revision="
                    << record.raw.stage_mesh_revision.field_layout_revision
                    << " stage_mesh_label_revision="
                    << record.raw.stage_mesh_revision.label_revision
                    << " stage_mesh_active_configuration_epoch="
                    << record.raw.stage_mesh_revision
                           .active_configuration_epoch
                    << " stage_coordinate_configuration_key="
                    << record.raw.stage_mesh_revision
                           .coordinate_configuration_key
                    << " geometry_provenance_scope=rank_local";
            FE_LOG_INFO(message.str());
        }
    } catch (...) {
        // Accepted semantic history is authoritative; diagnostics are best effort.
    }
}


} // namespace svmp::FE::systems

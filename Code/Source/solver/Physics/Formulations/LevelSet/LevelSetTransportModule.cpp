/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/LevelSet/LevelSetTransportModule.h"

#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {
namespace {

[[nodiscard]] FE::FieldId resolveNamedField(
    const FE::systems::FESystem& system,
    const std::string& field_name,
    const char* context)
{
    const auto field = system.findFieldByName(field_name);
    if (field == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            std::string("LevelSetTransportModule::registerOn: missing ") +
            context + " field '" + field_name + "'");
    }
    return field;
}

[[nodiscard]] FE::systems::FieldSourceKind sourceKind(LevelSetFieldSource source) noexcept
{
    switch (source) {
    case LevelSetFieldSource::Unknown:
        return FE::systems::FieldSourceKind::Unknown;
    case LevelSetFieldSource::PrescribedData:
        return FE::systems::FieldSourceKind::PrescribedData;
    }
    return FE::systems::FieldSourceKind::Unknown;
}

[[nodiscard]] FE::systems::FieldSourceKind sourceKind(LevelSetVelocitySource source) noexcept
{
    switch (source) {
    case LevelSetVelocitySource::CoupledField:
        return FE::systems::FieldSourceKind::Unknown;
    case LevelSetVelocitySource::PrescribedData:
        return FE::systems::FieldSourceKind::PrescribedData;
    }
    return FE::systems::FieldSourceKind::Unknown;
}

[[nodiscard]] FE::FieldId ensureLevelSetField(
    FE::systems::FESystem& system,
    const LevelSetFieldOptions& options,
    std::shared_ptr<const FE::spaces::FunctionSpace> space)
{
    const auto existing = system.findFieldByName(options.field_name);
    if (existing != FE::INVALID_FIELD_ID) {
        return existing;
    }
    if (!options.auto_register_field) {
        return resolveNamedField(system, options.field_name, "level-set");
    }
    if (!space) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: auto-registering the level-set field requires a function space");
    }
    return system.addField(FE::systems::FieldSpec{
        .name = options.field_name,
        .space = std::move(space),
        .components = 1,
        .source_kind = sourceKind(options.source),
    });
}

[[nodiscard]] FE::FieldId ensureVelocityField(
    FE::systems::FESystem& system,
    const LevelSetVelocityOptions& options)
{
    const auto existing = system.findFieldByName(options.field_name);
    if (existing != FE::INVALID_FIELD_ID) {
        return existing;
    }
    if (!options.auto_register_field) {
        return resolveNamedField(system, options.field_name, "velocity");
    }
    if (!options.space) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: auto-registering the velocity field requires a function space");
    }
    return system.addField(FE::systems::FieldSpec{
        .name = options.field_name,
        .space = options.space,
        .components = options.space->value_dimension(),
        .source_kind = sourceKind(options.source),
    });
}

void validateScalarField(const FE::systems::FESystem& system,
                         FE::FieldId field,
                         const std::string& field_name)
{
    const auto& rec = system.fieldRecord(field);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: level-set field '" +
            field_name + "' must be scalar");
    }
}

void validateVelocityField(const FE::systems::FESystem& system,
                           FE::FieldId field,
                           const std::string& field_name)
{
    const auto& rec = system.fieldRecord(field);
    if (!rec.space || rec.space->value_dimension() < 1) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: velocity field '" +
            field_name + "' must have a vector function space");
    }
}

void validateBoundaryOptions(const LevelSetBoundaryOptions& boundaries)
{
    std::unordered_set<int> markers;
    markers.reserve(boundaries.inflow.size() + boundaries.outflow.size());

    for (const auto& bc : boundaries.inflow) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "LevelSetTransportModule::registerOn: inflow boundary");
        if (!markers.insert(marker).second) {
            throw std::invalid_argument(
                "LevelSetTransportModule::registerOn: duplicate level-set boundary marker");
        }
        if (!(bc.penalty_scale > 0.0)) {
            throw std::invalid_argument(
                "LevelSetTransportModule::registerOn: inflow boundary penalty_scale must be positive");
        }
    }

    for (const auto& bc : boundaries.outflow) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "LevelSetTransportModule::registerOn: outflow boundary");
        if (!markers.insert(marker).second) {
            throw std::invalid_argument(
                "LevelSetTransportModule::registerOn: duplicate level-set boundary marker");
        }
    }
}

} // namespace

LevelSetTransportModule::LevelSetTransportModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space,
    LevelSetTransportOptions options)
    : level_set_space_(std::move(level_set_space))
    , options_(std::move(options))
{
}

void LevelSetTransportModule::registerOn(FE::systems::FESystem& system) const
{
    if (options_.level_set.field_name.empty()) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: level-set field name must be non-empty");
    }
    if (options_.velocity.field_name.empty()) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: velocity field name must be non-empty");
    }
    if (level_set_space_ && level_set_space_->value_dimension() != 1) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: level-set field space must be scalar");
    }
    if (options_.supg.enabled && !(options_.supg.tau_scale > 0.0)) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: SUPG tau_scale must be positive");
    }
    if (options_.supg.enabled && !(options_.supg.velocity_epsilon > 0.0)) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: SUPG velocity_epsilon must be positive");
    }
    validateBoundaryOptions(options_.boundaries);

    const auto phi_id = ensureLevelSetField(system, options_.level_set, level_set_space_);
    validateScalarField(system, phi_id, options_.level_set.field_name);
    if (!system.fieldParticipatesInUnknownVector(phi_id)) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: level-set field must be an unknown for transport residual assembly");
    }

    const auto velocity_id = ensureVelocityField(system, options_.velocity);
    validateVelocityField(system, velocity_id, options_.velocity.field_name);
    if (options_.velocity.source == LevelSetVelocitySource::CoupledField &&
        !system.fieldParticipatesInUnknownVector(velocity_id)) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: coupled velocity source must be an unknown field");
    }
    if (options_.velocity.source == LevelSetVelocitySource::PrescribedData &&
        system.fieldParticipatesInUnknownVector(velocity_id)) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: prescribed velocity source must not be an unknown field");
    }

    const auto& phi_rec = system.fieldRecord(phi_id);
    const auto& velocity_rec = system.fieldRecord(velocity_id);

    using namespace FE::forms;
    const auto phi = StateField(phi_id, *phi_rec.space, options_.level_set.field_name);
    const auto eta = TestField(phi_id, *phi_rec.space, "eta");
    const auto velocity =
        options_.velocity.source == LevelSetVelocitySource::CoupledField
            ? StateField(velocity_id, *velocity_rec.space, options_.velocity.field_name)
            : FormExpr::discreteField(
                  velocity_id,
                  *velocity_rec.space,
                  options_.velocity.field_name);

    const auto strong_residual = dt(phi) + dot(velocity, grad(phi));
    auto residual = (strong_residual * eta).dx();
    if (options_.supg.enabled) {
        const auto velocity_norm = sqrt(
            dot(velocity, velocity) +
            FormExpr::constant(options_.supg.velocity_epsilon));
        const auto tau =
            FormExpr::constant(options_.supg.tau_scale) * h() / velocity_norm;
        residual = residual +
                   (tau * dot(velocity, grad(eta)) * strong_residual).dx();
    }

    for (const auto& bc : options_.boundaries.inflow) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "LevelSetTransportModule::registerOn: inflow boundary");
        const auto normal_velocity = dot(velocity, FormExpr::normal());
        const auto inflow_speed =
            FormExpr::constant(0.5) * (abs(normal_velocity) - normal_velocity);
        const auto target = FE::forms::bc::toScalarExpr(
            bc.value,
            FE::forms::bc::markerValueName("level_set_inflow", marker));
        const auto penalty = FormExpr::constant(bc.penalty_scale) * inflow_speed;
        residual = residual + (penalty * (phi - target) * eta).ds(marker);
    }

    if (!system.hasOperator("level_set")) {
        system.addOperator("level_set");
    }

    auto install = physicsInstallOptions(options_.jit_policy);
    install.compiler_options.use_symbolic_tangent = true;
    if (options_.velocity.source == LevelSetVelocitySource::CoupledField) {
        install.extra_trial_fields.push_back(velocity_id);
    }
    (void)FE::systems::installFormulation(
        system,
        "level_set",
        {phi_id},
        residual,
        install);
}

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp

#include "LevelSet/LevelSetTransport.h"

#include "Forms/Vocabulary.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] FieldId resolveNamedField(
    const systems::FESystem& system,
    const std::string& field_name,
    const char* context)
{
    const auto field = system.findFieldByName(field_name);
    if (field == INVALID_FIELD_ID) {
        throw std::invalid_argument(
            std::string("installLevelSetTransport: missing ") +
            context + " field '" + field_name + "'");
    }
    return field;
}

[[nodiscard]] systems::FieldSourceKind sourceKind(LevelSetFieldSource source) noexcept
{
    switch (source) {
    case LevelSetFieldSource::Unknown:
        return systems::FieldSourceKind::Unknown;
    case LevelSetFieldSource::PrescribedData:
        // The source describes initialization; transport still owns the field.
        return systems::FieldSourceKind::Unknown;
    }
    return systems::FieldSourceKind::Unknown;
}

[[nodiscard]] systems::FieldSourceKind sourceKind(LevelSetVelocitySource source) noexcept
{
    switch (source) {
    case LevelSetVelocitySource::CoupledField:
        return systems::FieldSourceKind::Unknown;
    case LevelSetVelocitySource::PrescribedData:
    case LevelSetVelocitySource::ConstantVector:
        return systems::FieldSourceKind::PrescribedData;
    }
    return systems::FieldSourceKind::Unknown;
}

[[nodiscard]] FieldId ensureLevelSetField(
    systems::FESystem& system,
    const LevelSetFieldOptions& options,
    std::shared_ptr<const spaces::FunctionSpace> space)
{
    const auto existing = system.findFieldByName(options.field_name);
    if (existing != INVALID_FIELD_ID) {
        return existing;
    }
    if (!options.auto_register_field) {
        return resolveNamedField(system, options.field_name, "level-set");
    }
    if (!space) {
        throw std::invalid_argument(
            "installLevelSetTransport: auto-registering the level-set field requires a function space");
    }
    return system.addField(systems::FieldSpec{
        .name = options.field_name,
        .space = std::move(space),
        .components = 1,
        .source_kind = sourceKind(options.source),
    });
}

[[nodiscard]] FieldId ensureVelocityField(
    systems::FESystem& system,
    const LevelSetVelocityOptions& options)
{
    const auto existing = system.findFieldByName(options.field_name);
    if (existing != INVALID_FIELD_ID) {
        return existing;
    }
    if (!options.auto_register_field) {
        return resolveNamedField(system, options.field_name, "velocity");
    }
    if (!options.space) {
        throw std::invalid_argument(
            "installLevelSetTransport: auto-registering the velocity field requires a function space");
    }
    return system.addField(systems::FieldSpec{
        .name = options.field_name,
        .space = options.space,
        .components = options.space->value_dimension(),
        .source_kind = sourceKind(options.source),
    });
}

void validateScalarField(const systems::FESystem& system,
                         FieldId field,
                         const std::string& field_name)
{
    const auto& rec = system.fieldRecord(field);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field '" +
            field_name + "' must be scalar");
    }
}

void validateVelocityField(const systems::FESystem& system,
                           FieldId field,
                           const std::string& field_name)
{
    const auto& rec = system.fieldRecord(field);
    if (!rec.space || rec.space->value_dimension() < 1) {
        throw std::invalid_argument(
            "installLevelSetTransport: velocity field '" +
            field_name + "' must have a vector function space");
    }
}

void validateBoundaryOptions(const LevelSetBoundaryOptions& boundaries)
{
    std::unordered_set<int> markers;
    markers.reserve(boundaries.inflow.size() + boundaries.outflow.size());

    for (const auto& bc : boundaries.inflow) {
        const int marker = forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "installLevelSetTransport: inflow boundary");
        if (!markers.insert(marker).second) {
            throw std::invalid_argument(
                "installLevelSetTransport: duplicate level-set boundary marker");
        }
        if (!(bc.penalty_scale > 0.0)) {
            throw std::invalid_argument(
                "installLevelSetTransport: inflow boundary penalty_scale must be positive");
        }
    }

    for (const auto& bc : boundaries.outflow) {
        const int marker = forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "installLevelSetTransport: outflow boundary");
        if (!markers.insert(marker).second) {
            throw std::invalid_argument(
                "installLevelSetTransport: duplicate level-set boundary marker");
        }
    }
}

void validateReinitializationOptions(const LevelSetReinitializationOptions& options)
{
    if (!options.enabled) {
        return;
    }
    if (options.cadence_steps <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization cadence_steps must be positive");
    }
    if (options.max_iterations <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization max_iterations must be positive");
    }
    if (!(options.pseudo_time_step_scale > 0.0)) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization pseudo_time_step_scale must be positive");
    }
    if (!(options.interface_band_width > 0.0)) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization interface_band_width must be positive");
    }
    if (!(options.signed_distance_tolerance > 0.0)) {
        throw std::invalid_argument(
            "installLevelSetTransport: reinitialization signed_distance_tolerance must be positive");
    }
}

void validateVolumeCorrectionOptions(const LevelSetVolumeCorrectionOptions& options)
{
    if (!options.enabled) {
        return;
    }
    if (options.cadence_steps <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction cadence_steps must be positive");
    }
    if (!(options.volume_tolerance > 0.0)) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction volume_tolerance must be positive");
    }
    if (options.max_iterations <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction max_iterations must be positive");
    }
    if (!options.use_initial_negative_volume_as_target &&
        options.target_negative_volume < 0.0) {
        throw std::invalid_argument(
            "installLevelSetTransport: volume correction target_negative_volume must be nonnegative");
    }
}

} // namespace

bool shouldReinitializeLevelSet(
    const LevelSetReinitializationOptions& options,
    int completed_step_index) noexcept
{
    return options.enabled &&
           options.cadence_steps > 0 &&
           completed_step_index > 0 &&
           completed_step_index % options.cadence_steps == 0;
}

bool shouldApplyLevelSetVolumeCorrection(
    const LevelSetVolumeCorrectionOptions& options,
    int completed_step_index) noexcept
{
    return options.enabled &&
           options.cadence_steps > 0 &&
           completed_step_index > 0 &&
           completed_step_index % options.cadence_steps == 0;
}

systems::CoupledResidualKernels installLevelSetTransport(
    systems::FESystem& system,
    std::shared_ptr<const spaces::FunctionSpace> level_set_space,
    const LevelSetTransportOptions& options,
    const systems::FormInstallOptions& install_options)
{
    if (options.level_set.field_name.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field name must be non-empty");
    }
    if (options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        options.velocity.field_name.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: velocity field name must be non-empty");
    }
    if (level_set_space && level_set_space->value_dimension() != 1) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field space must be scalar");
    }
    if (options.supg.enabled && !(options.supg.tau_scale > 0.0)) {
        throw std::invalid_argument(
            "installLevelSetTransport: SUPG tau_scale must be positive");
    }
    if (options.supg.enabled && !(options.supg.velocity_epsilon > 0.0)) {
        throw std::invalid_argument(
            "installLevelSetTransport: SUPG velocity_epsilon must be positive");
    }
    validateReinitializationOptions(options.reinitialization);
    validateVolumeCorrectionOptions(options.volume_correction);
    validateBoundaryOptions(options.boundaries);

    const auto phi_id = ensureLevelSetField(system, options.level_set, std::move(level_set_space));
    validateScalarField(system, phi_id, options.level_set.field_name);
    if (!system.fieldParticipatesInUnknownVector(phi_id)) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field must be an unknown for transport residual assembly");
    }

    FieldId velocity_id = INVALID_FIELD_ID;
    if (options.velocity.source != LevelSetVelocitySource::ConstantVector) {
        velocity_id = ensureVelocityField(system, options.velocity);
        validateVelocityField(system, velocity_id, options.velocity.field_name);
        if (options.velocity.source == LevelSetVelocitySource::CoupledField &&
            !system.fieldParticipatesInUnknownVector(velocity_id)) {
            throw std::invalid_argument(
                "installLevelSetTransport: coupled velocity source must be an unknown field");
        }
        if (options.velocity.source == LevelSetVelocitySource::PrescribedData &&
            system.fieldParticipatesInUnknownVector(velocity_id)) {
            throw std::invalid_argument(
                "installLevelSetTransport: prescribed velocity source must not be an unknown field");
        }
    }

    const auto& phi_rec = system.fieldRecord(phi_id);

    using namespace forms;
    const auto phi = StateField(phi_id, *phi_rec.space, options.level_set.field_name);
    const auto eta = TestField(phi_id, *phi_rec.space, "eta");
    FormExpr velocity;
    if (options.velocity.source == LevelSetVelocitySource::ConstantVector) {
        const int dim = phi_rec.space ? phi_rec.space->topological_dimension() : 0;
        if (dim < 1 || dim > 3) {
            throw std::invalid_argument(
                "installLevelSetTransport: constant velocity requires level-set space dimension in [1, 3]");
        }
        std::vector<FormExpr> components;
        components.reserve(static_cast<std::size_t>(dim));
        for (int d = 0; d < dim; ++d) {
            components.push_back(
                FormExpr::constant(options.velocity.constant_value[static_cast<std::size_t>(d)]));
        }
        velocity = FormExpr::asVector(std::move(components));
    } else {
        const auto& velocity_rec = system.fieldRecord(velocity_id);
        velocity = options.velocity.source == LevelSetVelocitySource::CoupledField
                       ? StateField(velocity_id, *velocity_rec.space, options.velocity.field_name)
                       : FormExpr::discreteField(
                             velocity_id,
                             *velocity_rec.space,
                             options.velocity.field_name);
    }

    const auto strong_residual = dt(phi) + dot(velocity, grad(phi));
    auto residual = (strong_residual * eta).dx();
    if (options.supg.enabled) {
        const auto velocity_norm = sqrt(
            dot(velocity, velocity) +
            FormExpr::constant(options.supg.velocity_epsilon));
        const auto tau =
            FormExpr::constant(options.supg.tau_scale) * h() / velocity_norm;
        residual = residual +
                   (tau * dot(velocity, grad(eta)) * strong_residual).dx();
    }

    for (const auto& bc : options.boundaries.inflow) {
        const int marker = forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "installLevelSetTransport: inflow boundary");
        const auto normal_velocity = dot(velocity, FormExpr::normal());
        const auto inflow_speed =
            FormExpr::constant(0.5) * (abs(normal_velocity) - normal_velocity);
        const auto target = forms::bc::toScalarExpr(
            bc.value,
            forms::bc::markerValueName("level_set_inflow", marker));
        const auto penalty = FormExpr::constant(bc.penalty_scale) * inflow_speed;
        residual = residual + (penalty * (phi - target) * eta).ds(marker);
    }

    if (!system.hasOperator("level_set")) {
        system.addOperator("level_set");
    }

    auto install = install_options;
    install.compiler_options.use_symbolic_tangent = true;
    if (options.velocity.source == LevelSetVelocitySource::CoupledField) {
        install.extra_trial_fields.push_back(velocity_id);
    }
    return systems::installFormulation(
        system,
        "level_set",
        {phi_id},
        residual,
        install);
}

} // namespace svmp::FE::level_set

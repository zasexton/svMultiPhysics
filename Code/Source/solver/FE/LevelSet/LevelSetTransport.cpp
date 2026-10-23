#include "LevelSet/LevelSetTransport.h"
#include "LevelSet/LevelSetVelocityExtensionConstraint.h"

#include "Forms/Vocabulary.h"
#include "Interfaces/MaterialInterfaceTransportVelocity.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
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

[[nodiscard]] const interfaces::
    MaterialInterfaceTransportVelocityDeclaration&
requireMaterialInterfaceVelocityDeclaration(
    const systems::FESystem& system,
    FieldId level_set_field,
    int interface_marker)
{
    const auto declarations =
        system.materialInterfaceTransportVelocityDeclarations();
    const interfaces::MaterialInterfaceTransportVelocityDeclaration*
        found = nullptr;
    for (const auto& declaration : declarations) {
        if (declaration.level_set_field != level_set_field ||
            declaration.interface_marker != interface_marker) {
            continue;
        }
        if (found != nullptr) {
            throw std::invalid_argument(
                "installLevelSetTransport: material-interface phase-pair velocity has more than one matching owner");
        }
        found = &declaration;
    }
    if (found == nullptr) {
        throw std::invalid_argument(
            "installLevelSetTransport: material-interface phase-pair velocity has no matching declaration");
    }
    return *found;
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
    case LevelSetVelocitySource::MaterialInterfacePhasePair:
        return systems::FieldSourceKind::Unknown;
    }
    return systems::FieldSourceKind::Unknown;
}

struct PendingTransportField {
    FieldId existing_id{INVALID_FIELD_ID};
    std::shared_ptr<const spaces::FunctionSpace> space{};
    int components{0};
    systems::FieldSourceKind source_kind{systems::FieldSourceKind::Unknown};

    [[nodiscard]] bool needsRegistration() const noexcept
    {
        return existing_id == INVALID_FIELD_ID;
    }
};

[[nodiscard]] PendingTransportField preflightLevelSetField(
    const systems::FESystem& system,
    const LevelSetFieldOptions& options,
    const std::shared_ptr<const spaces::FunctionSpace>& requested_space)
{
    const auto existing = system.findFieldByName(options.field_name);
    if (existing != INVALID_FIELD_ID) {
        const auto& record = system.fieldRecord(existing);
        if (record.components != 1 || !record.space ||
            record.space->value_dimension() != 1) {
            throw std::invalid_argument(
                "installLevelSetTransport: level-set field '" +
                options.field_name + "' must be scalar");
        }
        if (!system.fieldParticipatesInUnknownVector(existing)) {
            throw std::invalid_argument(
                "installLevelSetTransport: level-set field must be an unknown for transport residual assembly");
        }
        return PendingTransportField{
            .existing_id = existing,
            .space = record.space,
            .components = record.components,
            .source_kind = record.source_kind,
        };
    }

    if (!options.auto_register_field) {
        (void)resolveNamedField(system, options.field_name, "level-set");
    }
    if (!requested_space) {
        throw std::invalid_argument(
            "installLevelSetTransport: auto-registering the level-set field requires a function space");
    }
    if (requested_space->value_dimension() != 1) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field space must be scalar");
    }
    return PendingTransportField{
        .space = requested_space,
        .components = 1,
        .source_kind = sourceKind(options.source),
    };
}

[[nodiscard]] PendingTransportField preflightVelocityField(
    const systems::FESystem& system,
    const LevelSetVelocityOptions& options,
    int expected_dimension)
{
    const auto existing = system.findFieldByName(options.field_name);
    PendingTransportField pending;
    if (existing != INVALID_FIELD_ID) {
        const auto& record = system.fieldRecord(existing);
        pending = PendingTransportField{
            .existing_id = existing,
            .space = record.space,
            .components = record.components,
            .source_kind = record.source_kind,
        };
    } else {
        if (!options.auto_register_field) {
            (void)resolveNamedField(system, options.field_name, "velocity");
        }
        if (!options.space) {
            throw std::invalid_argument(
                "installLevelSetTransport: auto-registering the velocity field requires a function space");
        }
        pending.space = options.space;
        pending.components = options.space->value_dimension();
        pending.source_kind = sourceKind(options.source);
    }

    if (!pending.space || pending.components != expected_dimension ||
        pending.space->value_dimension() != expected_dimension) {
        throw std::invalid_argument(
            "installLevelSetTransport: velocity field '" +
            options.field_name + "' must have exactly " +
            std::to_string(expected_dimension) + " component(s)");
    }
    if (pending.space->topological_dimension() != expected_dimension) {
        throw std::invalid_argument(
            "installLevelSetTransport: velocity and level-set dimensions differ");
    }
    if (existing != INVALID_FIELD_ID) {
        const bool is_unknown =
            system.fieldParticipatesInUnknownVector(existing);
        if (options.source == LevelSetVelocitySource::CoupledField &&
            !is_unknown) {
            throw std::invalid_argument(
                "installLevelSetTransport: coupled velocity source must be an unknown field");
        }
        if (options.source == LevelSetVelocitySource::PrescribedData &&
            is_unknown) {
            throw std::invalid_argument(
                "installLevelSetTransport: prescribed velocity source must not be an unknown field");
        }
    }
    return pending;
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

[[nodiscard]] bool isFixedP1ScalarH1(
    const PendingTransportField& field) noexcept
{
    return field.components == 1 && field.space &&
           field.space->space_type() == spaces::SpaceType::H1 &&
           field.space->field_type() == FieldType::Scalar &&
           field.space->continuity() == Continuity::C0 &&
           field.space->value_dimension() == 1 &&
           !field.space->is_variable_order() &&
           field.space->polynomial_order() == 1;
}

void validateConservativePhaseOptions(
    const LevelSetTransportOptions& options)
{
    const auto& phase = options.conservative_phase;
    if (!phase.enabled) {
        return;
    }
    if (phase.liquid_indicator.field_name.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase field name must be non-empty");
    }
    if (phase.liquid_indicator.source != LevelSetFieldSource::Unknown) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase field must be an unknown state");
    }
    if (phase.boundary_flux_policy !=
            LevelSetConservativePhaseBoundaryFluxPolicy::
                ClosedDomainDiscreteQFluxOnly &&
        phase.boundary_flux_policy !=
            LevelSetConservativePhaseBoundaryFluxPolicy::
                GloballyBalancedDiscreteQFlux) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase boundary-flux policy is unsupported");
    }
    if (!std::isfinite(phase.invariant_tolerance) ||
        phase.invariant_tolerance < Real{0.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase invariant tolerance must be finite and nonnegative");
    }
    if (!std::isfinite(phase.component_activity_tolerance) ||
        !(phase.component_activity_tolerance > Real{0.0}) ||
        phase.component_activity_tolerance > Real{1.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase component activity tolerance must be in (0, 1]");
    }
    if (!std::isfinite(phase.maximum_courant) ||
        !(phase.maximum_courant > Real{0.0}) ||
        phase.maximum_courant > Real{1.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase maximum Courant number must be in (0, 1]");
    }
    if (phase.flux_artifact_cadence_steps <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase flux artifact cadence must be positive");
    }
    if (!std::isfinite(phase.momentum_relative_tolerance) ||
        !(phase.momentum_relative_tolerance > Real{0.0})) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase momentum relative tolerance must be positive and finite");
    }
    try {
        static_cast<void>(makeAxisAlignedBoxPhaseRegions(
            phase.fixed_flux_regions,
            std::span<const std::array<Real, 3>>{}));
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            std::string(
                "installLevelSetTransport: invalid conservative phase fixed flux region: ") +
            error.what());
    }
    if (!std::isfinite(phase.impermeable_normal_velocity_tolerance) ||
        phase.impermeable_normal_velocity_tolerance < Real{0.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase wall-normal velocity tolerance must be finite and nonnegative");
    }
    if (phase
            .pointwise_impermeable_velocity_tolerance_explicitly_requested) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase pointwise velocity-normal wall enforcement is unsupported; the configured boundary policy checks only discrete q-flux at invariant_tolerance and can be blind where q=0");
    }
    if (!std::isfinite(phase.geometry_measure_tolerance) ||
        !(phase.geometry_measure_tolerance > Real{0.0})) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase geometry measure tolerance must be positive and finite");
    }
    if (phase.geometry_correction_max_iterations <= 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase geometry correction requires positive max iterations");
    }
    if (!std::isfinite(phase.maximum_geometry_displacement_fraction) ||
        !(phase.maximum_geometry_displacement_fraction > Real{0.0}) ||
        phase.maximum_geometry_displacement_fraction > Real{1.0}) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase geometry displacement fraction must be in (0, 1]");
    }
    if (!options.boundaries.inflow.empty() ||
        !options.boundaries.outflow.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase transport does not support declared inflow or outflow data; only its configured discrete q-flux policy is available");
    }
}

void validateInterfaceKinematicOptions(const LevelSetInterfaceKinematicOptions& options)
{
    if (!options.enabled) {
        return;
    }
    if (options.interface_marker < 0) {
        throw std::invalid_argument(
            "installLevelSetTransport: interface kinematic marker must be nonnegative when enabled");
    }
    if (!(options.weight_scale > 0.0)) {
        throw std::invalid_argument(
            "installLevelSetTransport: interface kinematic weight_scale must be positive when enabled");
    }
}

} // namespace

LevelSetConservationDiagnostic levelSetConservationDiagnostic(
    LevelSetTransportForm transport_form) noexcept
{
    if (transport_form == LevelSetTransportForm::ConservativeDivergence) {
        return LevelSetConservationDiagnostic::ConservativeDivergenceAdvectionNotLocallyConservative;
    }
    return LevelSetConservationDiagnostic::PlainAdvectionNotConservative;
}

LevelSetConservationDiagnostic levelSetConservationDiagnostic(
    const LevelSetTransportOptions& options) noexcept
{
    if (options.conservative_phase.enabled) {
        return LevelSetConservationDiagnostic::
            ConservativePhaseIndicatorLocallyConservative;
    }
    return levelSetConservationDiagnostic(options.transport_form);
}

const char* levelSetConservationDiagnosticName(
    LevelSetConservationDiagnostic diagnostic) noexcept
{
    switch (diagnostic) {
    case LevelSetConservationDiagnostic::PlainAdvectionNotConservative:
        return "plain_level_set_advection_not_conservative";
    case LevelSetConservationDiagnostic::ConservativeDivergenceAdvectionNotLocallyConservative:
        return "conservative_divergence_level_set_advection_not_locally_conservative";
    case LevelSetConservationDiagnostic::ConservativePhaseIndicatorLocallyConservative:
        return "conservative_p1_phase_indicator_locally_conservative";
    }
    return "unknown_level_set_conservation";
}

bool shouldApplyLevelSetKinematicReconciliation(
    const LevelSetKinematicReconciliationOptions& options,
    int completed_step_index) noexcept
{
    return options.enabled && completed_step_index > 0;
}

bool shouldApplyLevelSetSignDefinitePatchBounds(
    const LevelSetSignDefinitePatchBoundsOptions& options,
    int completed_step_index) noexcept
{
    return options.enabled && completed_step_index > 0;
}

bool shouldApplyLevelSetInterfacePreservingRedistancing(
    const LevelSetInterfacePreservingRedistancingOptions& options,
    int completed_step_index) noexcept
{
    return options.enabled && completed_step_index > 0;
}

systems::CoupledResidualKernels installLevelSetTransport(
    systems::FESystem& system,
    std::shared_ptr<const spaces::FunctionSpace> level_set_space,
    const LevelSetTransportOptions& options,
    const systems::FormInstallOptions& install_options)
{
    const bool material_interface_velocity =
        options.velocity.source ==
        LevelSetVelocitySource::MaterialInterfacePhasePair;
    if (options.level_set.field_name.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field name must be non-empty");
    }
    if (options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity &&
        options.velocity.field_name.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: velocity field name must be non-empty");
    }
    if (level_set_space && level_set_space->value_dimension() != 1) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set field space must be scalar");
    }
    if (options.supg.enabled &&
        (!(options.supg.tau_scale > 0.0) ||
         !std::isfinite(options.supg.tau_scale))) {
        throw std::invalid_argument(
            "installLevelSetTransport: SUPG tau_scale must be positive");
    }
    if (options.supg.enabled &&
        (!(options.supg.velocity_epsilon > 0.0) ||
         !std::isfinite(options.supg.velocity_epsilon))) {
        throw std::invalid_argument(
            "installLevelSetTransport: SUPG velocity_epsilon must be positive");
    }
    if (options.supg.enabled &&
        (!(options.supg.transient_scale > 0.0) ||
         !std::isfinite(options.supg.transient_scale))) {
        throw std::invalid_argument(
            "installLevelSetTransport: SUPG transient_scale must be positive");
    }
    validateInterfaceKinematicOptions(options.interface_kinematic);
    validateBoundaryOptions(options.boundaries);
    validateConservativePhaseOptions(options);
    if (options.operator_tag.empty()) {
        throw std::invalid_argument(
            "installLevelSetTransport: operator_tag must be non-empty");
    }

    if (options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity &&
        options.level_set.field_name == options.velocity.field_name) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set and velocity fields must be distinct");
    }
    if (options.conservative_phase.enabled &&
        options.conservative_phase.liquid_indicator.field_name ==
            options.level_set.field_name) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set and conservative phase fields must be distinct");
    }
    if (options.conservative_phase.enabled &&
        options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity &&
        options.conservative_phase.liquid_indicator.field_name ==
            options.velocity.field_name) {
        throw std::invalid_argument(
            "installLevelSetTransport: conservative phase and velocity fields must be distinct");
    }

    // Complete the definition preflight before adding either field.  In
    // particular, a missing/incompatible velocity or extension source must not
    // leave behind an auto-registered level-set field.
    const auto pending_phi = preflightLevelSetField(
        system, options.level_set, level_set_space);
    const int dimension = pending_phi.space
        ? pending_phi.space->topological_dimension()
        : 0;
    if (dimension < 1 || dimension > 3) {
        throw std::invalid_argument(
            "installLevelSetTransport: level-set space dimension must be in [1, 3]");
    }

    if (material_interface_velocity &&
        (pending_phi.existing_id == INVALID_FIELD_ID ||
         options.velocity.material_interface_marker < 0 ||
         options.velocity.auto_register_field || options.velocity.space ||
         !options.velocity.algebraic_extension_source_field_name.empty() ||
         options.transport_form != LevelSetTransportForm::Advective ||
         !options.conservative_phase.enabled ||
         !options.conservative_phase.reconcile_geometry ||
         !options.interface_kinematic.enabled ||
         options.interface_kinematic.interface_marker !=
             options.velocity.material_interface_marker ||
         !options.boundaries.inflow.empty() ||
         !options.boundaries.outflow.empty())) {
        throw std::invalid_argument(
            "installLevelSetTransport: material-interface phase-pair velocity requires an existing level-set field, one matching interface marker, advective-form conservative phase transport with geometry reconciliation, interface kinematic enforcement, no separate velocity registration or extension, and no declared inflow or outflow boundary");
    }

    std::optional<PendingTransportField> pending_phase;
    if (options.conservative_phase.enabled) {
        pending_phase = preflightLevelSetField(
            system,
            options.conservative_phase.liquid_indicator,
            pending_phi.space);
        if (!isFixedP1ScalarH1(pending_phi) ||
            !isFixedP1ScalarH1(*pending_phase) ||
            pending_phase->space->topological_dimension() != dimension) {
            throw std::invalid_argument(
                "installLevelSetTransport: conservative phase transport requires fixed scalar P1 H1 level-set and phase fields on the same mesh dimension");
        }
    }

    std::optional<PendingTransportField> pending_velocity;
    if (options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity) {
        pending_velocity = preflightVelocityField(
            system, options.velocity, dimension);
    }

    const interfaces::MaterialInterfaceTransportVelocityDeclaration*
        material_velocity_declaration = nullptr;
    if (material_interface_velocity) {
        material_velocity_declaration =
            &requireMaterialInterfaceVelocityDeclaration(
                system,
                pending_phi.existing_id,
                options.velocity.material_interface_marker);
        if (material_velocity_declaration->dimension != dimension) {
            throw std::invalid_argument(
                "installLevelSetTransport: material-interface phase-pair velocity dimension differs from the level-set space");
        }
    }

    FieldId algebraic_extension_source_id = INVALID_FIELD_ID;
    if (!options.velocity.algebraic_extension_source_field_name.empty()) {
        if (options.velocity.source != LevelSetVelocitySource::CoupledField) {
            throw std::invalid_argument(
                "installLevelSetTransport: a state-dependent algebraic velocity extension must be a CoupledField unknown");
        }
        if (options.velocity.algebraic_extension_source_field_name ==
            options.velocity.field_name) {
            throw std::invalid_argument(
                "installLevelSetTransport: algebraic extension and physical velocity fields must be distinct");
        }
        if (options.conservative_phase.enabled &&
            options.velocity.algebraic_extension_source_field_name ==
                options.conservative_phase.liquid_indicator.field_name) {
            throw std::invalid_argument(
                "installLevelSetTransport: conservative phase and physical velocity fields must be distinct");
        }
        algebraic_extension_source_id = system.findFieldByName(
            options.velocity.algebraic_extension_source_field_name);
        if (algebraic_extension_source_id == INVALID_FIELD_ID ||
            !system.hasField(
                options.velocity.algebraic_extension_source_field_name)) {
            throw std::invalid_argument(
                "installLevelSetTransport: algebraic extension references unknown physical velocity field '" +
                options.velocity.algebraic_extension_source_field_name + "'");
        }
        validateVelocityField(
            system,
            algebraic_extension_source_id,
            options.velocity.algebraic_extension_source_field_name);
        if (!system.fieldParticipatesInUnknownVector(
                algebraic_extension_source_id)) {
            throw std::invalid_argument(
                "installLevelSetTransport: algebraic extension physical velocity must be an unknown field");
        }
        const auto& source_record =
            system.fieldRecord(algebraic_extension_source_id);
        if (!source_record.space || source_record.components != dimension ||
            source_record.space->value_dimension() != dimension ||
            source_record.space->topological_dimension() != dimension) {
            throw std::invalid_argument(
                "installLevelSetTransport: algebraic extension and physical velocity dimensions differ");
        }
        if (pending_velocity.has_value() &&
            pending_velocity->existing_id != INVALID_FIELD_ID &&
            findLevelSetVelocityExtensionConstraintKernel(
                system,
                options.operator_tag,
                pending_velocity->existing_id)) {
            throw std::invalid_argument(
                "installLevelSetTransport: duplicate algebraic velocity extension constraint for field '" +
                options.velocity.field_name + "'");
        }
    }

    const auto phi_id = ensureLevelSetField(
        system, options.level_set, std::move(level_set_space));

    FieldId phase_id = INVALID_FIELD_ID;
    if (options.conservative_phase.enabled) {
        phase_id = ensureLevelSetField(
            system,
            options.conservative_phase.liquid_indicator,
            pending_phase->space);
    }

    FieldId velocity_id = INVALID_FIELD_ID;
    if (options.velocity.source != LevelSetVelocitySource::ConstantVector &&
        !material_interface_velocity) {
        velocity_id = ensureVelocityField(system, options.velocity);
    }

    const auto& phi_rec = system.fieldRecord(phi_id);

    using namespace forms;
    const auto phi = StateField(phi_id, *phi_rec.space, options.level_set.field_name);
    const auto eta = TestField(phi_id, *phi_rec.space, "eta");
    FormExpr velocity;
    FormExpr negative_material_velocity;
    FormExpr positive_material_velocity;
    FormExpr material_interface_trace_velocity;
    if (material_interface_velocity) {
        const auto& negative_record = system.fieldRecord(
            material_velocity_declaration->negative_velocity_field);
        const auto& positive_record = system.fieldRecord(
            material_velocity_declaration->positive_velocity_field);
        negative_material_velocity = StateField(
            material_velocity_declaration->negative_velocity_field,
            *negative_record.space,
            negative_record.name);
        positive_material_velocity = StateField(
            material_velocity_declaration->positive_velocity_field,
            *positive_record.space,
            positive_record.name);
        material_interface_trace_velocity =
            FormExpr::constant(
                material_velocity_declaration->negative_trace_weight) *
                negative_material_velocity +
            FormExpr::constant(
                material_velocity_declaration->positive_trace_weight) *
                positive_material_velocity;
    } else if (options.velocity.source ==
               LevelSetVelocitySource::ConstantVector) {
        std::vector<FormExpr> components;
        components.reserve(static_cast<std::size_t>(dimension));
        for (int d = 0; d < dimension; ++d) {
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

    const auto time_residual = dt(phi);
    const auto spatialResidual = [&](const FormExpr& advecting_velocity) {
        return options.transport_form ==
                       LevelSetTransportForm::ConservativeDivergence
                   ? div(phi * advecting_velocity)
                   : dot(advecting_velocity, grad(phi));
    };
    auto residual = (time_residual * eta).dx();
    FormExpr interface_spatial_residual;
    if (material_interface_velocity) {
        const auto negative_spatial_residual =
            spatialResidual(negative_material_velocity);
        const auto positive_spatial_residual =
            spatialResidual(positive_material_velocity);
        residual = residual +
                   (negative_spatial_residual * eta)
                       .dCutVolume(
                           material_velocity_declaration->interface_marker,
                           CutVolumeSide::Negative) +
                   (positive_spatial_residual * eta)
                       .dCutVolume(
                           material_velocity_declaration->interface_marker,
                           CutVolumeSide::Positive);
        interface_spatial_residual =
            spatialResidual(material_interface_trace_velocity);
    } else {
        interface_spatial_residual = spatialResidual(velocity);
        residual = residual + (interface_spatial_residual * eta).dx();
    }

    const auto appendSupg = [&](const FormExpr& advecting_velocity,
                                const FormExpr& spatial_residual,
                                std::optional<CutVolumeSide> side) {
        if (!options.supg.enabled) {
            return;
        }
        // Directional/transient SUPG scale:
        //   tau = c_tau / sqrt((c_t/dt_eff)^2 + u . G u + eps),
        //   G   = J^{-T} J^{-1}.
        // Unlike h/|u| this remains finite as u -> 0, responds to anisotropic
        // cells in the streamline direction, and contracts with the actual
        // transient integration scale.
        const auto eps = FormExpr::constant(options.supg.velocity_epsilon);
        const auto dt_eff = deltat_eff();
        const auto metric = transpose(Jinv()) * Jinv();
        const auto transient_rate =
            FormExpr::constant(options.supg.transient_scale) / dt_eff;
        const auto directional_rate_squared =
            inner(advecting_velocity, metric * advecting_velocity);
        const auto tau = FormExpr::constant(options.supg.tau_scale) /
                         sqrt(transient_rate * transient_rate +
                              directional_rate_squared + eps);
        const auto streamline_test =
            tau * dot(advecting_velocity, grad(eta));
        const auto integrate = [&](const FormExpr& integrand) {
            return side.has_value()
                       ? integrand.dCutVolume(
                             material_velocity_declaration->interface_marker,
                             *side)
                       : integrand.dx();
        };
        residual = residual + integrate(streamline_test * time_residual) +
                   integrate(streamline_test * spatial_residual);
    };
    if (material_interface_velocity) {
        appendSupg(
            negative_material_velocity,
            spatialResidual(negative_material_velocity),
            CutVolumeSide::Negative);
        appendSupg(
            positive_material_velocity,
            spatialResidual(positive_material_velocity),
            CutVolumeSide::Positive);
    } else {
        appendSupg(velocity, interface_spatial_residual, std::nullopt);
    }
    if (options.interface_kinematic.enabled) {
        residual = residual +
                   (FormExpr::constant(options.interface_kinematic.weight_scale) *
                    h() * time_residual * eta)
                       .dI(options.interface_kinematic.interface_marker) +
                   (FormExpr::constant(options.interface_kinematic.weight_scale) *
                    h() * interface_spatial_residual * eta)
                       .dI(options.interface_kinematic.interface_marker);
    }

    if (phase_id != INVALID_FIELD_ID) {
        const auto& phase_record = system.fieldRecord(phase_id);
        const auto phase = StateField(
            phase_id,
            *phase_record.space,
            options.conservative_phase.liquid_indicator.field_name);
        const auto phase_test = TestField(
            phase_id, *phase_record.space, "phase_test");
        // The conservative stage is an explicit accepted-candidate operation.
        // This algebraic equation keeps q at its previous accepted endpoint
        // during Newton without classifying q as a time-derivative field.
        residual = residual +
                   ((phase - FormExpr::previousSolution(1)) * phase_test).dx();
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
        const auto boundary_measure =
            forms::ExteriorBoundaryMeasure::fullPhysical(marker);
        residual =
            residual +
            (penalty * (phi - target) * eta)
                .dExteriorBoundary(boundary_measure);
    }

    if (!system.hasOperator(options.operator_tag)) {
        system.addOperator(options.operator_tag);
    }

    auto install = install_options;
    install.compiler_options.use_symbolic_tangent = true;
    if (options.velocity.source == LevelSetVelocitySource::CoupledField) {
        install.extra_trial_fields.push_back(velocity_id);
    } else if (material_interface_velocity) {
        install.extra_trial_fields.push_back(
            material_velocity_declaration->negative_velocity_field);
        install.extra_trial_fields.push_back(
            material_velocity_declaration->positive_velocity_field);
    }
    std::vector<FieldId> residual_fields{phi_id};
    if (phase_id != INVALID_FIELD_ID) {
        residual_fields.push_back(phase_id);
    }
    auto kernels = systems::installFormulation(
        system,
        options.operator_tag,
        residual_fields,
        residual,
        install);
    if (algebraic_extension_source_id != INVALID_FIELD_ID) {
        system.addGlobalKernel(
            options.operator_tag,
            std::make_shared<LevelSetVelocityExtensionConstraintKernel>(
                LevelSetVelocityExtensionConstraintConfig{
                    .extension_field = velocity_id,
                    .source_velocity_field =
                        algebraic_extension_source_id,
                    .components =
                        system.fieldRecord(velocity_id).components,
                    .operator_tag = options.operator_tag,
                }));
    }
    return kernels;
}

} // namespace svmp::FE::level_set

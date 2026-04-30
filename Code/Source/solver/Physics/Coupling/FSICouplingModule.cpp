/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Coupling/FSICouplingModule.h"

#include "Core/FEException.h"
#include "FE/Coupling/CouplingDefinitionBuilder.h"
#include "FE/Coupling/CouplingFormBuilder.h"
#include "FE/Coupling/CouplingGraph.h"

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace Physics {
namespace coupling {

namespace {

namespace fec = FE::coupling;
namespace forms = FE::forms;

fec::CouplingFieldUse fieldUse(const std::string& participant,
                               const std::string& field)
{
    return fec::CouplingFieldUse{
        .participant_name = participant,
        .field_name = field,
    };
}

fec::CouplingVariableUse fieldVariable(const std::string& participant,
                                       const std::string& field)
{
    return fec::CouplingVariableUse{
        .kind = fec::CouplingVariableKind::Field,
        .participant_name = participant,
        .name = field,
    };
}

fec::CouplingValueDescriptor scalarValue()
{
    return fec::CouplingValueDescriptor{
        .rank = fec::CouplingValueRank::Scalar,
        .components = 1,
    };
}

void declareFieldRequirement(fec::CouplingDefinitionBuilder& builder,
                             fec::CouplingFieldUse field,
                             fec::CouplingValueDescriptor value)
{
    builder.field(field);
    builder.fieldRequirement(fec::CouplingFieldRequirement{
        .field = std::move(field),
        .value = std::move(value),
    });
}

void appendImplicitDependency(fec::CouplingDefinitionBuilder& builder,
                              fec::CouplingVariableUse residual_row,
                              fec::CouplingVariableUse dependency)
{
    builder.dependency(fec::CouplingResidualDependency{
        .residual_row = residual_row,
        .dependency = dependency,
    });
    builder.expectedBlock(fec::CouplingBlockExpectation{
        .residual_row = std::move(residual_row),
        .dependency = std::move(dependency),
    });
}

void appendMonolithicDependencies(
    const FSICouplingOptions& options,
    fec::CouplingDefinitionBuilder& builder)
{
    if (options.mode != fec::CouplingMode::Monolithic) {
        return;
    }

    const auto fluid_velocity =
        fieldVariable(options.fluid_name, options.fluid_velocity_field);
    const auto solid_displacement =
        fieldVariable(options.solid_name, options.solid_displacement_field);

    appendImplicitDependency(builder,
                             fluid_velocity,
                             fluid_velocity);
    if (options.use_solid_displacement_derivative) {
        appendImplicitDependency(builder,
                                 fluid_velocity,
                                 solid_displacement);
    } else if (options.solid_velocity_field.has_value()) {
        appendImplicitDependency(
            builder,
            fluid_velocity,
            fieldVariable(options.solid_name, *options.solid_velocity_field));
    }

    appendImplicitDependency(
        builder,
        solid_displacement,
        fieldVariable(options.fluid_name, options.fluid_pressure_field));
}

void appendMonolithicGeometryRequirements(
    const FSICouplingOptions& options,
    fec::CouplingDefinitionBuilder& builder)
{
    if (options.mode != fec::CouplingMode::Monolithic) {
        return;
    }

    builder.geometryRequirement(fec::CouplingGeometryTerminalRequirement{
        .quantity = fec::CouplingGeometryTerminalQuantity::Normal,
        .scope = fec::CouplingGeometryTerminalScope{
            .participant_name = options.fluid_name,
            .region = fec::CouplingRegionEndpointDeclaration{
                .participant_name = options.fluid_name,
                .shared_region_name = options.interface_name,
            },
            .location = fec::CouplingGeometryTerminalLocationDeclaration{
                .region_kind = fec::CouplingRegionKind::InterfaceFace,
                .shared_region_name = options.interface_name,
                .coordinate_configuration =
                    forms::GeometryConfiguration::Reference,
            },
        },
    });
}

fec::CouplingValueDescriptor interfaceVectorValue(const FSICouplingOptions& options)
{
    return fec::CouplingValueDescriptor{
        .rank = fec::CouplingValueRank::Vector,
        .components = options.interface_components,
    };
}

void appendPartitionedExchangeDeclarations(
    const FSICouplingOptions& options,
    fec::CouplingDefinitionBuilder& builder)
{
    if (options.mode != fec::CouplingMode::Partitioned) {
        return;
    }

    const auto value = interfaceVectorValue(options);
    builder
        .exchange("solid_displacement",
                  fieldUse(options.solid_name,
                           options.solid_displacement_field),
                  fieldUse(options.fluid_name,
                           options.fluid_velocity_field))
        .producerPort("solid_displacement")
        .consumerPort("fluid_displacement")
        .sharedInterface(options.interface_name)
        .value(value)
        .transfer(options.solid_to_fluid_transfer)
        .producerTemporal(options.partitioned_temporal
                              .solid_displacement_source)
        .consumerTemporal(options.partitioned_temporal
                              .fluid_displacement_target);

    builder
        .exchange("fluid_load",
                  fieldUse(options.fluid_name,
                           options.fluid_velocity_field),
                  fieldUse(options.solid_name,
                           options.solid_displacement_field))
        .producerPort("fluid_load")
        .consumerPort("solid_load")
        .sharedInterface(options.interface_name)
        .value(value)
        .transfer(options.fluid_to_solid_transfer)
        .producerTemporal(options.partitioned_temporal.fluid_load_source)
        .consumerTemporal(options.partitioned_temporal.solid_load_target);

    builder.group(options.contract_name + "_participants",
                  {options.fluid_name, options.solid_name});
}

fec::CouplingValidationResult validateOptionShape(const FSICouplingOptions& options)
{
    fec::CouplingValidationResult result;
    if (options.contract_name.empty()) {
        result.addError("FSI coupling requires a contract instance name");
    }
    if (options.fluid_name.empty() || options.solid_name.empty()) {
        result.addError("FSI coupling requires fluid and solid participant names");
    }
    if (options.mesh_name.has_value() && options.mesh_name->empty()) {
        result.addError("FSI ALE mode requires a mesh participant name");
    }
    if (options.interface_name.empty()) {
        result.addError("FSI coupling requires an interface shared-region name");
    }
    if (options.interface_components <= 0) {
        result.addError("FSI coupling requires a positive interface component count");
    }
    if (options.multiplier.enabled) {
        if (options.multiplier.field_name.empty()) {
            result.addError("FSI multiplier requires a field name");
        }
        if (options.multiplier.space == nullptr) {
            result.addError("FSI multiplier requires a function space");
        }
        if (options.multiplier.components < 0) {
            result.addError("FSI multiplier component count cannot be negative");
        }
    }
    if (options.mode == fec::CouplingMode::Partitioned) {
        if (options.solid_to_fluid_transfer.kind == fec::CouplingTransferKind::Unspecified ||
            options.fluid_to_solid_transfer.kind == fec::CouplingTransferKind::Unspecified) {
            result.addError("FSI partitioned mode requires explicit transfer declarations");
        }
    }
    return result;
}

void validateALEMeshRequirements(fec::CouplingValidationResult& result,
                                 const fec::CouplingContext& ctx,
                                 const FSICouplingOptions& options)
{
    if (!options.mesh_name.has_value()) {
        return;
    }
    if (!options.mesh_displacement_field.has_value() ||
        options.mesh_displacement_field->empty()) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .participant_name = *options.mesh_name,
            .message = "FSI ALE mode requires a mesh displacement field",
        });
        return;
    }
    if (!ctx.hasParticipant(*options.mesh_name)) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .participant_name = *options.mesh_name,
            .message = "FSI ALE mode requires a registered mesh participant",
        });
    }
    if (!ctx.hasField(*options.mesh_name, *options.mesh_displacement_field)) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .participant_name = *options.mesh_name,
            .field_name = *options.mesh_displacement_field,
            .message = "FSI ALE mode requires a registered mesh displacement field",
        });
    }
}

const fec::CouplingRegionRef* findSharedRegionParticipant(
    const fec::SharedRegionRef& group,
    const std::string& participant)
{
    const auto it = std::find_if(
        group.participant_regions.begin(),
        group.participant_regions.end(),
        [&](const fec::CouplingRegionRef& region) {
            return region.participant_name == participant;
        });
    return it == group.participant_regions.end() ? nullptr : &*it;
}

std::optional<fec::CouplingRegionEndpointDeclaration> partitionedRegionEndpoint(
    const fec::CouplingContext& ctx,
    const FSICouplingOptions& options,
    const std::string& participant_name)
{
    if (!ctx.hasSharedRegion(options.interface_name)) {
        return std::nullopt;
    }
    const auto group = ctx.sharedRegionGroup(options.interface_name);
    const auto* region = findSharedRegionParticipant(group, participant_name);
    if (region == nullptr) {
        return std::nullopt;
    }
    return fec::CouplingRegionEndpointDeclaration{
        .participant_name = region->participant_name,
        .region_name = region->region_name,
        .shared_region_name = options.interface_name,
    };
}

void attachPartitionedRegionEndpoints(
    const fec::CouplingContext& ctx,
    const FSICouplingOptions& options,
    std::vector<fec::CouplingExchangeDeclaration>& exchanges)
{
    if (options.mode != fec::CouplingMode::Partitioned) {
        return;
    }
    for (auto& exchange : exchanges) {
        if (exchange.producer.has_value() &&
            exchange.producer->participant_name.has_value()) {
            exchange.producer_region =
                partitionedRegionEndpoint(
                    ctx, options, *exchange.producer->participant_name);
        }
        if (exchange.consumer.has_value() &&
            exchange.consumer->participant_name.has_value()) {
            exchange.consumer_region =
                partitionedRegionEndpoint(
                    ctx, options, *exchange.consumer->participant_name);
        }
    }
}

} // namespace

FSICouplingModule::FSICouplingModule(FSICouplingOptions options)
    : options_(std::move(options))
{
}

std::string FSICouplingModule::name() const
{
    return "fsi";
}

std::string FSICouplingModule::contractInstanceName() const
{
    return options_.contract_name;
}

void FSICouplingModule::define(fec::CouplingDefinitionBuilder& builder) const
{
    builder.participant(options_.fluid_name);
    builder.participant(options_.solid_name);
    if (options_.mesh_name.has_value()) {
        builder.participant(*options_.mesh_name);
    }

    declareFieldRequirement(builder,
                            fieldUse(options_.fluid_name,
                                     options_.fluid_velocity_field),
                            interfaceVectorValue(options_));
    declareFieldRequirement(builder,
                            fieldUse(options_.fluid_name,
                                     options_.fluid_pressure_field),
                            scalarValue());
    declareFieldRequirement(builder,
                            fieldUse(options_.solid_name,
                                     options_.solid_displacement_field),
                            interfaceVectorValue(options_));
    if (!options_.use_solid_displacement_derivative &&
        options_.solid_velocity_field.has_value()) {
        declareFieldRequirement(builder,
                                fieldUse(options_.solid_name,
                                         *options_.solid_velocity_field),
                                interfaceVectorValue(options_));
    }
    if (options_.mesh_name.has_value() &&
        options_.mesh_displacement_field.has_value()) {
        declareFieldRequirement(builder,
                                fieldUse(*options_.mesh_name,
                                         *options_.mesh_displacement_field),
                                interfaceVectorValue(options_));
    }

    builder.sharedRegion(fec::CouplingSharedRegionUse{
        .shared_region_name = options_.interface_name,
        .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
    });
    builder.sharedInterface(fec::CouplingSharedInterfaceRequirement{
        .shared_region_name = options_.interface_name,
        .participant_names = {options_.fluid_name, options_.solid_name},
        .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
        .require_all_participants = true,
        .require_opposite_sides_for_two_participants = true,
        .require_monolithic_topology =
            options_.mode == fec::CouplingMode::Monolithic,
    });

    if (options_.use_solid_displacement_derivative) {
        builder.temporalRequirement(fec::CouplingTemporalRequirement{
            .quantity = fec::CouplingTemporalQuantity::FieldDerivative,
            .field = fieldUse(options_.solid_name,
                              options_.solid_displacement_field),
            .derivative_order = 1,
        });
    }

    if (options_.multiplier.enabled) {
        const auto multiplier_namespace =
            options_.multiplier.contract_field_namespace.empty()
                ? options_.contract_name
                : options_.multiplier.contract_field_namespace;
        builder.additionalField(fec::CouplingAdditionalFieldDeclaration{
            .field_namespace = fec::CouplingAdditionalFieldNamespace::Contract,
            .namespace_name = multiplier_namespace,
            .system_participant_name =
                options_.multiplier.system_participant_name.value_or(""),
            .field_name = options_.multiplier.field_name,
            .space = options_.multiplier.space,
            .components = options_.multiplier.components,
            .scope = fec::CouplingAdditionalFieldScope::InterfaceFace,
            .shared_region_name =
                options_.multiplier.shared_region_name.value_or(options_.interface_name),
        });
    }

    appendPartitionedExchangeDeclarations(options_, builder);
    appendMonolithicDependencies(options_, builder);
    appendMonolithicGeometryRequirements(options_, builder);
}

void FSICouplingModule::validate(const fec::CouplingContext& ctx) const
{
    auto result = validateOptionShape(options_);
    validateALEMeshRequirements(result, ctx, options_);
    auto declaration = declare();
    if (options_.mode == fec::CouplingMode::Partitioned) {
        declaration.partitioned_exchange_declarations =
            buildPartitionedExchangeDeclarations(ctx);
    }
    FE::coupling::CouplingGraph graph;
    const std::array<fec::CouplingContractDeclaration, 1> declarations{declaration};
    result.append(graph.buildDeclarationGraph(
        ctx,
        std::span<const fec::CouplingContractDeclaration>(declarations)));
    throwIfInvalid(result);
}

bool FSICouplingModule::supportsMonolithicLowering() const
{
    return options_.mode == fec::CouplingMode::Monolithic;
}

std::vector<fec::CouplingFormContribution>
FSICouplingModule::buildMonolithicForms(
    const fec::CouplingContext& ctx,
    const fec::CouplingFormBuilder& form_builder) const
{
    static_cast<void>(ctx);
    if (options_.mode != fec::CouplingMode::Monolithic) {
        return {};
    }

    const auto interface = form_builder.sharedInterface(options_.interface_name);
    const auto fluid_side = interface.side(options_.fluid_name);
    const auto solid_side = interface.side(options_.solid_name);

    const auto fluid_velocity =
        fluid_side.state(options_.fluid_velocity_field, "u_f");
    const auto fluid_velocity_test =
        fluid_side.test(options_.fluid_velocity_field, "w_f");

    fec::CouplingFieldUse solid_dependency;
    forms::FormExpr solid_velocity;
    if (options_.use_solid_displacement_derivative) {
        solid_dependency = fieldUse(options_.solid_name,
                                    options_.solid_displacement_field);
        solid_velocity = solid_side.dt(options_.solid_displacement_field,
                                       "dt_d_s");
    } else {
        FE_THROW_IF(!options_.solid_velocity_field.has_value() ||
                        options_.solid_velocity_field->empty(),
                    FE::InvalidArgumentException,
                    "FSI monolithic form requires a solid velocity field or displacement derivative");
        const auto& solid_velocity_field = *options_.solid_velocity_field;
        solid_dependency = fieldUse(options_.solid_name, solid_velocity_field);
        solid_velocity = solid_side.state(solid_velocity_field, "v_s");
    }

    std::vector<fec::CouplingFormContribution> contributions;
    std::optional<fec::CouplingGeometrySensitivityDeclaration>
        mesh_geometry_sensitivity;
    if (options_.mesh_name.has_value() &&
        options_.mesh_displacement_field.has_value()) {
        mesh_geometry_sensitivity = fec::CouplingGeometrySensitivityDeclaration{
            .mode = forms::GeometrySensitivityMode::MeshMotionUnknowns,
            .mesh_motion_field = fieldUse(*options_.mesh_name,
                                          *options_.mesh_displacement_field),
        };
    }

    fec::CouplingFormContribution kinematic;
    kinematic.contribution_name =
        options_.contract_name + "_velocity_continuity";
    kinematic.origin = "FSICouplingModule";
    kinematic.operator_name = "equations";
    kinematic.field_uses = {fieldUse(options_.fluid_name,
                                     options_.fluid_velocity_field)};
    kinematic.extra_trial_field_uses = {std::move(solid_dependency)};
    kinematic.residual =
        interface.integral(forms::inner(fluid_velocity - solid_velocity,
                                        fluid_velocity_test),
                           options_.fluid_name);
    if (mesh_geometry_sensitivity.has_value()) {
        kinematic.install_options_declaration.geometry_sensitivity =
            *mesh_geometry_sensitivity;
        kinematic.extra_trial_field_uses.push_back(
            fieldUse(*options_.mesh_name, *options_.mesh_displacement_field));
    }
    contributions.push_back(
        form_builder.attachTerminalProvenance(std::move(kinematic)));

    const auto fluid_pressure =
        fluid_side.state(options_.fluid_pressure_field, "p_f");
    const auto solid_displacement_test =
        solid_side.test(options_.solid_displacement_field, "w_s");
    const auto interface_normal = fluid_side.normal();

    fec::CouplingFormContribution traction;
    traction.contribution_name =
        options_.contract_name + "_pressure_traction_balance";
    traction.origin = "FSICouplingModule";
    traction.operator_name = "equations";
    traction.field_uses = {fieldUse(options_.solid_name,
                                    options_.solid_displacement_field)};
    traction.extra_trial_field_uses = {fieldUse(options_.fluid_name,
                                                options_.fluid_pressure_field)};
    traction.residual =
        interface.integral(
            -forms::inner(fluid_pressure * interface_normal,
                          solid_displacement_test),
            options_.fluid_name);
    if (mesh_geometry_sensitivity.has_value()) {
        traction.install_options_declaration.geometry_sensitivity =
            *mesh_geometry_sensitivity;
        traction.extra_trial_field_uses.push_back(
            fieldUse(*options_.mesh_name, *options_.mesh_displacement_field));
    }
    contributions.push_back(
        form_builder.attachTerminalProvenance(std::move(traction)));

    return contributions;
}

std::vector<fec::CouplingExchangeDeclaration>
FSICouplingModule::buildPartitionedExchangeDeclarations(
    const fec::CouplingContext& ctx) const
{
    if (options_.mode != fec::CouplingMode::Partitioned) {
        return {};
    }
    auto declaration = declare();
    auto exchanges = std::move(declaration.partitioned_exchange_declarations);
    attachPartitionedRegionEndpoints(ctx, options_, exchanges);
    return exchanges;
}

} // namespace coupling
} // namespace Physics
} // namespace svmp

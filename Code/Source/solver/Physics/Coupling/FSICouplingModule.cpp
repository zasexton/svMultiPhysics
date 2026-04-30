/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Coupling/FSICouplingModule.h"

#include "FE/Coupling/CouplingGraph.h"

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace Physics {
namespace coupling {

namespace {

namespace fec = FE::coupling;

fec::CouplingFieldUse fieldUse(const std::string& participant,
                               const std::string& field)
{
    return fec::CouplingFieldUse{
        .participant_name = participant,
        .field_name = field,
    };
}

fec::CouplingEndpointRef fieldEndpoint(const std::string& participant,
                                       const std::string& field)
{
    return fec::CouplingEndpointRef{
        .kind = fec::CouplingEndpointKind::Field,
        .participant_name = participant,
        .endpoint_name = field,
        .temporal = fec::CouplingTemporalSlotDescriptor{
            .slot = fec::CouplingTemporalSlot::Current,
        },
    };
}

fec::CouplingPortId port(const FSICouplingOptions& options,
                         std::string port_name)
{
    return fec::CouplingPortId{
        .contract_instance_name = options.contract_name,
        .port_name = std::move(port_name),
    };
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
    fec::CouplingContractDeclaration& declaration)
{
    if (options.mode != fec::CouplingMode::Partitioned) {
        return;
    }

    declaration.partitioned_exchange_declarations.push_back(
        fec::CouplingExchangeDeclaration{
            .producer_port = port(options, "solid_displacement"),
            .consumer_port = port(options, "fluid_displacement"),
            .value = interfaceVectorValue(options),
            .producer = fieldEndpoint(options.solid_name,
                                      options.solid_displacement_field),
            .consumer = fieldEndpoint(options.fluid_name,
                                      options.fluid_velocity_field),
            .shared_region_name = options.interface_name,
            .transfer = options.solid_to_fluid_transfer,
        });

    declaration.partitioned_exchange_declarations.push_back(
        fec::CouplingExchangeDeclaration{
            .producer_port = port(options, "fluid_load"),
            .consumer_port = port(options, "solid_load"),
            .value = interfaceVectorValue(options),
            .producer = fieldEndpoint(options.fluid_name,
                                      options.fluid_velocity_field),
            .consumer = fieldEndpoint(options.solid_name,
                                      options.solid_displacement_field),
            .shared_region_name = options.interface_name,
            .transfer = options.fluid_to_solid_transfer,
        });

    declaration.group_hints.push_back(fec::CouplingGroupHint{
        .name = options.contract_name + "_participants",
        .participant_names = {options.fluid_name, options.solid_name},
    });
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

void validateVectorFieldComponents(fec::CouplingValidationResult& result,
                                   const fec::CouplingContext& ctx,
                                   const FSICouplingOptions& options,
                                   const std::string& participant,
                                   const std::string& field,
                                   std::string_view label)
{
    if (!ctx.hasField(participant, field)) {
        return;
    }
    const auto ref = ctx.field(participant, field);
    if (ref.components == options.interface_components) {
        return;
    }
    result.add(fec::CouplingDiagnostic{
        .severity = fec::CouplingDiagnosticSeverity::Error,
        .contract_name = options.contract_name,
        .participant_name = participant,
        .field_name = field,
        .message = "FSI " + std::string(label) +
                   " field component count must match the interface component count",
    });
}

void validateScalarPressureComponents(fec::CouplingValidationResult& result,
                                      const fec::CouplingContext& ctx,
                                      const FSICouplingOptions& options)
{
    if (!ctx.hasField(options.fluid_name, options.fluid_pressure_field)) {
        return;
    }
    const auto ref = ctx.field(options.fluid_name, options.fluid_pressure_field);
    if (ref.components == 1) {
        return;
    }
    result.add(fec::CouplingDiagnostic{
        .severity = fec::CouplingDiagnosticSeverity::Error,
        .contract_name = options.contract_name,
        .participant_name = options.fluid_name,
        .field_name = options.fluid_pressure_field,
        .message = "FSI fluid pressure field component count must be scalar",
    });
}

void validateFieldComponentCounts(fec::CouplingValidationResult& result,
                                  const fec::CouplingContext& ctx,
                                  const FSICouplingOptions& options)
{
    if (options.interface_components <= 0) {
        return;
    }

    validateVectorFieldComponents(result,
                                  ctx,
                                  options,
                                  options.fluid_name,
                                  options.fluid_velocity_field,
                                  "fluid velocity");
    validateScalarPressureComponents(result, ctx, options);
    validateVectorFieldComponents(result,
                                  ctx,
                                  options,
                                  options.solid_name,
                                  options.solid_displacement_field,
                                  "solid displacement");
    if (!options.use_solid_displacement_derivative &&
        options.solid_velocity_field.has_value()) {
        validateVectorFieldComponents(result,
                                      ctx,
                                      options,
                                      options.solid_name,
                                      *options.solid_velocity_field,
                                      "solid velocity");
    }
    if (options.mesh_name.has_value() &&
        options.mesh_displacement_field.has_value()) {
        validateVectorFieldComponents(result,
                                      ctx,
                                      options,
                                      *options.mesh_name,
                                      *options.mesh_displacement_field,
                                      "mesh displacement");
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

void validateInterfaceRegionMappings(fec::CouplingValidationResult& result,
                                      const fec::CouplingContext& ctx,
                                      const FSICouplingOptions& options)
{
    if (!ctx.hasSharedRegion(options.interface_name)) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .region_name = options.interface_name,
            .message = "FSI interface shared region is missing",
        });
        return;
    }

    const auto group = ctx.sharedRegionGroup(options.interface_name);
    const auto* fluid_region =
        findSharedRegionParticipant(group, options.fluid_name);
    if (fluid_region == nullptr) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .participant_name = options.fluid_name,
            .region_name = options.interface_name,
            .message = "FSI interface shared region must map the fluid participant",
        });
    }

    const auto* solid_region =
        findSharedRegionParticipant(group, options.solid_name);
    if (solid_region == nullptr) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .participant_name = options.solid_name,
            .region_name = options.interface_name,
            .message = "FSI interface shared region must map the solid participant",
        });
    }

    if (fluid_region != nullptr && solid_region != nullptr &&
        fluid_region->side != fec::CouplingInterfaceSide::None &&
        solid_region->side != fec::CouplingInterfaceSide::None &&
        fluid_region->side == solid_region->side) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .region_name = options.interface_name,
            .message = "FSI interface shared-region fluid and solid mappings must occupy opposite sides",
        });
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

fec::CouplingContractDeclaration FSICouplingModule::declare() const
{
    fec::CouplingContractDeclaration declaration;
    declaration.contract_type = name();
    declaration.contract_name = options_.contract_name;
    declaration.participants.push_back({.participant_name = options_.fluid_name});
    declaration.participants.push_back({.participant_name = options_.solid_name});
    if (options_.mesh_name.has_value()) {
        declaration.participants.push_back({.participant_name = *options_.mesh_name});
    }

    declaration.fields.push_back(fieldUse(options_.fluid_name,
                                          options_.fluid_velocity_field));
    declaration.fields.push_back(fieldUse(options_.fluid_name,
                                          options_.fluid_pressure_field));
    declaration.fields.push_back(fieldUse(options_.solid_name,
                                          options_.solid_displacement_field));
    if (!options_.use_solid_displacement_derivative &&
        options_.solid_velocity_field.has_value()) {
        declaration.fields.push_back(fieldUse(options_.solid_name,
                                              *options_.solid_velocity_field));
    }
    if (options_.mesh_name.has_value() &&
        options_.mesh_displacement_field.has_value()) {
        declaration.fields.push_back(fieldUse(*options_.mesh_name,
                                              *options_.mesh_displacement_field));
    }

    declaration.shared_regions.push_back(fec::CouplingSharedRegionUse{
        .shared_region_name = options_.interface_name,
        .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
    });

    if (options_.use_solid_displacement_derivative) {
        declaration.temporal_requirements.push_back(fec::CouplingTemporalRequirement{
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
        declaration.additional_fields.push_back(
            fec::CouplingAdditionalFieldDeclaration{
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

    appendPartitionedExchangeDeclarations(options_, declaration);
    return declaration;
}

void FSICouplingModule::validate(const fec::CouplingContext& ctx) const
{
    auto result = validateOptionShape(options_);
    validateALEMeshRequirements(result, ctx, options_);
    validateFieldComponentCounts(result, ctx, options_);
    validateInterfaceRegionMappings(result, ctx, options_);
    const auto declaration = declare();
    FE::coupling::CouplingGraph graph;
    const std::array<fec::CouplingContractDeclaration, 1> declarations{declaration};
    result.append(graph.buildDeclarationGraph(
        ctx,
        std::span<const fec::CouplingContractDeclaration>(declarations)));
    throwIfInvalid(result);
}

std::vector<fec::CouplingExchangeDeclaration>
FSICouplingModule::buildPartitionedExchangeDeclarations(const fec::CouplingContext&) const
{
    return {};
}

} // namespace coupling
} // namespace Physics
} // namespace svmp

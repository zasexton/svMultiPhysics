/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingDefinitionBuilder.h"

#include "Core/FEException.h"
#include "Spaces/FunctionSpace.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

bool required(CouplingRequirement requirement) noexcept
{
    return requirement == CouplingRequirement::Required;
}

void mergeRequirement(CouplingRequirement& current,
                      CouplingRequirement incoming) noexcept
{
    if (required(incoming)) {
        current = CouplingRequirement::Required;
    }
}

void appendParticipantIfMissing(CouplingContractDeclaration& declaration,
                                std::string participant_name,
                                CouplingRequirement requirement)
{
    const auto it = std::find_if(
        declaration.participants.begin(),
        declaration.participants.end(),
        [&](const CouplingParticipantUse& participant) {
            return participant.participant_name == participant_name;
        });
    if (it == declaration.participants.end()) {
        declaration.participants.push_back(CouplingParticipantUse{
            .participant_name = std::move(participant_name),
            .requirement = requirement,
        });
        return;
    }
    mergeRequirement(it->requirement, requirement);
}

void appendFieldIfMissing(CouplingContractDeclaration& declaration,
                          CouplingFieldUse field)
{
    const auto it = std::find_if(
        declaration.fields.begin(),
        declaration.fields.end(),
        [&](const CouplingFieldUse& existing) {
            return existing.participant_name == field.participant_name &&
                   existing.field_name == field.field_name;
        });
    if (it == declaration.fields.end()) {
        declaration.fields.push_back(std::move(field));
        return;
    }
    mergeRequirement(it->requirement, field.requirement);
}

void appendSharedRegionIfMissing(CouplingContractDeclaration& declaration,
                                 CouplingSharedRegionUse region)
{
    const auto it = std::find_if(
        declaration.shared_regions.begin(),
        declaration.shared_regions.end(),
        [&](const CouplingSharedRegionUse& existing) {
            return existing.shared_region_name == region.shared_region_name;
        });
    if (it == declaration.shared_regions.end()) {
        declaration.shared_regions.push_back(std::move(region));
        return;
    }
    mergeRequirement(it->requirement, region.requirement);
    if (!it->required_region_kind.has_value()) {
        it->required_region_kind = region.required_region_kind;
    }
}

std::vector<CouplingRelationLoweringCapability> sidePairedLoweringCapabilities(
    const CouplingSidePairedInterfaceRequest& request)
{
    std::vector<CouplingRelationLoweringCapability> capabilities;
    if (request.supports_monolithic_forms) {
        capabilities.push_back(CouplingRelationLoweringCapability{
            .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            .fidelity = request.monolithic_fidelity,
            .enforcement_strategies = {request.enforcement_strategy},
        });
    }
    if (request.supports_partitioned_exchange) {
        capabilities.push_back(CouplingRelationLoweringCapability{
            .lowering_kind = CouplingRelationLoweringKind::PartitionedExchange,
            .fidelity = request.partitioned_fidelity,
            .enforcement_strategies = {request.enforcement_strategy},
            .partitioned_solve_strategies =
                request.partitioned_solve_strategies,
        });
    }
    return capabilities;
}

bool hasText(const std::optional<std::string>& value) noexcept
{
    return value.has_value() && !value->empty();
}

bool startsWith(std::string_view value, std::string_view prefix) noexcept
{
    return value.size() >= prefix.size() &&
           value.substr(0, prefix.size()) == prefix;
}

std::string defaultConsumerPortName(
    const CouplingPartitionedFieldChannelRequest& channel)
{
    const std::string participant_prefix =
        channel.producer_participant_name + "_";
    if (startsWith(channel.channel_name, participant_prefix)) {
        return channel.consumer_participant_name + "_" +
               channel.channel_name.substr(participant_prefix.size());
    }
    return channel.channel_name + ".consumer";
}

void applySidePairedChannelDefaults(
    CouplingPartitionedFieldChannelRequest& channel,
    const CouplingSidePairedInterfaceRequest& interface,
    bool infer_shared_region,
    bool infer_ports)
{
    if (infer_shared_region && channel.shared_region_name.empty()) {
        channel.shared_region_name = interface.shared_region_name;
    }
    if (infer_ports && channel.producer_port_name.empty()) {
        channel.producer_port_name = channel.channel_name;
    }
    if (infer_ports && channel.consumer_port_name.empty()) {
        channel.consumer_port_name = defaultConsumerPortName(channel);
    }
}

} // namespace

CouplingSidePairedInterfaceRequest sidePairedInterface(
    std::string relation_name,
    std::string shared_region_name,
    std::string first_participant_name,
    std::string second_participant_name,
    CouplingMode mode,
    std::string enforcement_strategy,
    std::vector<CouplingPartitionedSolveStrategy>
        partitioned_solve_strategies)
{
    return CouplingSidePairedInterfaceRequest{
        .relation_name = std::move(relation_name),
        .shared_region_name = std::move(shared_region_name),
        .first_participant_name = std::move(first_participant_name),
        .second_participant_name = std::move(second_participant_name),
        .mode = mode,
        .enforcement_strategy = std::move(enforcement_strategy),
        .partitioned_solve_strategies =
            std::move(partitioned_solve_strategies),
    };
}

CouplingFieldRoleRequest fieldRole(CouplingFieldUse field,
                                   CouplingValueDescriptor value,
                                   bool enabled)
{
    return CouplingFieldRoleRequest{
        .field = std::move(field),
        .value = std::move(value),
        .enabled = enabled,
    };
}

CouplingFieldRoleRequest scalarFieldRole(std::string participant_name,
                                         std::string field_name,
                                         bool enabled)
{
    return fieldRole(fieldUse(std::move(participant_name),
                              std::move(field_name)),
                     scalarValue(),
                     enabled);
}

CouplingFieldRoleRequest vectorFieldRole(std::string participant_name,
                                         std::string field_name,
                                         int components,
                                         bool enabled)
{
    return fieldRole(fieldUse(std::move(participant_name),
                              std::move(field_name)),
                     vectorValue(components),
                     enabled);
}

CouplingDiagnostic optionErrorDiagnostic(std::string contract_name,
                                         std::string message,
                                         std::string participant_name,
                                         std::string field_name)
{
    return CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = std::move(contract_name),
        .participant_name = std::move(participant_name),
        .field_name = std::move(field_name),
        .message = std::move(message),
    };
}

CouplingOptionalFieldRoleRequest optionalFieldRole(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    CouplingValueDescriptor value,
    CouplingDiagnostic missing_field_diagnostic,
    bool enabled)
{
    return CouplingOptionalFieldRoleRequest{
        .participant_name = std::move(participant_name),
        .field_name = std::move(field_name),
        .value = std::move(value),
        .missing_field_diagnostic = std::move(missing_field_diagnostic),
        .enabled = enabled,
    };
}

CouplingOptionalFieldRoleRequest optionalVectorFieldRole(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    int components,
    CouplingDiagnostic missing_field_diagnostic,
    bool enabled)
{
    return optionalFieldRole(std::move(participant_name),
                             std::move(field_name),
                             vectorValue(components),
                             std::move(missing_field_diagnostic),
                             enabled);
}

CouplingOptionalFieldRoleRequest optionalFieldRole(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    CouplingValueDescriptor value,
    bool enabled)
{
    return optionalFieldRole(std::move(participant_name),
                             std::move(field_name),
                             std::move(value),
                             CouplingDiagnostic{},
                             enabled);
}

CouplingOptionalFieldRoleRequest optionalFieldRole(
    std::string participant_name,
    std::optional<std::string> field_name,
    CouplingValueDescriptor value,
    bool enabled)
{
    return optionalFieldRole(std::optional<std::string>{std::move(participant_name)},
                             std::move(field_name),
                             std::move(value),
                             enabled);
}

CouplingOptionalFieldRoleRequest optionalVectorFieldRole(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    int components,
    bool enabled)
{
    return optionalFieldRole(std::move(participant_name),
                             std::move(field_name),
                             vectorValue(components),
                             enabled);
}

CouplingOptionalFieldRoleRequest optionalVectorFieldRole(
    std::string participant_name,
    std::optional<std::string> field_name,
    int components,
    bool enabled)
{
    return optionalVectorFieldRole(
        std::optional<std::string>{std::move(participant_name)},
        std::move(field_name),
        components,
        enabled);
}

CouplingContractInterfaceFieldRequest contractInterfaceField(
    std::string field_name,
    std::shared_ptr<const spaces::FunctionSpace> space,
    std::string shared_region_name,
    int components,
    std::string field_namespace,
    std::optional<std::string> system_participant_name,
    CouplingRequirement requirement,
    bool enabled,
    CouplingLocalEliminationPolicy local_elimination_policy)
{
    return CouplingContractInterfaceFieldRequest{
        .field_name = std::move(field_name),
        .space = std::move(space),
        .components = components,
        .field_namespace = std::move(field_namespace),
        .system_participant_name = std::move(system_participant_name),
        .shared_region_name = std::move(shared_region_name),
        .requirement = requirement,
        .enabled = enabled,
        .local_elimination_policy = local_elimination_policy,
    };
}

CouplingPartitionedFieldChannelRequest partitionedFieldChannel(
    std::string channel_name,
    std::string producer_participant_name,
    std::string producer_field_name,
    std::string consumer_participant_name,
    std::string consumer_field_name,
    std::string shared_region_name,
    CouplingTransferDeclaration transfer,
    CouplingTemporalSlotDescriptor producer_temporal,
    CouplingTemporalSlotDescriptor consumer_temporal,
    std::string producer_port_name,
    std::string consumer_port_name,
    CouplingPartitionedStrategyDeclaration strategy)
{
    return CouplingPartitionedFieldChannelRequest{
        .channel_name = std::move(channel_name),
        .producer_participant_name = std::move(producer_participant_name),
        .producer_field_name = std::move(producer_field_name),
        .consumer_participant_name = std::move(consumer_participant_name),
        .consumer_field_name = std::move(consumer_field_name),
        .producer_port_name = std::move(producer_port_name),
        .consumer_port_name = std::move(consumer_port_name),
        .shared_region_name = std::move(shared_region_name),
        .transfer = std::move(transfer),
        .strategy = strategy,
        .producer_temporal = producer_temporal,
        .consumer_temporal = consumer_temporal,
    };
}

CouplingPartitionedFieldChannelRequest fieldExchange(
    std::string channel_name,
    std::string producer_participant_name,
    std::string producer_field_name,
    std::string consumer_participant_name,
    std::string consumer_field_name,
    CouplingTransferDeclaration transfer,
    CouplingTemporalSlotDescriptor producer_temporal,
    CouplingTemporalSlotDescriptor consumer_temporal,
    CouplingPartitionedStrategyDeclaration strategy)
{
    return partitionedFieldChannel(std::move(channel_name),
                                   std::move(producer_participant_name),
                                   std::move(producer_field_name),
                                   std::move(consumer_participant_name),
                                   std::move(consumer_field_name),
                                   {},
                                   std::move(transfer),
                                   producer_temporal,
                                   consumer_temporal,
                                   {},
                                   {},
                                   strategy);
}

CouplingPartitionedGroupRequest partitionedGroup(
    std::string group_name,
    std::vector<std::string> participant_names)
{
    return CouplingPartitionedGroupRequest{
        .group_name = std::move(group_name),
        .participant_names = std::move(participant_names),
    };
}

CouplingDefinitionBuilder::CouplingDefinitionBuilder(
    std::string contract_type,
    std::string contract_name)
    : partitioned_(contract_name)
{
    FE_THROW_IF(contract_type.empty(), InvalidArgumentException,
                "coupling definition requires a contract type");
    FE_THROW_IF(contract_name.empty(), InvalidArgumentException,
                "coupling definition requires a contract name");

    declaration_.contract_type = std::move(contract_type);
    declaration_.contract_name = std::move(contract_name);
    declaration_.dependency_declaration_mode =
        CouplingDependencyDeclarationMode::InferFromInstalledForms;
}

const std::string& CouplingDefinitionBuilder::contractType() const noexcept
{
    return declaration_.contract_type;
}

const std::string& CouplingDefinitionBuilder::contractName() const noexcept
{
    return declaration_.contract_name;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::participant(
    std::string participant_name,
    CouplingRequirement requirement)
{
    declaration_.participants.push_back(CouplingParticipantUse{
        .participant_name = std::move(participant_name),
        .requirement = requirement,
    });
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::field(
    std::string participant_name,
    std::string field_name,
    CouplingRequirement requirement)
{
    declaration_.fields.push_back(CouplingFieldUse{
        .participant_name = std::move(participant_name),
        .field_name = std::move(field_name),
        .requirement = requirement,
    });
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::field(
    CouplingFieldUse field)
{
    declaration_.fields.push_back(std::move(field));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::fieldRequirement(
    CouplingFieldRequirement requirement)
{
    partitioned_.addFieldRequirement(requirement);
    declaration_.field_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::requiredField(
    CouplingFieldUse field,
    CouplingValueDescriptor value)
{
    field.requirement = CouplingRequirement::Required;
    appendParticipantIfMissing(declaration_,
                               field.participant_name,
                               CouplingRequirement::Required);
    appendFieldIfMissing(declaration_, field);
    fieldRequirement(CouplingFieldRequirement{
        .field = std::move(field),
        .value = std::move(value),
    });
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::requiredScalarField(
    std::string participant_name,
    std::string field_name)
{
    return requiredField(fieldUse(std::move(participant_name),
                                  std::move(field_name)),
                         scalarValue());
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::requiredVectorField(
    std::string participant_name,
    std::string field_name,
    int components)
{
    return requiredField(fieldUse(std::move(participant_name),
                                  std::move(field_name)),
                         vectorValue(components));
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::additionalField(
    CouplingAdditionalFieldDeclaration declaration)
{
    declaration_.additional_fields.push_back(std::move(declaration));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::contractInterfaceField(
    CouplingContractInterfaceFieldRequest request)
{
    const auto components =
        request.components > 0 || request.space == nullptr
            ? request.components
            : request.space->value_dimension();
    additionalField(CouplingAdditionalFieldDeclaration{
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = request.field_namespace.empty()
                              ? declaration_.contract_name
                              : std::move(request.field_namespace),
        .system_participant_name =
            request.system_participant_name.value_or(std::string{}),
        .field_name = std::move(request.field_name),
        .space = std::move(request.space),
        .components = components,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
        .shared_region_name = std::move(request.shared_region_name),
        .requirement = request.requirement,
        .enabled = request.enabled,
        .local_elimination_policy = request.local_elimination_policy,
    });
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::nonFieldDependency(
    CouplingNonFieldDependencyRequirement requirement)
{
    declaration_.non_field_dependencies.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::temporalRequirement(
    CouplingTemporalRequirement requirement)
{
    declaration_.temporal_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::geometryRequirement(
    CouplingGeometryTerminalRequirement requirement)
{
    declaration_.geometry_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::dependency(
    CouplingResidualDependency dependency)
{
    declaration_.dependencies.push_back(std::move(dependency));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::expectedBlock(
    CouplingBlockExpectation expectation)
{
    declaration_.expected_blocks.push_back(std::move(expectation));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::dependencyDeclarationMode(
    CouplingDependencyDeclarationMode mode)
{
    declaration_.dependency_declaration_mode = mode;
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::region(
    CouplingRegionUse region)
{
    declaration_.regions.push_back(std::move(region));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::sharedRegion(
    CouplingSharedRegionUse region)
{
    declaration_.shared_regions.push_back(std::move(region));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::sharedInterface(
    CouplingSharedInterfaceRequirement requirement)
{
    declaration_.shared_interface_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::regionRelation(
    CouplingRegionRelationRequirement requirement)
{
    declaration_.region_relation_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::sidePairedInterface(
    CouplingSidePairedInterfaceRequest request)
{
    FE_THROW_IF(request.enforcement_strategy.empty(),
                InvalidArgumentException,
                "side-paired interface requires an enforcement strategy");

    appendParticipantIfMissing(declaration_,
                               request.first_participant_name,
                               CouplingRequirement::Required);
    appendParticipantIfMissing(declaration_,
                               request.second_participant_name,
                               CouplingRequirement::Required);

    if (request.declare_shared_region) {
        appendSharedRegionIfMissing(declaration_,
                                    CouplingSharedRegionUse{
                                        .shared_region_name =
                                            request.shared_region_name,
                                        .required_region_kind =
                                            request.required_region_kind,
                                    });
    }

    if (request.declare_shared_interface) {
        sharedInterface(CouplingSharedInterfaceRequirement{
            .shared_region_name = request.shared_region_name,
            .participant_names = {
                request.first_participant_name,
                request.second_participant_name,
            },
            .required_region_kind = request.required_region_kind,
            .require_all_participants = request.require_all_participants,
            .require_opposite_sides_for_two_participants =
                request.require_opposite_sides,
            .require_monolithic_topology =
                request.require_common_monolithic_system_when_monolithic &&
                request.mode == CouplingMode::Monolithic,
        });
    }

    const auto selected_partitioned_strategy =
        request.mode == CouplingMode::Partitioned &&
                !request.partitioned_solve_strategies.empty()
            ? std::optional<CouplingPartitionedSolveStrategy>{
                  request.partitioned_solve_strategies.front()}
            : std::nullopt;
    regionRelation(CouplingRegionRelationRequirement{
        .relation_name = std::move(request.relation_name),
        .relation_kind = CouplingRegionRelationKind::SidePairedInterface,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = request.first_participant_name,
                .shared_region_name = request.shared_region_name,
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = request.second_participant_name,
                .shared_region_name = request.shared_region_name,
            },
        },
        .lowering_capabilities = sidePairedLoweringCapabilities(request),
        .selected_lowering = selectedLoweringForMode(
            request.mode,
            request.enforcement_strategy,
            selected_partitioned_strategy),
        .required_region_kind = request.required_region_kind,
        .require_all_endpoints = request.require_all_participants,
        .require_distinct_participants = true,
        .require_opposite_sides_for_side_pair =
            request.require_opposite_sides,
        .require_common_monolithic_system =
            request.require_common_monolithic_system_when_monolithic &&
            request.mode == CouplingMode::Monolithic,
        .require_registered_topology =
            request.require_registered_topology_when_monolithic &&
            request.mode == CouplingMode::Monolithic,
    });
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::sidePairedPDECoupling(
    CouplingSidePairedPDERequest request)
{
    for (auto& role : request.required_fields) {
        if (role.enabled) {
            requiredField(std::move(role.field), std::move(role.value));
        }
    }

    for (auto& role : request.optional_fields) {
        if (!role.enabled) {
            continue;
        }
        if (hasText(role.participant_name) && hasText(role.field_name)) {
            requiredField(fieldUse(*role.participant_name, *role.field_name),
                          std::move(role.value));
            continue;
        }

        auto diagnostic = std::move(role.missing_field_diagnostic);
        if (diagnostic.message.empty()) {
            diagnostic.message =
                "enabled coupling field role requires participant and field names";
        }
        if (diagnostic.contract_name.empty()) {
            diagnostic.contract_name = declaration_.contract_name;
        }
        if (diagnostic.participant_name.empty() &&
            role.participant_name.has_value()) {
            diagnostic.participant_name = *role.participant_name;
        }
        if (diagnostic.field_name.empty() && role.field_name.has_value()) {
            diagnostic.field_name = *role.field_name;
        }
        option_validation_.add(std::move(diagnostic));
    }

    const auto mode = request.interface.mode;
    const auto interface_defaults = request.interface;
    sidePairedInterface(std::move(request.interface));

    if (request.interface_field.has_value() && request.interface_field->enabled) {
        contractInterfaceField(std::move(*request.interface_field));
    }

    if (mode == CouplingMode::Partitioned) {
        for (auto& channel : request.partitioned_channels) {
            applySidePairedChannelDefaults(
                channel,
                interface_defaults,
                request.infer_partitioned_channel_shared_regions,
                request.infer_partitioned_channel_ports);
            partitionedFieldChannel(std::move(channel));
        }
        if (request.partitioned_group.has_value()) {
            group(std::move(request.partitioned_group->group_name),
                  std::move(request.partitioned_group->participant_names));
        } else if (request.infer_partitioned_group) {
            group(declaration_.contract_name + "_participants",
                  {
                      interface_defaults.first_participant_name,
                      interface_defaults.second_participant_name,
                  });
        }
    }

    if (mode == CouplingMode::Monolithic && request.monolithic_forms) {
        monolithic(std::move(request.monolithic_forms));
    }

    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::group(
    std::string name,
    std::vector<std::string> participant_names)
{
    partitioned_.group(std::move(name), std::move(participant_names));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::monolithic(
    MonolithicFormsCallback callback)
{
    FE_THROW_IF(!callback, InvalidArgumentException,
                "coupling definition monolithic callback is empty");
    monolithic_callbacks_.push_back(std::move(callback));
    return *this;
}

PartitionedExchangeBuilder CouplingDefinitionBuilder::exchange(
    std::string_view name,
    const CouplingFieldUse& producer_field,
    const CouplingFieldUse& consumer_field)
{
    return partitioned_.exchange(name, producer_field, consumer_field);
}

PartitionedExchangeBuilder CouplingDefinitionBuilder::exchange(
    std::string_view name,
    CouplingEndpointRef producer_endpoint,
    CouplingEndpointRef consumer_endpoint)
{
    return partitioned_.exchange(name,
                                 std::move(producer_endpoint),
                                 std::move(consumer_endpoint));
}

PartitionedExchangeBuilder CouplingDefinitionBuilder::partitionedFieldChannel(
    CouplingPartitionedFieldChannelRequest request)
{
    auto handle =
        exchange(request.channel_name,
                 fieldUse(std::move(request.producer_participant_name),
                          std::move(request.producer_field_name)),
                 fieldUse(std::move(request.consumer_participant_name),
                          std::move(request.consumer_field_name)));
    if (!request.producer_port_name.empty()) {
        handle.producerPort(request.producer_port_name);
    }
    if (!request.consumer_port_name.empty()) {
        handle.consumerPort(request.consumer_port_name);
    }
    if (!request.shared_region_name.empty()) {
        handle.sharedInterface(request.shared_region_name);
    }
    handle.transfer(std::move(request.transfer))
        .strategy(request.strategy)
        .producerTemporal(request.producer_temporal)
        .consumerTemporal(request.consumer_temporal);
    return handle;
}

bool CouplingDefinitionBuilder::hasMonolithicForms() const noexcept
{
    return !monolithic_callbacks_.empty();
}

bool CouplingDefinitionBuilder::hasPartitionedExchanges() const noexcept
{
    return !partitioned_.declarations().empty();
}

CouplingValidationResult CouplingDefinitionBuilder::optionValidation() const
{
    return option_validation_;
}

CouplingContractDeclaration CouplingDefinitionBuilder::compileDeclaration() const
{
    auto declaration = declaration_;
    const auto& exchanges = partitioned_.declarations();
    declaration.partitioned_exchange_declarations.insert(
        declaration.partitioned_exchange_declarations.end(),
        exchanges.begin(),
        exchanges.end());
    const auto& groups = partitioned_.groupHints();
    declaration.group_hints.insert(declaration.group_hints.end(),
                                   groups.begin(),
                                   groups.end());
    return declaration;
}

std::vector<CouplingFormContribution>
CouplingDefinitionBuilder::buildMonolithicForms(
    const CouplingContext& context,
    const CouplingFormBuilder& forms) const
{
    std::vector<CouplingFormContribution> contributions;
    for (const auto& callback : monolithic_callbacks_) {
        auto next = callback(context, forms);
        contributions.insert(contributions.end(),
                             std::make_move_iterator(next.begin()),
                             std::make_move_iterator(next.end()));
    }
    return contributions;
}

std::vector<CouplingExchangeDeclaration>
CouplingDefinitionBuilder::buildPartitionedExchangeDeclarations() const
{
    return partitioned_.declarations();
}

} // namespace coupling
} // namespace FE
} // namespace svmp

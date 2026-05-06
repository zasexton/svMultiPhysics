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

std::string defaultConsumerPortName(
    const CouplingPayloadExtractionRequest& request)
{
    const std::string participant_prefix =
        request.producer_participant_name + "_";
    if (startsWith(request.exchange_name, participant_prefix)) {
        return request.consumer_participant_name + "_" +
               request.exchange_name.substr(participant_prefix.size());
    }
    return request.exchange_name + ".consumer";
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

void applySidePairedPayloadExtractionDefaults(
    CouplingPayloadExtractionRequest& request,
    const CouplingSidePairedInterfaceRequest& interface,
    bool infer_shared_region,
    bool infer_ports)
{
    if (infer_shared_region && request.shared_region_name.empty()) {
        request.shared_region_name = interface.shared_region_name;
    }
    if (infer_ports && request.producer_port_name.empty()) {
        request.producer_port_name = request.exchange_name;
    }
    if (infer_ports && request.consumer_port_name.empty()) {
        request.consumer_port_name = defaultConsumerPortName(request);
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

CouplingOptionalFieldRoleBuilder::CouplingOptionalFieldRoleBuilder(
    CouplingSidePairedPDEBuilder& builder,
    std::size_t role_index)
    : builder_(&builder)
    , role_index_(role_index)
{
}

CouplingOptionalFieldRoleBuilder& CouplingOptionalFieldRoleBuilder::diagnostic(
    CouplingDiagnostic missing_field_diagnostic)
{
    FE_THROW_IF(builder_ == nullptr, InvalidArgumentException,
                "optional coupling field role builder is empty");
    builder_->optionalRole(role_index_).missing_field_diagnostic =
        std::move(missing_field_diagnostic);
    return *this;
}

CouplingSidePairedPDEBuilder&
CouplingOptionalFieldRoleBuilder::requiredWhen(bool enabled)
{
    FE_THROW_IF(builder_ == nullptr, InvalidArgumentException,
                "optional coupling field role builder is empty");
    builder_->optionalRole(role_index_).enabled = enabled;
    return *builder_;
}

CouplingInterfaceFieldBuilder::CouplingInterfaceFieldBuilder(
    CouplingSidePairedPDEBuilder& builder)
    : builder_(&builder)
{
}

CouplingContractInterfaceFieldRequest& CouplingInterfaceFieldBuilder::request()
{
    FE_THROW_IF(builder_ == nullptr, InvalidArgumentException,
                "interface field builder is empty");
    return builder_->ensureInterfaceField();
}

CouplingInterfaceFieldBuilder& CouplingInterfaceFieldBuilder::enabled(
    bool enabled)
{
    request().enabled = enabled;
    return *this;
}

CouplingInterfaceFieldBuilder& CouplingInterfaceFieldBuilder::name(
    std::string field_name)
{
    request().field_name = std::move(field_name);
    return *this;
}

CouplingInterfaceFieldBuilder& CouplingInterfaceFieldBuilder::space(
    std::shared_ptr<const spaces::FunctionSpace> space)
{
    request().space = std::move(space);
    return *this;
}

CouplingInterfaceFieldBuilder& CouplingInterfaceFieldBuilder::components(
    int components)
{
    request().components = components;
    return *this;
}

CouplingInterfaceFieldBuilder& CouplingInterfaceFieldBuilder::fieldNamespace(
    std::string field_namespace)
{
    request().field_namespace = std::move(field_namespace);
    return *this;
}

CouplingInterfaceFieldBuilder&
CouplingInterfaceFieldBuilder::systemParticipant(
    std::optional<std::string> participant_name)
{
    request().system_participant_name = std::move(participant_name);
    return *this;
}

CouplingInterfaceFieldBuilder&
CouplingInterfaceFieldBuilder::systemParticipant(std::string participant_name)
{
    return systemParticipant(
        std::optional<std::string>{std::move(participant_name)});
}

CouplingInterfaceFieldBuilder& CouplingInterfaceFieldBuilder::sharedRegion(
    std::string shared_region_name)
{
    request().shared_region_name = std::move(shared_region_name);
    return *this;
}

CouplingInterfaceFieldBuilder& CouplingInterfaceFieldBuilder::requirement(
    CouplingRequirement requirement)
{
    request().requirement = requirement;
    return *this;
}

CouplingInterfaceFieldBuilder&
CouplingInterfaceFieldBuilder::localEliminationPolicy(
    CouplingLocalEliminationPolicy policy)
{
    request().local_elimination_policy = policy;
    return *this;
}

CouplingPartitionedExchangeAuthoringBuilder::
    CouplingPartitionedExchangeAuthoringBuilder(
        CouplingSidePairedPDEBuilder& builder,
        std::size_t channel_index)
    : builder_(&builder)
    , channel_index_(channel_index)
{
}

CouplingPartitionedFieldChannelRequest&
CouplingPartitionedExchangeAuthoringBuilder::channel()
{
    FE_THROW_IF(builder_ == nullptr, InvalidArgumentException,
                "partitioned exchange builder is empty");
    return builder_->channel(channel_index_);
}

CouplingPartitionedExchangeAuthoringBuilder&
CouplingPartitionedExchangeAuthoringBuilder::from(
    const CouplingFieldRoleHandle& producer_field)
{
    auto& request = channel();
    request.producer_participant_name = producer_field.field.participant_name;
    request.producer_field_name = producer_field.field.field_name;
    return *this;
}

CouplingPartitionedExchangeAuthoringBuilder&
CouplingPartitionedExchangeAuthoringBuilder::to(
    const CouplingFieldRoleHandle& consumer_field)
{
    auto& request = channel();
    request.consumer_participant_name = consumer_field.field.participant_name;
    request.consumer_field_name = consumer_field.field.field_name;
    return *this;
}

CouplingPartitionedExchangeAuthoringBuilder&
CouplingPartitionedExchangeAuthoringBuilder::transfer(
    CouplingTransferDeclaration declaration)
{
    channel().transfer = std::move(declaration);
    return *this;
}

CouplingPartitionedExchangeAuthoringBuilder&
CouplingPartitionedExchangeAuthoringBuilder::strategy(
    CouplingPartitionedStrategyDeclaration declaration)
{
    channel().strategy = declaration;
    return *this;
}

CouplingPartitionedExchangeAuthoringBuilder&
CouplingPartitionedExchangeAuthoringBuilder::producerTemporal(
    CouplingTemporalSlotDescriptor temporal)
{
    channel().producer_temporal = temporal;
    return *this;
}

CouplingPartitionedExchangeAuthoringBuilder&
CouplingPartitionedExchangeAuthoringBuilder::consumerTemporal(
    CouplingTemporalSlotDescriptor temporal)
{
    channel().consumer_temporal = temporal;
    return *this;
}

CouplingPartitionedExchangeAuthoringBuilder&
CouplingPartitionedExchangeAuthoringBuilder::producerPort(
    std::string port_name)
{
    channel().producer_port_name = std::move(port_name);
    return *this;
}

CouplingPartitionedExchangeAuthoringBuilder&
CouplingPartitionedExchangeAuthoringBuilder::consumerPort(
    std::string port_name)
{
    channel().consumer_port_name = std::move(port_name);
    return *this;
}

CouplingPartitionedExchangeAuthoringBuilder&
CouplingPartitionedExchangeAuthoringBuilder::sharedRegion(
    std::string shared_region_name)
{
    channel().shared_region_name = std::move(shared_region_name);
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder::
    CouplingPartitionedPayloadExtractionAuthoringBuilder(
        CouplingSidePairedPDEBuilder& builder,
        std::size_t extraction_index)
    : builder_(&builder)
    , extraction_index_(extraction_index)
{
}

CouplingPayloadExtractionRequest&
CouplingPartitionedPayloadExtractionAuthoringBuilder::request()
{
    FE_THROW_IF(builder_ == nullptr, InvalidArgumentException,
                "partitioned payload extraction builder is empty");
    return builder_->payloadExtraction(extraction_index_);
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::contribution(
    std::string contribution_name)
{
    request().contribution_name = std::move(contribution_name);
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::from(
    const CouplingFieldRoleHandle& producer_field)
{
    auto& req = request();
    req.producer_participant_name = producer_field.field.participant_name;
    req.producer_field_name = producer_field.field.field_name;
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::to(
    const CouplingFieldRoleHandle& consumer_field)
{
    auto& req = request();
    req.consumer_participant_name = consumer_field.field.participant_name;
    req.consumer_field_name = consumer_field.field.field_name;
    req.value = consumer_field.value;
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::preferred(
    CouplingPayloadKind kind)
{
    request().preferred_kind = kind;
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::fallback(
    CouplingPayloadFallbackPolicy policy)
{
    request().fallback_policy = policy;
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::transfer(
    CouplingTransferDeclaration declaration)
{
    request().transfer = std::move(declaration);
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::strategy(
    CouplingPartitionedStrategyDeclaration declaration)
{
    request().strategy = declaration;
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::producerTemporal(
    CouplingTemporalSlotDescriptor temporal)
{
    request().producer_temporal = temporal;
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::consumerTemporal(
    CouplingTemporalSlotDescriptor temporal)
{
    request().consumer_temporal = temporal;
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::producerPort(
    std::string port_name)
{
    request().producer_port_name = std::move(port_name);
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::consumerPort(
    std::string port_name)
{
    request().consumer_port_name = std::move(port_name);
    return *this;
}

CouplingPartitionedPayloadExtractionAuthoringBuilder&
CouplingPartitionedPayloadExtractionAuthoringBuilder::sharedRegion(
    std::string shared_region_name)
{
    request().shared_region_name = std::move(shared_region_name);
    return *this;
}

CouplingSidePairedPDEBuilder::CouplingSidePairedPDEBuilder(
    CouplingDefinitionBuilder& builder,
    std::string relation_name)
    : builder_(&builder)
{
    request_.interface.relation_name = std::move(relation_name);
}

CouplingSidePairedPDEBuilder& CouplingSidePairedPDEBuilder::onInterface(
    std::string shared_region_name)
{
    request_.interface.shared_region_name = std::move(shared_region_name);
    if (request_.interface_field.has_value() &&
        request_.interface_field->shared_region_name.empty()) {
        request_.interface_field->shared_region_name =
            request_.interface.shared_region_name;
    }
    return *this;
}

CouplingSidePairedPDEBuilder& CouplingSidePairedPDEBuilder::between(
    std::string first_participant_name,
    std::string second_participant_name)
{
    request_.interface.first_participant_name =
        std::move(first_participant_name);
    request_.interface.second_participant_name =
        std::move(second_participant_name);
    return *this;
}

CouplingSidePairedPDEBuilder& CouplingSidePairedPDEBuilder::mode(
    CouplingMode mode)
{
    request_.interface.mode = mode;
    return *this;
}

CouplingSidePairedPDEBuilder& CouplingSidePairedPDEBuilder::enforcement(
    std::string enforcement_strategy)
{
    request_.interface.enforcement_strategy =
        std::move(enforcement_strategy);
    return *this;
}

CouplingSidePairedPDEBuilder&
CouplingSidePairedPDEBuilder::partitionedStrategies(
    std::vector<CouplingPartitionedSolveStrategy> strategies)
{
    request_.interface.partitioned_solve_strategies = std::move(strategies);
    return *this;
}

CouplingFieldRoleHandle CouplingSidePairedPDEBuilder::scalarField(
    std::string participant_name,
    std::string field_name)
{
    auto role =
        scalarFieldRole(std::move(participant_name), std::move(field_name));
    auto handle = CouplingFieldRoleHandle{
        .field = role.field,
        .value = role.value,
    };
    request_.required_fields.push_back(std::move(role));
    return handle;
}

CouplingFieldRoleHandle CouplingSidePairedPDEBuilder::vectorField(
    std::string participant_name,
    std::string field_name,
    int components)
{
    auto role = vectorFieldRole(std::move(participant_name),
                                std::move(field_name),
                                components);
    auto handle = CouplingFieldRoleHandle{
        .field = role.field,
        .value = role.value,
    };
    request_.required_fields.push_back(std::move(role));
    return handle;
}

CouplingSidePairedPDEBuilder&
CouplingSidePairedPDEBuilder::requiredScalarField(
    std::string participant_name,
    std::string field_name)
{
    request_.required_fields.push_back(
        scalarFieldRole(std::move(participant_name), std::move(field_name)));
    return *this;
}

CouplingSidePairedPDEBuilder&
CouplingSidePairedPDEBuilder::requiredVectorField(
    std::string participant_name,
    std::string field_name,
    int components)
{
    request_.required_fields.push_back(
        vectorFieldRole(std::move(participant_name),
                        std::move(field_name),
                        components));
    return *this;
}

CouplingOptionalFieldRoleBuilder CouplingSidePairedPDEBuilder::optionalField(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    CouplingValueDescriptor value)
{
    request_.optional_fields.push_back(optionalFieldRole(
        std::move(participant_name),
        std::move(field_name),
        std::move(value),
        false));
    return CouplingOptionalFieldRoleBuilder(
        *this,
        request_.optional_fields.size() - 1);
}

CouplingOptionalFieldRoleBuilder
CouplingSidePairedPDEBuilder::optionalVectorField(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    int components)
{
    return optionalField(std::move(participant_name),
                         std::move(field_name),
                         vectorValue(components));
}

CouplingInterfaceFieldBuilder CouplingSidePairedPDEBuilder::interfaceField()
{
    ensureInterfaceField();
    return CouplingInterfaceFieldBuilder(*this);
}

CouplingPartitionedExchangeAuthoringBuilder
CouplingSidePairedPDEBuilder::partitionedExchange(std::string channel_name)
{
    request_.partitioned_channels.push_back(
        CouplingPartitionedFieldChannelRequest{
            .channel_name = std::move(channel_name),
        });
    return CouplingPartitionedExchangeAuthoringBuilder(
        *this,
        request_.partitioned_channels.size() - 1);
}

CouplingPartitionedPayloadExtractionAuthoringBuilder
CouplingSidePairedPDEBuilder::partitionedPayloadFromForm(
    std::string exchange_name)
{
    request_.partitioned_payload_extractions.push_back(
        CouplingPayloadExtractionRequest{
            .exchange_name = std::move(exchange_name),
        });
    return CouplingPartitionedPayloadExtractionAuthoringBuilder(
        *this,
        request_.partitioned_payload_extractions.size() - 1);
}

CouplingDefinitionBuilder& CouplingSidePairedPDEBuilder::monolithicForms(
    CouplingMonolithicFormsCallback callback)
{
    request_.monolithic_forms = std::move(callback);
    return install();
}

CouplingDefinitionBuilder& CouplingSidePairedPDEBuilder::install()
{
    FE_THROW_IF(builder_ == nullptr, InvalidArgumentException,
                "side-paired PDE builder is empty");
    FE_THROW_IF(installed_, InvalidArgumentException,
                "side-paired PDE coupling has already been installed");
    installed_ = true;
    return builder_->sidePairedPDECoupling(std::move(request_));
}

CouplingContractInterfaceFieldRequest&
CouplingSidePairedPDEBuilder::ensureInterfaceField()
{
    if (!request_.interface_field.has_value()) {
        request_.interface_field = CouplingContractInterfaceFieldRequest{
            .shared_region_name = request_.interface.shared_region_name,
        };
    }
    return *request_.interface_field;
}

CouplingPartitionedFieldChannelRequest& CouplingSidePairedPDEBuilder::channel(
    std::size_t channel_index)
{
    FE_THROW_IF(channel_index >= request_.partitioned_channels.size(),
                InvalidArgumentException,
                "partitioned exchange index is out of range");
    return request_.partitioned_channels[channel_index];
}

CouplingPayloadExtractionRequest&
CouplingSidePairedPDEBuilder::payloadExtraction(std::size_t extraction_index)
{
    FE_THROW_IF(extraction_index >= request_.partitioned_payload_extractions.size(),
                InvalidArgumentException,
                "partitioned payload extraction index is out of range");
    return request_.partitioned_payload_extractions[extraction_index];
}

CouplingOptionalFieldRoleRequest& CouplingSidePairedPDEBuilder::optionalRole(
    std::size_t role_index)
{
    FE_THROW_IF(role_index >= request_.optional_fields.size(),
                InvalidArgumentException,
                "optional coupling field role index is out of range");
    return request_.optional_fields[role_index];
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

CouplingSidePairedPDEBuilder CouplingDefinitionBuilder::sidePairedPDE(
    std::string relation_name)
{
    return CouplingSidePairedPDEBuilder(*this, std::move(relation_name));
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
        for (auto& extraction : request.partitioned_payload_extractions) {
            applySidePairedPayloadExtractionDefaults(
                extraction,
                interface_defaults,
                request.infer_partitioned_channel_shared_regions,
                request.infer_partitioned_channel_ports);
            payloadExtraction(std::move(extraction));
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

    if ((mode == CouplingMode::Monolithic ||
         !request.partitioned_payload_extractions.empty()) &&
        request.monolithic_forms) {
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

CouplingDefinitionBuilder& CouplingDefinitionBuilder::payloadExtraction(
    CouplingPayloadExtractionRequest request)
{
    payload_extraction_requests_.push_back(std::move(request));
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
    return !partitioned_.declarations().empty() ||
           !payload_extraction_requests_.empty();
}

bool CouplingDefinitionBuilder::hasPayloadExtractions() const noexcept
{
    return !payload_extraction_requests_.empty();
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
    declaration.payload_extraction_requests.insert(
        declaration.payload_extraction_requests.end(),
        payload_extraction_requests_.begin(),
        payload_extraction_requests_.end());
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

const std::vector<CouplingPayloadExtractionRequest>&
CouplingDefinitionBuilder::payloadExtractionRequests() const noexcept
{
    return payload_extraction_requests_;
}

} // namespace coupling
} // namespace FE
} // namespace svmp

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/PartitionedCouplingBuilder.h"

#include "Core/FEException.h"

#include <algorithm>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

CouplingEndpointRef fieldEndpoint(const CouplingFieldUse& field)
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::Field,
        .participant_name = field.participant_name,
        .endpoint_name = field.field_name,
    };
}

std::string producerPortName(std::string_view exchange_name)
{
    return std::string(exchange_name) + ".producer";
}

std::string consumerPortName(std::string_view exchange_name)
{
    return std::string(exchange_name) + ".consumer";
}

} // namespace

PartitionedExchangeBuilder::PartitionedExchangeBuilder(
    PartitionedCouplingBuilder& builder,
    std::size_t exchange_index)
    : builder_(&builder)
    , exchange_index_(exchange_index)
{
}

PartitionedExchangeBuilder& PartitionedExchangeBuilder::sharedInterface(
    std::string_view name)
{
    exchange().shared_region_name = std::string(name);
    return *this;
}

PartitionedExchangeBuilder& PartitionedExchangeBuilder::producerRegion(
    CouplingRegionEndpointDeclaration region)
{
    exchange().producer_region = std::move(region);
    return *this;
}

PartitionedExchangeBuilder& PartitionedExchangeBuilder::consumerRegion(
    CouplingRegionEndpointDeclaration region)
{
    exchange().consumer_region = std::move(region);
    return *this;
}

PartitionedExchangeBuilder& PartitionedExchangeBuilder::value(
    CouplingValueDescriptor descriptor)
{
    exchange().value = std::move(descriptor);
    return *this;
}

PartitionedExchangeBuilder& PartitionedExchangeBuilder::transfer(
    CouplingTransferDeclaration declaration)
{
    exchange().transfer = std::move(declaration);
    return *this;
}

PartitionedExchangeBuilder& PartitionedExchangeBuilder::producerTemporal(
    CouplingTemporalSlotDescriptor temporal)
{
    auto& current = exchange();
    FE_THROW_IF(!current.producer.has_value(), InvalidArgumentException,
                "partitioned exchange producer endpoint is missing");
    current.producer->temporal = temporal;
    return *this;
}

PartitionedExchangeBuilder& PartitionedExchangeBuilder::consumerTemporal(
    CouplingTemporalSlotDescriptor temporal)
{
    auto& current = exchange();
    FE_THROW_IF(!current.consumer.has_value(), InvalidArgumentException,
                "partitioned exchange consumer endpoint is missing");
    current.consumer->temporal = temporal;
    return *this;
}

const CouplingExchangeDeclaration& PartitionedExchangeBuilder::declaration() const
{
    return exchange();
}

CouplingExchangeDeclaration& PartitionedExchangeBuilder::exchange()
{
    FE_CHECK_NOT_NULL(builder_, "partitioned exchange builder owner");
    return builder_->mutableExchange(exchange_index_);
}

const CouplingExchangeDeclaration& PartitionedExchangeBuilder::exchange() const
{
    FE_CHECK_NOT_NULL(builder_, "partitioned exchange builder owner");
    return builder_->exchange(exchange_index_);
}

PartitionedCouplingBuilder::PartitionedCouplingBuilder(std::string contract_name)
    : contract_name_(std::move(contract_name))
{
    FE_THROW_IF(contract_name_.empty(), InvalidArgumentException,
                "partitioned coupling builder requires a contract name");
}

PartitionedCouplingBuilder::PartitionedCouplingBuilder(
    std::string contract_name,
    std::span<const CouplingFieldRequirement> field_requirements)
    : PartitionedCouplingBuilder(std::move(contract_name))
{
    field_requirements_.assign(field_requirements.begin(),
                               field_requirements.end());
}

std::string_view PartitionedCouplingBuilder::contractName() const noexcept
{
    return contract_name_;
}

PartitionedCouplingBuilder& PartitionedCouplingBuilder::addFieldRequirement(
    CouplingFieldRequirement requirement)
{
    field_requirements_.push_back(std::move(requirement));
    return *this;
}

PartitionedExchangeBuilder PartitionedCouplingBuilder::exchange(
    std::string_view name,
    const CouplingFieldUse& producer_field,
    const CouplingFieldUse& consumer_field)
{
    auto handle = exchange(name,
                           fieldEndpoint(producer_field),
                           fieldEndpoint(consumer_field));
    if (auto value = inferredExchangeValueDescriptor(producer_field,
                                                     consumer_field)) {
        handle.value(std::move(*value));
    }
    return handle;
}

PartitionedExchangeBuilder PartitionedCouplingBuilder::exchange(
    std::string_view name,
    CouplingEndpointRef producer_endpoint,
    CouplingEndpointRef consumer_endpoint)
{
    FE_THROW_IF(name.empty(), InvalidArgumentException,
                "partitioned exchange requires a name");

    CouplingExchangeDeclaration declaration;
    declaration.producer_port = CouplingPortId{
        .contract_instance_name = contract_name_,
        .port_name = producerPortName(name),
    };
    declaration.consumer_port = CouplingPortId{
        .contract_instance_name = contract_name_,
        .port_name = consumerPortName(name),
    };
    declaration.producer = std::move(producer_endpoint);
    declaration.consumer = std::move(consumer_endpoint);

    declarations_.push_back(std::move(declaration));
    return PartitionedExchangeBuilder(*this, declarations_.size() - 1u);
}

const std::vector<CouplingExchangeDeclaration>&
PartitionedCouplingBuilder::declarations() const noexcept
{
    return declarations_;
}

std::vector<CouplingExchangeDeclaration>
PartitionedCouplingBuilder::takeDeclarations()
{
    return std::move(declarations_);
}

PartitionedCouplingBuilder& PartitionedCouplingBuilder::group(
    std::string name,
    std::vector<std::string> participant_names)
{
    FE_THROW_IF(name.empty(), InvalidArgumentException,
                "partitioned coupling group hint requires a name");
    FE_THROW_IF(participant_names.empty(), InvalidArgumentException,
                "partitioned coupling group hint requires participants");
    group_hints_.push_back(CouplingGroupHint{
        .name = std::move(name),
        .participant_names = std::move(participant_names),
    });
    return *this;
}

const std::vector<CouplingGroupHint>&
PartitionedCouplingBuilder::groupHints() const noexcept
{
    return group_hints_;
}

std::vector<CouplingGroupHint> PartitionedCouplingBuilder::takeGroupHints()
{
    return std::move(group_hints_);
}

std::optional<CouplingValueDescriptor>
PartitionedCouplingBuilder::valueDescriptorForField(
    const CouplingFieldUse& field) const
{
    const auto it = std::find_if(
        field_requirements_.begin(),
        field_requirements_.end(),
        [&](const CouplingFieldRequirement& requirement) {
            return requirement.field.participant_name == field.participant_name &&
                   requirement.field.field_name == field.field_name;
        });
    if (it == field_requirements_.end()) {
        return std::nullopt;
    }
    return it->value;
}

std::optional<CouplingValueDescriptor>
PartitionedCouplingBuilder::inferredExchangeValueDescriptor(
    const CouplingFieldUse& producer_field,
    const CouplingFieldUse& consumer_field) const
{
    const auto producer_value = valueDescriptorForField(producer_field);
    const auto consumer_value = valueDescriptorForField(consumer_field);
    if (producer_value.has_value() && consumer_value.has_value()) {
        FE_THROW_IF(!couplingValueDescriptorsCompatible(*producer_value,
                                                        *consumer_value),
                    InvalidArgumentException,
                    "partitioned exchange field requirements have incompatible value descriptors");
        return producer_value;
    }
    if (producer_value.has_value()) {
        return producer_value;
    }
    return consumer_value;
}

CouplingExchangeDeclaration& PartitionedCouplingBuilder::mutableExchange(
    std::size_t index)
{
    return declarations_.at(index);
}

const CouplingExchangeDeclaration& PartitionedCouplingBuilder::exchange(
    std::size_t index) const
{
    return declarations_.at(index);
}

} // namespace coupling
} // namespace FE
} // namespace svmp

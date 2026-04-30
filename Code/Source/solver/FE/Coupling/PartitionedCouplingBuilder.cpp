/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/PartitionedCouplingBuilder.h"

#include "Core/FEException.h"

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

std::string_view PartitionedCouplingBuilder::contractName() const noexcept
{
    return contract_name_;
}

PartitionedExchangeBuilder PartitionedCouplingBuilder::exchange(
    std::string_view name,
    const CouplingFieldUse& producer_field,
    const CouplingFieldUse& consumer_field)
{
    return exchange(name,
                    fieldEndpoint(producer_field),
                    fieldEndpoint(consumer_field));
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

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_PARTITIONEDCOUPLINGBUILDER_H
#define SVMP_FE_COUPLING_PARTITIONEDCOUPLINGBUILDER_H

/**
 * @file PartitionedCouplingBuilder.h
 * @brief Compact authoring helper for partitioned coupling exchanges.
 */

#include "Coupling/CouplingDeclaration.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

class PartitionedCouplingBuilder;

class PartitionedExchangeBuilder {
public:
    PartitionedExchangeBuilder(PartitionedCouplingBuilder& builder,
                               std::size_t exchange_index);

    PartitionedExchangeBuilder& sharedInterface(std::string_view name);
    PartitionedExchangeBuilder& producerRegion(
        CouplingRegionEndpointDeclaration region);
    PartitionedExchangeBuilder& consumerRegion(
        CouplingRegionEndpointDeclaration region);
    PartitionedExchangeBuilder& value(CouplingValueDescriptor descriptor);
    PartitionedExchangeBuilder& transfer(CouplingTransferDeclaration declaration);
    PartitionedExchangeBuilder& producerTemporal(
        CouplingTemporalSlotDescriptor temporal);
    PartitionedExchangeBuilder& consumerTemporal(
        CouplingTemporalSlotDescriptor temporal);

    [[nodiscard]] const CouplingExchangeDeclaration& declaration() const;

private:
    [[nodiscard]] CouplingExchangeDeclaration& exchange();
    [[nodiscard]] const CouplingExchangeDeclaration& exchange() const;

    PartitionedCouplingBuilder* builder_{nullptr};
    std::size_t exchange_index_{0};
};

class PartitionedCouplingBuilder {
public:
    explicit PartitionedCouplingBuilder(std::string contract_name);

    [[nodiscard]] std::string_view contractName() const noexcept;

    [[nodiscard]] PartitionedExchangeBuilder exchange(
        std::string_view name,
        const CouplingFieldUse& producer_field,
        const CouplingFieldUse& consumer_field);

    [[nodiscard]] PartitionedExchangeBuilder exchange(
        std::string_view name,
        CouplingEndpointRef producer_endpoint,
        CouplingEndpointRef consumer_endpoint);

    [[nodiscard]] const std::vector<CouplingExchangeDeclaration>&
    declarations() const noexcept;
    [[nodiscard]] std::vector<CouplingExchangeDeclaration> takeDeclarations();

private:
    friend class PartitionedExchangeBuilder;

    [[nodiscard]] CouplingExchangeDeclaration& mutableExchange(
        std::size_t index);
    [[nodiscard]] const CouplingExchangeDeclaration& exchange(
        std::size_t index) const;

    std::string contract_name_;
    std::vector<CouplingExchangeDeclaration> declarations_;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_PARTITIONEDCOUPLINGBUILDER_H

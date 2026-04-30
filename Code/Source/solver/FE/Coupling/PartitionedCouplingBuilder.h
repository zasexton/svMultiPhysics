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
#include <optional>
#include <span>
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
    PartitionedExchangeBuilder& strategy(
        CouplingPartitionedStrategyDeclaration declaration);
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
    PartitionedCouplingBuilder(
        std::string contract_name,
        std::span<const CouplingFieldRequirement> field_requirements);

    [[nodiscard]] std::string_view contractName() const noexcept;
    PartitionedCouplingBuilder& addFieldRequirement(
        CouplingFieldRequirement requirement);

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
    PartitionedCouplingBuilder& group(std::string name,
                                      std::vector<std::string> participant_names);
    [[nodiscard]] const std::vector<CouplingGroupHint>& groupHints() const noexcept;
    [[nodiscard]] std::vector<CouplingGroupHint> takeGroupHints();

private:
    friend class PartitionedExchangeBuilder;

    [[nodiscard]] std::optional<CouplingValueDescriptor> valueDescriptorForField(
        const CouplingFieldUse& field) const;
    [[nodiscard]] std::optional<CouplingValueDescriptor>
    inferredExchangeValueDescriptor(const CouplingFieldUse& producer_field,
                                    const CouplingFieldUse& consumer_field) const;
    [[nodiscard]] CouplingExchangeDeclaration& mutableExchange(
        std::size_t index);
    [[nodiscard]] const CouplingExchangeDeclaration& exchange(
        std::size_t index) const;

    std::string contract_name_;
    std::vector<CouplingFieldRequirement> field_requirements_;
    std::vector<CouplingExchangeDeclaration> declarations_;
    std::vector<CouplingGroupHint> group_hints_;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_PARTITIONEDCOUPLINGBUILDER_H

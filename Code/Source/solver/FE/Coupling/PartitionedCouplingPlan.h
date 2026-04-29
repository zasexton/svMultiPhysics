/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_PARTITIONEDCOUPLINGPLAN_H
#define SVMP_FE_COUPLING_PARTITIONEDCOUPLINGPLAN_H

/**
 * @file PartitionedCouplingPlan.h
 * @brief Metadata-only partitioned coupling exchange plan.
 */

#include "Coupling/CouplingContext.h"
#include "Coupling/TransferPlan.h"

#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

struct CouplingExchangeDeclaration {
    CouplingPortId producer_port;
    CouplingPortId consumer_port;
    CouplingValueDescriptor value{};
    std::optional<CouplingEndpointRef> producer;
    std::optional<CouplingEndpointRef> consumer;
    std::optional<std::string> shared_region_name;
    std::optional<CouplingRegionEndpointDeclaration> producer_region;
    std::optional<CouplingRegionEndpointDeclaration> consumer_region;
    CouplingTransferDeclaration transfer{};
};

enum class CouplingResolvedTemporalBackingKind : std::uint8_t {
    None,
    SystemStateCurrent,
    SystemStateAccepted,
    SystemStatePredicted,
    SystemStateHistory,
    SystemStateStage,
    ProviderDefined,
    ExternalBuffer,
};

struct ResolvedCouplingTemporalSlot {
    CouplingTemporalSlotDescriptor request{};
    CouplingTemporalSlotDescriptor provided{};
    CouplingResolvedTemporalBackingKind backing{CouplingResolvedTemporalBackingKind::None};
    std::string provider_name;
    std::optional<int> storage_index;
    std::uint64_t state_revision_key{0};
    Real time{0.0};
};

struct ResolvedCouplingEndpoint {
    CouplingEndpointRef declaration_provenance{};
    CouplingEndpointKind resolved_kind{CouplingEndpointKind::ExternalBuffer};
    CouplingValueDescriptor value{};
    std::optional<std::string> resolved_participant_name;
    std::string system_name;
    std::string registry_provider;
    std::string resolved_endpoint_key;
    ResolvedCouplingTemporalSlot temporal{};
    FieldId field_id{INVALID_FIELD_ID};
    std::optional<CouplingExternalBufferDescriptor> external_buffer;
    std::uint64_t layout_revision_key{0};
    std::uint64_t registry_revision_key{0};
};

struct CouplingExchange {
    CouplingPortId producer_port;
    CouplingPortId consumer_port;
    CouplingValueDescriptor value{};
    ResolvedCouplingEndpoint producer;
    ResolvedCouplingEndpoint consumer;
    std::optional<std::string> shared_region_name;
    std::optional<CouplingRegionRef> producer_region;
    std::optional<CouplingRegionRef> consumer_region;
    ResolvedCouplingTransfer transfer{};
};

struct CouplingGroupHint {
    std::string name;
    std::vector<std::string> participant_names;
};

struct CouplingExchangeCycle {
    std::vector<CouplingPortId> ports;
};

struct PartitionedCouplingPlan {
    std::vector<CouplingExchange> exchanges;
    std::vector<CouplingGroupHint> group_hints;
    std::vector<CouplingExchangeCycle> cycles;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_PARTITIONEDCOUPLINGPLAN_H

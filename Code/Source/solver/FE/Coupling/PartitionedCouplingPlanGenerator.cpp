/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/PartitionedCouplingPlanGenerator.h"

#include "Coupling/CouplingDeclaration.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

CouplingResolvedTemporalBackingKind backingForEndpoint(const CouplingEndpointRef& endpoint) noexcept
{
    if (endpoint.kind == CouplingEndpointKind::Field) {
        switch (endpoint.temporal.slot) {
        case CouplingTemporalSlot::Current:
            return CouplingResolvedTemporalBackingKind::SystemStateCurrent;
        case CouplingTemporalSlot::History:
            return CouplingResolvedTemporalBackingKind::SystemStateHistory;
        case CouplingTemporalSlot::Accepted:
        case CouplingTemporalSlot::Predicted:
        case CouplingTemporalSlot::Stage:
        case CouplingTemporalSlot::External:
            return CouplingResolvedTemporalBackingKind::None;
        }
    }
    if (endpoint.kind == CouplingEndpointKind::ExternalBuffer) {
        return CouplingResolvedTemporalBackingKind::ExternalBuffer;
    }
    if (endpoint.kind == CouplingEndpointKind::Parameter &&
        endpoint.temporal.slot == CouplingTemporalSlot::Current) {
        return CouplingResolvedTemporalBackingKind::ProviderDefined;
    }
    if (endpoint.kind == CouplingEndpointKind::AuxiliaryInput &&
        endpoint.temporal.slot == CouplingTemporalSlot::Current) {
        return CouplingResolvedTemporalBackingKind::AuxiliaryCurrent;
    }
    if (endpoint.kind == CouplingEndpointKind::RegionData &&
        endpoint.temporal.slot == CouplingTemporalSlot::Current) {
        return CouplingResolvedTemporalBackingKind::ProviderDefined;
    }
    return CouplingResolvedTemporalBackingKind::None;
}

std::optional<int> resolvedTemporalStorageIndex(const CouplingTemporalSlotDescriptor& temporal)
{
    if (temporal.slot == CouplingTemporalSlot::History &&
        temporal.history_index.has_value()) {
        return *temporal.history_index - 1;
    }
    if (temporal.slot == CouplingTemporalSlot::Stage &&
        temporal.stage_index.has_value()) {
        return *temporal.stage_index;
    }
    return std::nullopt;
}

bool temporalSlotsEqual(const CouplingTemporalSlotDescriptor& lhs,
                        const CouplingTemporalSlotDescriptor& rhs) noexcept
{
    return lhs.slot == rhs.slot &&
           lhs.history_index == rhs.history_index &&
           lhs.stage_index == rhs.stage_index;
}

bool supportsTemporalSlot(const std::vector<CouplingTemporalSlotDescriptor>& slots,
                          const CouplingTemporalSlotDescriptor& request)
{
    return std::any_of(slots.begin(), slots.end(),
                       [&request](const CouplingTemporalSlotDescriptor& slot) {
                           return temporalSlotsEqual(slot, request);
                       });
}

bool supportsRank(const std::vector<CouplingValueRank>& ranks,
                  CouplingValueRank rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

bool fitsUint32(std::size_t value) noexcept
{
    return value <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
}

bool feQuantityShapeMatchesValue(const systems::FEQuantityShape& shape,
                                 const CouplingValueDescriptor& value) noexcept
{
    if (shape.components != value.components) {
        return false;
    }
    switch (shape.kind) {
    case systems::FEQuantityShapeKind::Scalar:
        return value.rank == CouplingValueRank::Scalar && value.components == 1;
    case systems::FEQuantityShapeKind::Vector:
        return value.rank == CouplingValueRank::Vector;
    case systems::FEQuantityShapeKind::Tensor:
        return value.rank == CouplingValueRank::Rank2Tensor;
    }
    return false;
}

std::optional<std::string_view> endpointScope(const CouplingEndpointRef& endpoint)
{
    if (!endpoint.participant_name.has_value()) {
        return std::nullopt;
    }
    return std::string_view{*endpoint.participant_name};
}

CouplingValidationResult validateEndpointResolutionSupport(
    const CouplingContext& ctx,
    const CouplingEndpointRef& endpoint,
    const CouplingValueDescriptor& value,
    std::string_view role)
{
    CouplingValidationResult result;
    if (endpoint.kind == CouplingEndpointKind::Field) {
        switch (endpoint.temporal.slot) {
        case CouplingTemporalSlot::Current:
        case CouplingTemporalSlot::History:
            break;
        case CouplingTemporalSlot::Accepted:
        case CouplingTemporalSlot::Predicted:
        case CouplingTemporalSlot::Stage:
        case CouplingTemporalSlot::External:
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = endpoint.participant_name.value_or(""),
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " field endpoint temporal slot " +
                           toString(endpoint.temporal.slot) +
                           " requires a registered provider extension",
            });
            break;
        }
        return result;
    }

    if (endpoint.kind == CouplingEndpointKind::ExternalBuffer) {
        if (endpoint.participant_name.has_value() &&
            !ctx.hasParticipant(*endpoint.participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " external buffer endpoint references an unknown participant",
            });
        }
        const auto* descriptor =
            ctx.externalBufferDescriptor(endpointScope(endpoint), endpoint.endpoint_name);
        if (descriptor == nullptr) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = endpoint.participant_name.value_or(""),
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " external buffer endpoint requires a registered descriptor",
            });
            return result;
        }
        if (!couplingValueDescriptorsCompatible(descriptor->value, value)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = endpoint.participant_name.value_or(""),
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " external buffer descriptor value shape does not match the exchange value descriptor",
            });
        }
        if (!supportsTemporalSlot(descriptor->supported_temporal_slots, endpoint.temporal)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = endpoint.participant_name.value_or(""),
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " external buffer descriptor does not support the requested temporal slot",
            });
        }
        if (role == "producer" &&
            descriptor->access == CouplingExternalBufferAccess::WriteOnly) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = endpoint.participant_name.value_or(""),
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned producer external buffer endpoint requires read access",
            });
        }
        if (role == "consumer" &&
            descriptor->access == CouplingExternalBufferAccess::ReadOnly) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = endpoint.participant_name.value_or(""),
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned consumer external buffer endpoint requires write access",
            });
        }
        return result;
    }

    if (endpoint.kind == CouplingEndpointKind::Parameter) {
        if (!endpoint.participant_name.has_value() ||
            endpoint.participant_name->empty()) {
            return result;
        }
        if (!ctx.hasParticipant(*endpoint.participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " parameter endpoint references an unknown participant",
            });
            return result;
        }

        const auto participant = ctx.participant(*endpoint.participant_name);
        if (participant.system == nullptr) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " parameter endpoint requires an owning system",
            });
            return result;
        }

        if (endpoint.temporal.slot != CouplingTemporalSlot::Current) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " parameter endpoint temporal slot " +
                           toString(endpoint.temporal.slot) +
                           " requires a registered provider extension",
            });
        }

        const auto& registry = participant.system->parameterRegistry();
        const auto* spec = registry.find(endpoint.endpoint_name);
        if (spec == nullptr) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " parameter endpoint requires a ParameterRegistry entry",
            });
            return result;
        }
        if (value.rank != CouplingValueRank::Scalar || value.components != 1) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " parameter endpoint value descriptor must be scalar",
            });
        }
        if (spec->type != params::ValueType::Real) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " parameter endpoint requires a Real ParameterRegistry entry",
            });
        } else if (!registry.slotOf(endpoint.endpoint_name).has_value()) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " parameter endpoint requires a Real parameter slot",
            });
        }
        return result;
    }

    if (endpoint.kind == CouplingEndpointKind::RegionData) {
        if (!endpoint.participant_name.has_value() ||
            endpoint.participant_name->empty()) {
            return result;
        }
        if (!ctx.hasParticipant(*endpoint.participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " region data endpoint references an unknown participant",
            });
            return result;
        }

        const auto participant = ctx.participant(*endpoint.participant_name);
        if (participant.system == nullptr) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " region data endpoint requires an owning system",
            });
            return result;
        }

        if (endpoint.temporal.slot != CouplingTemporalSlot::Current) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " region data endpoint temporal slot " +
                           toString(endpoint.temporal.slot) +
                           " requires provider support",
            });
        }

        const auto* registry = participant.system->feQuantityRegistryIfPresent();
        if (registry == nullptr || !registry->hasDefinition(endpoint.endpoint_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " region data endpoint requires an FEQuantityRegistry entry",
            });
            return result;
        }

        const auto& definition = registry->get(endpoint.endpoint_name);
        if (!definition.capabilities.explicit_evaluation) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " region data endpoint requires explicit evaluation support",
            });
        }
        if (!feQuantityShapeMatchesValue(definition.shape, value)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " region data endpoint value shape does not match the FE quantity definition",
            });
        }
        return result;
    }

    if (endpoint.kind == CouplingEndpointKind::AuxiliaryInput) {
        if (!endpoint.participant_name.has_value() ||
            endpoint.participant_name->empty()) {
            return result;
        }
        if (!ctx.hasParticipant(*endpoint.participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " auxiliary input endpoint references an unknown participant",
            });
            return result;
        }

        const auto participant = ctx.participant(*endpoint.participant_name);
        if (participant.system == nullptr) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " auxiliary input endpoint requires an owning system",
            });
            return result;
        }

        if (endpoint.temporal.slot != CouplingTemporalSlot::Current) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " auxiliary input endpoint temporal slot " +
                           toString(endpoint.temporal.slot) +
                           " requires provider support",
            });
        }

        const auto* registry = participant.system->auxiliaryInputRegistryIfPresent();
        if (registry == nullptr || !registry->hasInput(endpoint.endpoint_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " auxiliary input endpoint requires an AuxiliaryInputRegistry entry",
            });
            return result;
        }

        const auto& spec = registry->specOf(endpoint.endpoint_name);
        if (spec.size != value.components) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " auxiliary input endpoint component count does not match the exchange value descriptor",
            });
        }
        const auto slot = registry->slotOf(endpoint.endpoint_name);
        if (!fitsUint32(slot)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = *endpoint.participant_name,
                .endpoint_name = endpoint.endpoint_name,
                .message = "partitioned " + std::string(role) +
                           " auxiliary input endpoint slot is out of range",
            });
        }
        return result;
    }

    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .participant_name = endpoint.participant_name.value_or(""),
        .endpoint_name = endpoint.endpoint_name,
        .message = "partitioned " + std::string(role) +
                   " endpoint kind " + toString(endpoint.kind) +
                   " requires a registry resolver before plan generation",
    });
    return result;
}

CouplingValidationResult validateDriverOwnedTransferDescriptor(
    const CouplingContext& ctx,
    const CouplingExchangeDeclaration& exchange)
{
    CouplingValidationResult result;
    if (exchange.transfer.kind != CouplingTransferKind::DriverOwned ||
        exchange.transfer.driver_owned_name.empty()) {
        return result;
    }

    const auto* descriptor = ctx.driverOwnedTransfer(exchange.transfer.driver_owned_name);
    if (descriptor == nullptr) {
        result.addError("driver-owned partitioned transfer requires a registered descriptor");
        return result;
    }
    if (!supportsRank(descriptor->supported_ranks, exchange.value.rank)) {
        result.addError(
            "driver-owned partitioned transfer descriptor does not support the exchange value rank");
    }
    if (!descriptor->preserves_component_layout &&
        !exchange.value.component_layout.empty()) {
        result.addError(
            "driver-owned partitioned transfer descriptor does not preserve component layout");
    }
    if (exchange.producer.has_value() &&
        !supportsTemporalSlot(descriptor->supported_source_temporal_slots,
                              exchange.producer->temporal)) {
        result.addError(
            "driver-owned partitioned transfer descriptor does not support the producer temporal slot");
    }
    if (exchange.consumer.has_value() &&
        !supportsTemporalSlot(descriptor->supported_target_temporal_slots,
                              exchange.consumer->temporal)) {
        result.addError(
            "driver-owned partitioned transfer descriptor does not support the consumer temporal slot");
    }
    return result;
}

CouplingValidationResult validateFieldEndpointValueDescriptor(
    const CouplingContext& ctx,
    const CouplingEndpointRef& endpoint,
    const CouplingValueDescriptor& value,
    std::string_view role)
{
    CouplingValidationResult result;
    if (endpoint.kind != CouplingEndpointKind::Field ||
        !endpoint.participant_name.has_value() ||
        !ctx.hasField(*endpoint.participant_name, endpoint.endpoint_name)) {
        return result;
    }

    const auto field = ctx.field(*endpoint.participant_name, endpoint.endpoint_name);
    if (field.components != value.components) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .participant_name = *endpoint.participant_name,
            .field_name = endpoint.endpoint_name,
            .endpoint_name = endpoint.endpoint_name,
            .message = "partitioned " + std::string(role) +
                       " field endpoint component count does not match the exchange value descriptor",
        });
    }
    return result;
}

CouplingValidationResult validateInterfaceTransferShape(
    const CouplingExchangeDeclaration& exchange)
{
    CouplingValidationResult result;
    if (!isInterfaceTransferKind(exchange.transfer.kind)) {
        return result;
    }

    const auto& value = exchange.value;
    if (value.rank == CouplingValueRank::SymmetricTensor) {
        result.addError("symmetric tensor interface transfers require an explicit rank-2 payload");
    }
    if (value.rank == CouplingValueRank::GeneralTensor) {
        result.addError("general tensor interface transfers are not supported");
    }
    if (!exchange.transfer.interface_declaration.has_value()) {
        return result;
    }

    const auto& interface = *exchange.transfer.interface_declaration;
    if (interface.conservation_tolerance <= 0.0) {
        result.addError("interface transfer conservation tolerance must be positive");
    }

    const bool has_embedding =
        interface.source_embedding_policy != CouplingFrameSourceEmbeddingPolicy::None;
    const bool has_restriction =
        interface.target_restriction_policy != CouplingFrameTargetRestrictionPolicy::None;

    switch (interface.frame_policy) {
    case CouplingInterfaceFramePolicy::None:
        if (has_embedding || has_restriction) {
            result.addError(
                "interface frame embedding and restriction policies require a frame transform");
        }
        break;
    case CouplingInterfaceFramePolicy::SourceToTargetVector:
        if (value.rank != CouplingValueRank::Vector) {
            result.addError("interface vector frame transforms require vector payloads");
            break;
        }
        if (value.components < 3) {
            result.addError(
                "true 2D vector interface transforms require a registered execution adapter");
        }
        if (value.components > 3 && value.component_layout.empty()) {
            result.addError(
                "vector interface frame transforms with pass-through components require component layout");
        }
        if (value.components >= 3 && (has_embedding || has_restriction)) {
            result.addError(
                "interface frame embedding and restriction policies are reserved for true 2D vector transforms");
        }
        break;
    case CouplingInterfaceFramePolicy::SourceToTargetRank2Tensor:
        if (value.rank != CouplingValueRank::Rank2Tensor) {
            result.addError("interface rank-2 frame transforms require rank-2 tensor payloads");
            break;
        }
        if (value.components < 9) {
            result.addError("interface rank-2 frame transforms require at least nine components");
        }
        if (value.components > 9 && value.component_layout.empty()) {
            result.addError(
                "rank-2 interface frame transforms with pass-through components require component layout");
        }
        if (has_embedding || has_restriction) {
            result.addError(
                "interface frame embedding and restriction policies are only valid for true 2D vector transforms");
        }
        break;
    }

    return result;
}

std::optional<std::string> effectiveSharedRegionName(
    const std::optional<std::string>& exchange_shared_region_name,
    const CouplingRegionEndpointDeclaration& region)
{
    if (region.shared_region_name.has_value()) {
        return region.shared_region_name;
    }
    return exchange_shared_region_name;
}

bool sharedRegionContainsParticipant(const CouplingContext& ctx,
                                     const std::string& shared_region_name,
                                     const std::string& participant_name)
{
    if (!ctx.hasSharedRegion(shared_region_name)) {
        return false;
    }
    const auto group = ctx.sharedRegionGroup(shared_region_name);
    return std::any_of(group.participant_regions.begin(),
                       group.participant_regions.end(),
                       [&participant_name](const CouplingRegionRef& region) {
                           return region.participant_name == participant_name;
                       });
}

CouplingValidationResult validateRegionEndpointScope(
    const CouplingContext& ctx,
    const std::optional<std::string>& exchange_shared_region_name,
    const std::optional<CouplingRegionEndpointDeclaration>& region,
    std::string_view role)
{
    CouplingValidationResult result;
    if (!region.has_value()) {
        return result;
    }
    if (region->participant_name.empty()) {
        result.addError("partitioned " + std::string(role) +
                        " region endpoint requires a participant");
    }
    if (region->region_name.empty()) {
        result.addError("partitioned " + std::string(role) +
                        " region endpoint requires a region name");
    }
    if (exchange_shared_region_name.has_value() &&
        region->shared_region_name.has_value() &&
        *exchange_shared_region_name != *region->shared_region_name) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .participant_name = region->participant_name,
            .region_name = region->region_name,
            .message = "partitioned " + std::string(role) +
                       " region endpoint conflicts with the exchange shared region",
        });
    }

    const auto shared_region_name =
        effectiveSharedRegionName(exchange_shared_region_name, *region);
    if (shared_region_name.has_value()) {
        if (!ctx.hasSharedRegion(*shared_region_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = region->participant_name,
                .region_name = *shared_region_name,
                .message = "partitioned " + std::string(role) +
                           " region endpoint shared region is missing from the context",
            });
            return result;
        }
        if (!sharedRegionContainsParticipant(
                ctx, *shared_region_name, region->participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = region->participant_name,
                .region_name = *shared_region_name,
                .message = "partitioned " + std::string(role) +
                           " region endpoint shared region has no participant mapping",
            });
            return result;
        }
        const auto resolved = ctx.sharedRegion(*shared_region_name, region->participant_name);
        if (!region->region_name.empty() &&
            resolved.region_name != region->region_name) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = region->participant_name,
                .region_name = region->region_name,
                .message = "partitioned " + std::string(role) +
                           " region endpoint does not match the shared-region mapping",
            });
        }
        return result;
    }

    if (!region->participant_name.empty() &&
        !region->region_name.empty() &&
        !ctx.hasRegion(region->participant_name, region->region_name)) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .participant_name = region->participant_name,
            .region_name = region->region_name,
            .message = "partitioned " + std::string(role) +
                       " region endpoint is missing from the context",
        });
    }
    return result;
}

bool hasPort(const std::vector<CouplingPortId>& stack, const CouplingPortId& port)
{
    return std::find(stack.begin(), stack.end(), port) != stack.end();
}

ResolvedCouplingEndpoint resolveEndpoint(const CouplingContext& ctx,
                                         const CouplingEndpointRef& endpoint,
                                         const CouplingValueDescriptor& value)
{
    ResolvedCouplingEndpoint resolved;
    resolved.declaration_provenance = endpoint;
    resolved.resolved_kind = endpoint.kind;
    resolved.value = value;
    resolved.resolved_participant_name = endpoint.participant_name;
    resolved.resolved_endpoint_key = endpoint.endpoint_name;
    resolved.temporal.request = endpoint.temporal;
    resolved.temporal.provided = endpoint.temporal;
    resolved.temporal.backing = backingForEndpoint(endpoint);
    resolved.temporal.storage_index = resolvedTemporalStorageIndex(endpoint.temporal);

    if (endpoint.kind == CouplingEndpointKind::Field &&
        endpoint.participant_name.has_value()) {
        const auto field = ctx.field(*endpoint.participant_name, endpoint.endpoint_name);
        resolved.system_name = field.system_name;
        resolved.system = field.system;
        resolved.registry_provider = "CouplingContext";
        resolved.temporal.provider_name = "FESystem";
        resolved.field_id = field.field_id;
    }
    if (endpoint.kind == CouplingEndpointKind::Parameter &&
        endpoint.participant_name.has_value()) {
        const auto participant = ctx.participant(*endpoint.participant_name);
        resolved.system_name = participant.system_name;
        resolved.system = participant.system;
        resolved.registry_provider = "ParameterRegistry";
        resolved.temporal.provider_name = "ParameterRegistry";
        const auto& registry = participant.system->parameterRegistry();
        if (const auto* spec = registry.find(endpoint.endpoint_name)) {
            resolved.parameter_value_type = spec->type;
        }
        resolved.parameter_slot = registry.slotOf(endpoint.endpoint_name);
    }
    if (endpoint.kind == CouplingEndpointKind::RegionData &&
        endpoint.participant_name.has_value()) {
        const auto participant = ctx.participant(*endpoint.participant_name);
        resolved.system_name = participant.system_name;
        resolved.system = participant.system;
        resolved.registry_provider = "FEQuantityRegistry";
        resolved.temporal.provider_name = "FEQuantityRegistry";
        resolved.region_data_provider_kind = CouplingRegionDataProviderKind::FEQuantity;
        resolved.region_data_provider_name = endpoint.endpoint_name;
        if (const auto* registry = participant.system->feQuantityRegistryIfPresent()) {
            resolved.fe_quantity_id = registry->idOf(endpoint.endpoint_name);
        }
    }
    if (endpoint.kind == CouplingEndpointKind::AuxiliaryInput &&
        endpoint.participant_name.has_value()) {
        const auto participant = ctx.participant(*endpoint.participant_name);
        resolved.system_name = participant.system_name;
        resolved.system = participant.system;
        resolved.registry_provider = "AuxiliaryInputRegistry";
        resolved.temporal.provider_name = "AuxiliaryInputRegistry";
        resolved.auxiliary_kind = CouplingAuxiliaryEndpointResolutionKind::InputSlot;
        resolved.auxiliary_key = endpoint.endpoint_name;
        if (const auto* registry = participant.system->auxiliaryInputRegistryIfPresent()) {
            const auto slot = registry->slotOf(endpoint.endpoint_name);
            if (fitsUint32(slot)) {
                resolved.auxiliary_input_slot = static_cast<std::uint32_t>(slot);
            }
        }
    }
    if (endpoint.kind == CouplingEndpointKind::ExternalBuffer) {
        if (const auto* descriptor =
                ctx.externalBufferDescriptor(endpointScope(endpoint), endpoint.endpoint_name)) {
            resolved.registry_provider = "CouplingContext";
            resolved.external_buffer = *descriptor;
            resolved.temporal.provider_name = "ExternalBuffer";
            resolved.layout_revision_key = descriptor->layout_revision_key;
            resolved.registry_revision_key = descriptor->data_revision_key;
        }
    }

    return resolved;
}

std::optional<CouplingRegionRef> resolveRegion(
    const CouplingContext& ctx,
    const std::optional<std::string>& exchange_shared_region_name,
    const std::optional<CouplingRegionEndpointDeclaration>& region)
{
    if (!region.has_value()) {
        return std::nullopt;
    }
    const auto shared_region_name =
        effectiveSharedRegionName(exchange_shared_region_name, *region);
    if (shared_region_name.has_value()) {
        return ctx.sharedRegion(*shared_region_name, region->participant_name);
    }
    return ctx.region(region->participant_name, region->region_name);
}

ResolvedCouplingTransfer resolveTransfer(const CouplingContext& ctx,
                                         const CouplingExchangeDeclaration& exchange)
{
    ResolvedCouplingTransfer resolved;
    const auto& transfer = exchange.transfer;
    resolved.kind = transfer.kind;
    if (transfer.interface_declaration.has_value()) {
        resolved.source_embedding_policy =
            transfer.interface_declaration->source_embedding_policy;
        resolved.target_restriction_policy =
            transfer.interface_declaration->target_restriction_policy;
    }
    resolved.driver_owned_name = transfer.driver_owned_name;
    if (transfer.kind == CouplingTransferKind::DriverOwned &&
        !transfer.driver_owned_name.empty()) {
        if (const auto* descriptor = ctx.driverOwnedTransfer(transfer.driver_owned_name)) {
            resolved.driver_owned_descriptor = *descriptor;
        }
    }
    return resolved;
}

void detectCyclesFrom(const std::vector<CouplingExchange>& exchanges,
                      const CouplingPortId& start,
                      const CouplingPortId& current,
                      std::vector<CouplingPortId>& stack,
                      std::set<std::vector<CouplingPortId>>& unique_cycles)
{
    stack.push_back(current);
    for (const auto& exchange : exchanges) {
        if (exchange.producer_port != current) {
            continue;
        }
        const auto& next = exchange.consumer_port;
        if (next == start) {
            auto cycle = stack;
            cycle.push_back(start);
            unique_cycles.insert(std::move(cycle));
            continue;
        }
        if (!hasPort(stack, next)) {
            detectCyclesFrom(exchanges, start, next, stack, unique_cycles);
        }
    }
    stack.pop_back();
}

std::vector<CouplingExchangeCycle> detectCycles(const std::vector<CouplingExchange>& exchanges)
{
    std::set<std::vector<CouplingPortId>> unique_cycles;
    for (const auto& exchange : exchanges) {
        std::vector<CouplingPortId> stack;
        detectCyclesFrom(exchanges,
                         exchange.producer_port,
                         exchange.producer_port,
                         stack,
                         unique_cycles);
    }

    std::vector<CouplingExchangeCycle> cycles;
    cycles.reserve(unique_cycles.size());
    for (auto& cycle : unique_cycles) {
        cycles.push_back(CouplingExchangeCycle{.ports = std::move(cycle)});
    }
    return cycles;
}

void collectDeclarationPartitionedInputs(
    std::span<const CouplingContractDeclaration> declarations,
    std::vector<CouplingExchangeDeclaration>& exchanges,
    std::vector<CouplingGroupHint>& group_hints)
{
    for (const auto& declaration : declarations) {
        exchanges.insert(exchanges.end(),
                         declaration.partitioned_exchange_declarations.begin(),
                         declaration.partitioned_exchange_declarations.end());
        group_hints.insert(group_hints.end(),
                           declaration.group_hints.begin(),
                           declaration.group_hints.end());
    }
}

} // namespace

CouplingValidationResult PartitionedCouplingPlanGenerator::validate(
    const CouplingContext& ctx,
    std::span<const CouplingExchangeDeclaration> exchanges) const
{
    return validate(ctx, exchanges, std::span<const CouplingGroupHint>{});
}

CouplingValidationResult PartitionedCouplingPlanGenerator::validate(
    const CouplingContext& ctx,
    std::span<const CouplingExchangeDeclaration> exchanges,
    std::span<const CouplingGroupHint> group_hints) const
{
    CouplingValidationResult result;
    for (const auto& exchange : exchanges) {
        result.append(validateCouplingPortId(exchange.producer_port));
        result.append(validateCouplingPortId(exchange.consumer_port));
        result.append(validateCouplingValueDescriptor(exchange.value));
        if (exchange.transfer.kind == CouplingTransferKind::Unspecified) {
            result.addError("partitioned coupling exchange requires an explicit transfer");
        }
        if (exchange.value.rank == CouplingValueRank::GeneralTensor &&
            exchange.transfer.kind != CouplingTransferKind::DriverOwned) {
            result.addError("general tensor partitioned values require driver-owned transfers");
        }
        if (exchange.transfer.kind == CouplingTransferKind::DriverOwned &&
            exchange.transfer.driver_owned_name.empty()) {
            result.addError("driver-owned partitioned transfer requires a transfer name");
        }
        if (isInterfaceTransferKind(exchange.transfer.kind) &&
            !exchange.transfer.interface_declaration.has_value()) {
            result.addError("interface partitioned transfer requires interface transfer metadata");
        }
        result.append(validateInterfaceTransferShape(exchange));
        if (exchange.producer_port == exchange.consumer_port) {
            result.addError("partitioned coupling exchange cannot connect a port to itself");
        }
        if (!exchange.producer.has_value() || !exchange.consumer.has_value()) {
            result.addError("partitioned coupling exchange requires producer and consumer endpoints");
            continue;
        }
        result.append(validateDriverOwnedTransferDescriptor(ctx, exchange));
        result.append(validateCouplingEndpointRef(*exchange.producer));
        result.append(validateCouplingEndpointRef(*exchange.consumer));
        result.append(validateEndpointResolutionSupport(
            ctx, *exchange.producer, exchange.value, "producer"));
        result.append(validateEndpointResolutionSupport(
            ctx, *exchange.consumer, exchange.value, "consumer"));
        result.append(validateFieldEndpointValueDescriptor(
            ctx, *exchange.producer, exchange.value, "producer"));
        result.append(validateFieldEndpointValueDescriptor(
            ctx, *exchange.consumer, exchange.value, "consumer"));

        if (exchange.producer->kind == CouplingEndpointKind::Field) {
            if (!exchange.producer->participant_name.has_value() ||
                !ctx.hasField(*exchange.producer->participant_name,
                              exchange.producer->endpoint_name)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name =
                        exchange.producer->participant_name.value_or(""),
                    .endpoint_name = exchange.producer->endpoint_name,
                    .message = "partitioned producer field endpoint is missing from the context",
                });
            }
        }
        if (exchange.consumer->kind == CouplingEndpointKind::Field) {
            if (!exchange.consumer->participant_name.has_value() ||
                !ctx.hasField(*exchange.consumer->participant_name,
                              exchange.consumer->endpoint_name)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name =
                        exchange.consumer->participant_name.value_or(""),
                    .endpoint_name = exchange.consumer->endpoint_name,
                    .message = "partitioned consumer field endpoint is missing from the context",
                });
            }
        }
        if (exchange.shared_region_name.has_value() &&
            !ctx.hasSharedRegion(*exchange.shared_region_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .region_name = *exchange.shared_region_name,
                .message = "partitioned coupling exchange shared region is missing from the context",
            });
        }
        result.append(validateRegionEndpointScope(
            ctx, exchange.shared_region_name, exchange.producer_region, "producer"));
        result.append(validateRegionEndpointScope(
            ctx, exchange.shared_region_name, exchange.consumer_region, "consumer"));
    }

    for (std::size_t hint_index = 0; hint_index < group_hints.size(); ++hint_index) {
        const auto& hint = group_hints[hint_index];
        if (hint.name.empty()) {
            result.addError("partitioned coupling group hint requires a name");
        }
        for (std::size_t other_index = hint_index + 1u;
             other_index < group_hints.size();
             ++other_index) {
            const auto& other = group_hints[other_index];
            if (hint.name.empty() || hint.name != other.name) {
                continue;
            }
            if (hint.participant_names != other.participant_names) {
                result.addError(
                    "partitioned coupling group hint duplicates a name with different participants");
                break;
            }
        }
        for (std::size_t i = 0; i < hint.participant_names.size(); ++i) {
            const auto& participant = hint.participant_names[i];
            if (participant.empty()) {
                result.addError("partitioned coupling group hint requires nonempty participants");
                continue;
            }
            if (!ctx.hasParticipant(participant)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = participant,
                    .message = "partitioned coupling group hint references an unknown participant",
                });
            }
            for (std::size_t j = i + 1u; j < hint.participant_names.size(); ++j) {
                if (participant == hint.participant_names[j]) {
                    result.add(CouplingDiagnostic{
                        .severity = CouplingDiagnosticSeverity::Error,
                        .participant_name = participant,
                        .message = "partitioned coupling group hint contains a duplicate participant",
                    });
                }
            }
        }
    }
    return result;
}

CouplingValidationResult PartitionedCouplingPlanGenerator::validate(
    const CouplingContext& ctx,
    std::span<const CouplingContractDeclaration> declarations) const
{
    std::vector<CouplingExchangeDeclaration> exchanges;
    std::vector<CouplingGroupHint> group_hints;
    collectDeclarationPartitionedInputs(declarations, exchanges, group_hints);
    return validate(ctx,
                    std::span<const CouplingExchangeDeclaration>(exchanges),
                    std::span<const CouplingGroupHint>(group_hints));
}

CouplingValidationResult PartitionedCouplingPlanGenerator::validate(
    const CouplingContext& ctx,
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingExchangeDeclaration> exchange_templates) const
{
    std::vector<CouplingExchangeDeclaration> exchanges;
    std::vector<CouplingGroupHint> group_hints;
    collectDeclarationPartitionedInputs(declarations, exchanges, group_hints);
    exchanges.insert(exchanges.end(), exchange_templates.begin(), exchange_templates.end());
    return validate(ctx,
                    std::span<const CouplingExchangeDeclaration>(exchanges),
                    std::span<const CouplingGroupHint>(group_hints));
}

PartitionedCouplingPlan PartitionedCouplingPlanGenerator::generate(
    const CouplingContext& ctx,
    std::span<const CouplingExchangeDeclaration> exchanges) const
{
    return generate(ctx, exchanges, std::span<const CouplingGroupHint>{});
}

PartitionedCouplingPlan PartitionedCouplingPlanGenerator::generate(
    const CouplingContext& ctx,
    std::span<const CouplingExchangeDeclaration> exchanges,
    std::span<const CouplingGroupHint> group_hints) const
{
    const auto validation = validate(ctx, exchanges, group_hints);
    throwIfInvalid(validation);

    PartitionedCouplingPlan plan;
    plan.exchanges.reserve(exchanges.size());
    for (const auto& exchange : exchanges) {
        CouplingExchange resolved;
        resolved.producer_port = exchange.producer_port;
        resolved.consumer_port = exchange.consumer_port;
        resolved.value = exchange.value;
        resolved.producer = resolveEndpoint(ctx, *exchange.producer, exchange.value);
        resolved.consumer = resolveEndpoint(ctx, *exchange.consumer, exchange.value);
        resolved.shared_region_name = exchange.shared_region_name;
        resolved.producer_region = resolveRegion(
            ctx, exchange.shared_region_name, exchange.producer_region);
        resolved.consumer_region = resolveRegion(
            ctx, exchange.shared_region_name, exchange.consumer_region);
        resolved.transfer = resolveTransfer(ctx, exchange);
        plan.exchanges.push_back(std::move(resolved));
    }
    plan.group_hints.assign(group_hints.begin(), group_hints.end());
    plan.cycles = detectCycles(plan.exchanges);
    return plan;
}

PartitionedCouplingPlan PartitionedCouplingPlanGenerator::generate(
    const CouplingContext& ctx,
    std::span<const CouplingContractDeclaration> declarations) const
{
    std::vector<CouplingExchangeDeclaration> exchanges;
    std::vector<CouplingGroupHint> group_hints;
    collectDeclarationPartitionedInputs(declarations, exchanges, group_hints);
    return generate(ctx,
                    std::span<const CouplingExchangeDeclaration>(exchanges),
                    std::span<const CouplingGroupHint>(group_hints));
}

PartitionedCouplingPlan PartitionedCouplingPlanGenerator::generate(
    const CouplingContext& ctx,
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingExchangeDeclaration> exchange_templates) const
{
    std::vector<CouplingExchangeDeclaration> exchanges;
    std::vector<CouplingGroupHint> group_hints;
    collectDeclarationPartitionedInputs(declarations, exchanges, group_hints);
    exchanges.insert(exchanges.end(), exchange_templates.begin(), exchange_templates.end());
    return generate(ctx,
                    std::span<const CouplingExchangeDeclaration>(exchanges),
                    std::span<const CouplingGroupHint>(group_hints));
}

} // namespace coupling
} // namespace FE
} // namespace svmp

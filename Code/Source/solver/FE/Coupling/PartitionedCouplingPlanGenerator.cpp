/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/PartitionedCouplingPlanGenerator.h"

#include "Coupling/CouplingDeclaration.h"

#include <algorithm>
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

CouplingValidationResult validateEndpointResolutionSupport(
    const CouplingEndpointRef& endpoint,
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
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .participant_name = endpoint.participant_name.value_or(""),
            .endpoint_name = endpoint.endpoint_name,
            .message = "partitioned " + std::string(role) +
                       " external buffer endpoint requires a registered descriptor",
        });
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
        resolved.registry_provider = "CouplingContext";
        resolved.temporal.provider_name = "FESystem";
        resolved.field_id = field.field_id;
    }

    return resolved;
}

std::optional<CouplingRegionRef> resolveRegion(
    const CouplingContext& ctx,
    const std::optional<CouplingRegionEndpointDeclaration>& region)
{
    if (!region.has_value()) {
        return std::nullopt;
    }
    if (region->shared_region_name.has_value()) {
        return ctx.sharedRegion(*region->shared_region_name, region->participant_name);
    }
    return ctx.region(region->participant_name, region->region_name);
}

ResolvedCouplingTransfer resolveTransfer(const CouplingTransferDeclaration& transfer)
{
    ResolvedCouplingTransfer resolved;
    resolved.kind = transfer.kind;
    if (transfer.interface_declaration.has_value()) {
        resolved.source_embedding_policy =
            transfer.interface_declaration->source_embedding_policy;
        resolved.target_restriction_policy =
            transfer.interface_declaration->target_restriction_policy;
    }
    resolved.driver_owned_name = transfer.driver_owned_name;
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
        if (exchange.producer_port == exchange.consumer_port) {
            result.addError("partitioned coupling exchange cannot connect a port to itself");
        }
        if (!exchange.producer.has_value() || !exchange.consumer.has_value()) {
            result.addError("partitioned coupling exchange requires producer and consumer endpoints");
            continue;
        }
        result.append(validateCouplingEndpointRef(*exchange.producer));
        result.append(validateCouplingEndpointRef(*exchange.consumer));
        result.append(validateEndpointResolutionSupport(*exchange.producer, "producer"));
        result.append(validateEndpointResolutionSupport(*exchange.consumer, "consumer"));
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
    }

    for (const auto& hint : group_hints) {
        if (hint.name.empty()) {
            result.addError("partitioned coupling group hint requires a name");
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
        resolved.producer_region = resolveRegion(ctx, exchange.producer_region);
        resolved.consumer_region = resolveRegion(ctx, exchange.consumer_region);
        resolved.transfer = resolveTransfer(exchange.transfer);
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

} // namespace coupling
} // namespace FE
} // namespace svmp

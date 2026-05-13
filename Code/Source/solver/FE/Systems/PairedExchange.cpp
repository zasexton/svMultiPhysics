/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/PairedExchange.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

PairedExchangeEndpoint PairedExchangeEndpoint::fieldEndpoint(FieldId field_id) noexcept
{
    PairedExchangeEndpoint endpoint;
    endpoint.kind = PairedExchangeEndpointKind::Field;
    endpoint.field = field_id;
    return endpoint;
}

PairedExchangeEndpoint PairedExchangeEndpoint::stateGroupEndpoint(StateGroupId group_id) noexcept
{
    PairedExchangeEndpoint endpoint;
    endpoint.kind = PairedExchangeEndpointKind::StateGroup;
    endpoint.state_group = group_id;
    return endpoint;
}

PairedExchangeEndpoint PairedExchangeEndpoint::auxiliaryEndpoint(std::string auxiliary_name)
{
    PairedExchangeEndpoint endpoint;
    endpoint.kind = PairedExchangeEndpointKind::AuxiliaryState;
    endpoint.name = std::move(auxiliary_name);
    return endpoint;
}

namespace {

[[nodiscard]] bool validSign(int sign) noexcept
{
    return sign == -1 || sign == 1;
}

[[nodiscard]] Real weightedSum(std::span<const Real> values, Real weight)
{
    return weight * std::accumulate(values.begin(), values.end(), Real{0.0});
}

} // namespace

void PairedExchange::validateDescriptor(const PairedExchangeDescriptor& descriptor)
{
    FE_THROW_IF(descriptor.id.empty(), InvalidArgumentException,
                "PairedExchange: descriptor id is empty");
    FE_THROW_IF(descriptor.exchanged_quantity_name.empty(), InvalidArgumentException,
                "PairedExchange: descriptor '" + descriptor.id +
                    "' has an empty exchanged quantity name");
    FE_THROW_IF(!validSign(descriptor.donor_sign) || !validSign(descriptor.receiver_sign),
                InvalidArgumentException,
                "PairedExchange: descriptor '" + descriptor.id +
                    "' signs must be -1 or +1");
    FE_THROW_IF(descriptor.donor_sign == descriptor.receiver_sign, InvalidArgumentException,
                "PairedExchange: descriptor '" + descriptor.id +
                    "' donor and receiver signs must oppose");
    FE_THROW_IF(descriptor.conservation_tolerance < 0.0, InvalidArgumentException,
                "PairedExchange: descriptor '" + descriptor.id +
                    "' has a negative conservation tolerance");

    auto validate_endpoint = [&](const PairedExchangeEndpoint& endpoint,
                                 const char* label) {
        switch (endpoint.kind) {
            case PairedExchangeEndpointKind::Field:
                FE_THROW_IF(endpoint.field == INVALID_FIELD_ID, InvalidArgumentException,
                            "PairedExchange: descriptor '" + descriptor.id +
                                "' has invalid field endpoint for " + label);
                break;
            case PairedExchangeEndpointKind::StateGroup:
                FE_THROW_IF(endpoint.state_group == INVALID_STATE_GROUP_ID,
                            InvalidArgumentException,
                            "PairedExchange: descriptor '" + descriptor.id +
                                "' has invalid state-group endpoint for " + label);
                break;
            case PairedExchangeEndpointKind::AuxiliaryState:
                FE_THROW_IF(endpoint.name.empty(), InvalidArgumentException,
                            "PairedExchange: descriptor '" + descriptor.id +
                                "' has empty auxiliary endpoint name for " + label);
                break;
        }
    };
    validate_endpoint(descriptor.donor, "donor");
    validate_endpoint(descriptor.receiver, "receiver");
}

void PairedExchange::applyEqualAndOpposite(const PairedExchangeDescriptor& descriptor,
                                           std::span<const Real> exchange,
                                           std::span<Real> donor_residual,
                                           std::span<Real> receiver_residual)
{
    validateDescriptor(descriptor);
    FE_THROW_IF(exchange.size() != donor_residual.size() ||
                    exchange.size() != receiver_residual.size(),
                InvalidArgumentException,
                "PairedExchange: exchange and residual spans must have matching sizes");

    for (std::size_t i = 0; i < exchange.size(); ++i) {
        donor_residual[i] += static_cast<Real>(descriptor.donor_sign) *
                             descriptor.donor_weight * exchange[i];
        receiver_residual[i] += static_cast<Real>(descriptor.receiver_sign) *
                                descriptor.receiver_weight * exchange[i];
    }
}

PairedExchangeBalanceResult PairedExchange::checkBalance(
    const PairedExchangeDescriptor& descriptor,
    std::span<const Real> donor_residual,
    std::span<const Real> receiver_residual)
{
    validateDescriptor(descriptor);
    FE_THROW_IF(donor_residual.size() != receiver_residual.size(), InvalidArgumentException,
                "PairedExchange: donor and receiver residual spans must have matching sizes");

    PairedExchangeBalanceResult result;
    for (std::size_t i = 0; i < donor_residual.size(); ++i) {
        const Real local = descriptor.donor_weight * donor_residual[i] +
                           descriptor.receiver_weight * receiver_residual[i];
        const Real residual = std::abs(local);
        result.local_residual_norm = std::max(result.local_residual_norm, residual);
        if (residual > descriptor.conservation_tolerance) {
            ++result.local_violation_count;
        }
    }
    result.global_residual_norm = std::abs(weightedSum(donor_residual, descriptor.donor_weight) +
                                           weightedSum(receiver_residual, descriptor.receiver_weight));
    result.conserved = result.local_violation_count == 0 &&
                       result.global_residual_norm <= descriptor.conservation_tolerance;
    return result;
}

std::vector<analysis::VariableKey> PairedExchange::endpointVariables(
    const PairedExchangeEndpoint& endpoint,
    const FieldRegistry* registry)
{
    switch (endpoint.kind) {
        case PairedExchangeEndpointKind::Field:
            return {analysis::VariableKey::field(endpoint.field)};
        case PairedExchangeEndpointKind::StateGroup: {
            FE_THROW_IF(registry == nullptr ||
                            !registry->hasStateGroup(endpoint.state_group),
                        InvalidArgumentException,
                        "PairedExchange: state-group endpoint requires a registry");
            const auto& group = registry->getStateGroup(endpoint.state_group);
            std::vector<analysis::VariableKey> variables;
            variables.reserve(group.fields.size());
            for (const FieldId field : group.fields) {
                variables.push_back(analysis::VariableKey::field(field));
            }
            return variables;
        }
        case PairedExchangeEndpointKind::AuxiliaryState:
            return {analysis::VariableKey::named(analysis::VariableKind::AuxiliaryState,
                                                 endpoint.name)};
    }
    return {};
}

std::vector<analysis::ContributionDescriptor> PairedExchange::makeContributions(
    const PairedExchangeDescriptor& descriptor,
    const FieldRegistry* registry)
{
    validateDescriptor(descriptor);
    const auto donor_variables = endpointVariables(descriptor.donor, registry);
    const auto receiver_variables = endpointVariables(descriptor.receiver, registry);

    std::vector<analysis::ContributionDescriptor> contributions;
    contributions.reserve(donor_variables.size() + receiver_variables.size());

    auto add_contribution = [&](analysis::VariableKey variable, int sign,
                                std::string suffix) {
        analysis::ContributionDescriptor contribution;
        const std::string& id_base = descriptor.source_contribution_id.empty()
                                         ? descriptor.id
                                         : descriptor.source_contribution_id;
        contribution.contribution_id = id_base + "_" + std::move(suffix);
        contribution.operator_tag = descriptor.operator_tag;
        contribution.origin = descriptor.origin;
        contribution.domain = analysis::DomainKind::Cell;
        contribution.related_variables = {variable};
        contribution.role = analysis::ContributionRole::SourceVector;
        contribution.traits = analysis::OperatorTraitFlags::SourceLike;
        contribution.balance = analysis::BalanceDescriptor{
            descriptor.balance_group,
            analysis::BalanceRole::ExchangeLike,
            sign,
            descriptor.local_closure_expected};
        contribution.ensureStableContributionId();
        contributions.push_back(std::move(contribution));
    };

    for (std::size_t i = 0; i < donor_variables.size(); ++i) {
        add_contribution(donor_variables[i], descriptor.donor_sign,
                         "donor_" + std::to_string(i));
    }
    for (std::size_t i = 0; i < receiver_variables.size(); ++i) {
        add_contribution(receiver_variables[i], descriptor.receiver_sign,
                         "receiver_" + std::to_string(i));
    }

    return contributions;
}

analysis::FluxBalanceSummary PairedExchange::makeFluxBalanceSummary(
    const PairedExchangeDescriptor& descriptor,
    const PairedExchangeBalanceResult& result)
{
    validateDescriptor(descriptor);

    analysis::FluxBalanceSummary summary;
    summary.balance_group = descriptor.balance_group;
    summary.balance_tolerance = descriptor.conservation_tolerance;
    summary.local_residual_norm = result.local_residual_norm;
    summary.global_residual_norm = result.global_residual_norm;
    summary.local_violation_count = result.local_violation_count;
    summary.interface_pair_count = 1;
    summary.symbolic_balance_evidence_present = !descriptor.balance_group.empty();
    summary.symbolic_balance_group = descriptor.balance_group;
    summary.source_quadrature_consistency_present = true;
    summary.steady_balance_scope = true;
    return summary;
}

} // namespace systems
} // namespace FE
} // namespace svmp

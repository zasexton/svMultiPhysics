/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_PAIREDEXCHANGE_H
#define SVMP_FE_SYSTEMS_PAIREDEXCHANGE_H

#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Core/Types.h"
#include "Systems/FieldRegistry.h"

#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

enum class PairedExchangeEndpointKind : std::uint8_t {
    Field,
    StateGroup,
    AuxiliaryState
};

struct PairedExchangeEndpoint {
    PairedExchangeEndpointKind kind{PairedExchangeEndpointKind::Field};
    FieldId field{INVALID_FIELD_ID};
    StateGroupId state_group{INVALID_STATE_GROUP_ID};
    std::string name;

    static PairedExchangeEndpoint fieldEndpoint(FieldId field_id) noexcept;
    static PairedExchangeEndpoint stateGroupEndpoint(StateGroupId group_id) noexcept;
    static PairedExchangeEndpoint auxiliaryEndpoint(std::string auxiliary_name);
};

struct PairedExchangeDescriptor {
    std::string id;
    std::string balance_group;
    std::string exchanged_quantity_name;
    std::string source_contribution_id;
    PairedExchangeEndpoint donor;
    PairedExchangeEndpoint receiver;
    int donor_sign{-1};
    int receiver_sign{1};
    Real donor_weight{1.0};
    Real receiver_weight{1.0};
    Real conservation_tolerance{0.0};
    bool local_closure_expected{true};
    std::string operator_tag{"paired_exchange"};
    std::string origin{"PairedExchange"};
};

struct PairedExchangeBalanceResult {
    Real local_residual_norm{0.0};
    Real global_residual_norm{0.0};
    std::uint64_t local_violation_count{0};
    bool conserved{true};
};

class PairedExchange {
public:
    static void applyEqualAndOpposite(const PairedExchangeDescriptor& descriptor,
                                      std::span<const Real> exchange,
                                      std::span<Real> donor_residual,
                                      std::span<Real> receiver_residual);

    static PairedExchangeBalanceResult checkBalance(
        const PairedExchangeDescriptor& descriptor,
        std::span<const Real> donor_residual,
        std::span<const Real> receiver_residual);

    static std::vector<analysis::ContributionDescriptor> makeContributions(
        const PairedExchangeDescriptor& descriptor,
        const FieldRegistry* registry = nullptr);

    static analysis::FluxBalanceSummary makeFluxBalanceSummary(
        const PairedExchangeDescriptor& descriptor,
        const PairedExchangeBalanceResult& result);

private:
    static void validateDescriptor(const PairedExchangeDescriptor& descriptor);
    static std::vector<analysis::VariableKey> endpointVariables(
        const PairedExchangeEndpoint& endpoint,
        const FieldRegistry* registry);
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_PAIREDEXCHANGE_H

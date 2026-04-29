/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_PARTITIONEDCOUPLINGPLANGENERATOR_H
#define SVMP_FE_COUPLING_PARTITIONEDCOUPLINGPLANGENERATOR_H

/**
 * @file PartitionedCouplingPlanGenerator.h
 * @brief Resolver interface for metadata-only partitioned exchange plans.
 */

#include "Coupling/CouplingDiagnostics.h"
#include "Coupling/PartitionedCouplingPlan.h"

#include <span>

namespace svmp {
namespace FE {
namespace coupling {

struct CouplingContractDeclaration;

class PartitionedCouplingPlanGenerator {
public:
    [[nodiscard]] CouplingValidationResult validate(
        const CouplingContext& ctx,
        std::span<const CouplingExchangeDeclaration> exchanges) const;

    [[nodiscard]] CouplingValidationResult validate(
        const CouplingContext& ctx,
        std::span<const CouplingExchangeDeclaration> exchanges,
        std::span<const CouplingGroupHint> group_hints) const;

    [[nodiscard]] CouplingValidationResult validate(
        const CouplingContext& ctx,
        std::span<const CouplingContractDeclaration> declarations) const;

    [[nodiscard]] PartitionedCouplingPlan generate(
        const CouplingContext& ctx,
        std::span<const CouplingExchangeDeclaration> exchanges) const;

    [[nodiscard]] PartitionedCouplingPlan generate(
        const CouplingContext& ctx,
        std::span<const CouplingExchangeDeclaration> exchanges,
        std::span<const CouplingGroupHint> group_hints) const;

    [[nodiscard]] PartitionedCouplingPlan generate(
        const CouplingContext& ctx,
        std::span<const CouplingContractDeclaration> declarations) const;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_PARTITIONEDCOUPLINGPLANGENERATOR_H

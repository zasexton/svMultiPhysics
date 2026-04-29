/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/PartitionedCouplingPlanGenerator.h"

namespace svmp {
namespace FE {
namespace coupling {

CouplingValidationResult PartitionedCouplingPlanGenerator::validate(
    const CouplingContext&,
    std::span<const CouplingExchangeDeclaration>) const
{
    return {};
}

PartitionedCouplingPlan PartitionedCouplingPlanGenerator::generate(
    const CouplingContext&,
    std::span<const CouplingExchangeDeclaration>) const
{
    return {};
}

} // namespace coupling
} // namespace FE
} // namespace svmp

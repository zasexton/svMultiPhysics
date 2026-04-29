/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingContract.h"

namespace svmp {
namespace FE {
namespace coupling {

void CouplingContract::validate(const CouplingContext&) const {}

std::vector<CouplingFormContribution> CouplingContract::buildMonolithicForms(
    const CouplingContext&,
    const CouplingFormBuilder&) const
{
    return {};
}

std::vector<CouplingInstallMetadata> CouplingContract::installMonolithicTerms(
    MonolithicCouplingInstallContext&,
    const CouplingContext&)
{
    return {};
}

std::vector<CouplingExchangeDeclaration> CouplingContract::buildPartitionedExchangeDeclarations(
    const CouplingContext&) const
{
    return {};
}

} // namespace coupling
} // namespace FE
} // namespace svmp

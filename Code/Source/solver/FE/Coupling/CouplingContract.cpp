/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingContract.h"

#include "Coupling/CouplingGraph.h"

#include <array>
#include <span>

namespace svmp {
namespace FE {
namespace coupling {

void CouplingContract::validate(const CouplingContext& ctx) const
{
    CouplingValidationResult validation;
    auto declaration = declare();
    if (declaration.contract_type != name()) {
        validation.addError(
            "coupling contract declaration type does not match the contract registry key");
    }

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    validation.append(graph.buildDeclarationGraph(
        ctx,
        std::span<const CouplingContractDeclaration>(declarations)));
    throwIfInvalid(validation);
}

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

/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/DefinitionBackedCouplingContract.h"

namespace svmp {
namespace FE {
namespace coupling {

CouplingContractDeclaration DefinitionBackedCouplingContract::declare() const
{
    return buildDefinition().compileDeclaration();
}

void DefinitionBackedCouplingContract::validate(const CouplingContext& ctx) const
{
    CouplingContract::validate(ctx);
}

bool DefinitionBackedCouplingContract::supportsMonolithicLowering() const
{
    return buildDefinition().hasMonolithicForms();
}

bool DefinitionBackedCouplingContract::supportsPartitionedLowering() const
{
    return buildDefinition().hasPartitionedExchanges();
}

std::vector<CouplingFormContribution>
DefinitionBackedCouplingContract::buildMonolithicForms(
    const CouplingContext& ctx,
    const CouplingFormBuilder& forms) const
{
    return buildDefinition().buildMonolithicForms(ctx, forms);
}

std::vector<CouplingExchangeDeclaration>
DefinitionBackedCouplingContract::buildPartitionedExchangeDeclarations(
    const CouplingContext& ctx) const
{
    static_cast<void>(ctx);
    return buildDefinition().buildPartitionedExchangeDeclarations();
}

std::string DefinitionBackedCouplingContract::contractInstanceName() const
{
    return name();
}

CouplingDefinitionBuilder
DefinitionBackedCouplingContract::buildDefinition() const
{
    CouplingDefinitionBuilder builder(name(), contractInstanceName());
    define(builder);
    return builder;
}

} // namespace coupling
} // namespace FE
} // namespace svmp

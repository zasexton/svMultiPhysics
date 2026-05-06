/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/DefinitionBackedCouplingContract.h"

#include "Coupling/CouplingGraph.h"
#include "Coupling/CouplingPayloadDetangler.h"

#include <array>
#include <iterator>
#include <span>

namespace svmp {
namespace FE {
namespace coupling {

CouplingContractDeclaration DefinitionBackedCouplingContract::declare() const
{
    return buildDefinition().compileDeclaration();
}

void DefinitionBackedCouplingContract::validate(const CouplingContext& ctx) const
{
    CouplingValidationResult validation;
    validateDefinitionOptions(ctx, validation);

    const auto definition = buildDefinition();
    validation.append(definition.optionValidation());

    const auto declaration = definition.compileDeclaration();
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
    const auto definition = buildDefinition();
    auto declarations = definition.buildPartitionedExchangeDeclarations();
    const auto& requests = definition.payloadExtractionRequests();
    if (requests.empty()) {
        return declarations;
    }

    const CouplingFormBuilder forms(ctx);
    const auto contributions = definition.buildMonolithicForms(ctx, forms);
    const CouplingPayloadDetangler detangler;
    auto extracted = detangler.extract(ctx,
                                       contributions,
                                       requests,
                                       definition.contractName());
    CouplingValidationResult validation;
    for (const auto& diagnostic : extracted.diagnostics) {
        validation.add(diagnosticFromPayloadExtraction(diagnostic));
    }
    throwIfInvalid(validation);
    declarations.insert(declarations.end(),
                        std::make_move_iterator(extracted.exchanges.begin()),
                        std::make_move_iterator(extracted.exchanges.end()));
    return declarations;
}

std::string DefinitionBackedCouplingContract::contractInstanceName() const
{
    return name();
}

void DefinitionBackedCouplingContract::validateDefinitionOptions(
    const CouplingContext&,
    CouplingValidationResult&) const
{
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

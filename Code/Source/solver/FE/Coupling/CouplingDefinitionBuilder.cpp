/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingDefinitionBuilder.h"

#include "Core/FEException.h"

#include <iterator>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

CouplingDefinitionBuilder::CouplingDefinitionBuilder(
    std::string contract_type,
    std::string contract_name)
    : partitioned_(contract_name)
{
    FE_THROW_IF(contract_type.empty(), InvalidArgumentException,
                "coupling definition requires a contract type");
    FE_THROW_IF(contract_name.empty(), InvalidArgumentException,
                "coupling definition requires a contract name");

    declaration_.contract_type = std::move(contract_type);
    declaration_.contract_name = std::move(contract_name);
    declaration_.dependency_declaration_mode =
        CouplingDependencyDeclarationMode::InferFromInstalledForms;
}

const std::string& CouplingDefinitionBuilder::contractType() const noexcept
{
    return declaration_.contract_type;
}

const std::string& CouplingDefinitionBuilder::contractName() const noexcept
{
    return declaration_.contract_name;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::participant(
    std::string participant_name,
    CouplingRequirement requirement)
{
    declaration_.participants.push_back(CouplingParticipantUse{
        .participant_name = std::move(participant_name),
        .requirement = requirement,
    });
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::field(
    std::string participant_name,
    std::string field_name,
    CouplingRequirement requirement)
{
    declaration_.fields.push_back(CouplingFieldUse{
        .participant_name = std::move(participant_name),
        .field_name = std::move(field_name),
        .requirement = requirement,
    });
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::field(
    CouplingFieldUse field)
{
    declaration_.fields.push_back(std::move(field));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::fieldRequirement(
    CouplingFieldRequirement requirement)
{
    partitioned_.addFieldRequirement(requirement);
    declaration_.field_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::additionalField(
    CouplingAdditionalFieldDeclaration declaration)
{
    declaration_.additional_fields.push_back(std::move(declaration));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::nonFieldDependency(
    CouplingNonFieldDependencyRequirement requirement)
{
    declaration_.non_field_dependencies.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::temporalRequirement(
    CouplingTemporalRequirement requirement)
{
    declaration_.temporal_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::geometryRequirement(
    CouplingGeometryTerminalRequirement requirement)
{
    declaration_.geometry_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::dependency(
    CouplingResidualDependency dependency)
{
    declaration_.dependencies.push_back(std::move(dependency));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::expectedBlock(
    CouplingBlockExpectation expectation)
{
    declaration_.expected_blocks.push_back(std::move(expectation));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::dependencyDeclarationMode(
    CouplingDependencyDeclarationMode mode)
{
    declaration_.dependency_declaration_mode = mode;
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::region(
    CouplingRegionUse region)
{
    declaration_.regions.push_back(std::move(region));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::sharedRegion(
    CouplingSharedRegionUse region)
{
    declaration_.shared_regions.push_back(std::move(region));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::sharedInterface(
    CouplingSharedInterfaceRequirement requirement)
{
    declaration_.shared_interface_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::regionRelation(
    CouplingRegionRelationRequirement requirement)
{
    declaration_.region_relation_requirements.push_back(std::move(requirement));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::group(
    std::string name,
    std::vector<std::string> participant_names)
{
    partitioned_.group(std::move(name), std::move(participant_names));
    return *this;
}

CouplingDefinitionBuilder& CouplingDefinitionBuilder::monolithic(
    MonolithicFormsCallback callback)
{
    FE_THROW_IF(!callback, InvalidArgumentException,
                "coupling definition monolithic callback is empty");
    monolithic_callbacks_.push_back(std::move(callback));
    return *this;
}

PartitionedExchangeBuilder CouplingDefinitionBuilder::exchange(
    std::string_view name,
    const CouplingFieldUse& producer_field,
    const CouplingFieldUse& consumer_field)
{
    return partitioned_.exchange(name, producer_field, consumer_field);
}

PartitionedExchangeBuilder CouplingDefinitionBuilder::exchange(
    std::string_view name,
    CouplingEndpointRef producer_endpoint,
    CouplingEndpointRef consumer_endpoint)
{
    return partitioned_.exchange(name,
                                 std::move(producer_endpoint),
                                 std::move(consumer_endpoint));
}

bool CouplingDefinitionBuilder::hasMonolithicForms() const noexcept
{
    return !monolithic_callbacks_.empty();
}

bool CouplingDefinitionBuilder::hasPartitionedExchanges() const noexcept
{
    return !partitioned_.declarations().empty();
}

CouplingContractDeclaration CouplingDefinitionBuilder::compileDeclaration() const
{
    auto declaration = declaration_;
    const auto& exchanges = partitioned_.declarations();
    declaration.partitioned_exchange_declarations.insert(
        declaration.partitioned_exchange_declarations.end(),
        exchanges.begin(),
        exchanges.end());
    const auto& groups = partitioned_.groupHints();
    declaration.group_hints.insert(declaration.group_hints.end(),
                                   groups.begin(),
                                   groups.end());
    return declaration;
}

std::vector<CouplingFormContribution>
CouplingDefinitionBuilder::buildMonolithicForms(
    const CouplingContext& context,
    const CouplingFormBuilder& forms) const
{
    std::vector<CouplingFormContribution> contributions;
    for (const auto& callback : monolithic_callbacks_) {
        auto next = callback(context, forms);
        contributions.insert(contributions.end(),
                             std::make_move_iterator(next.begin()),
                             std::make_move_iterator(next.end()));
    }
    return contributions;
}

std::vector<CouplingExchangeDeclaration>
CouplingDefinitionBuilder::buildPartitionedExchangeDeclarations() const
{
    return partitioned_.declarations();
}

} // namespace coupling
} // namespace FE
} // namespace svmp

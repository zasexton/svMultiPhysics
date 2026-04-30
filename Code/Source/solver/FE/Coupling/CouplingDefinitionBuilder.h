/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGDEFINITIONBUILDER_H
#define SVMP_FE_COUPLING_COUPLINGDEFINITIONBUILDER_H

/**
 * @file CouplingDefinitionBuilder.h
 * @brief Thin definition facade that compiles to existing coupling records.
 */

#include "Coupling/CouplingFormBuilder.h"
#include "Coupling/PartitionedCouplingBuilder.h"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

class CouplingDefinitionBuilder {
public:
    using MonolithicFormsCallback =
        std::function<std::vector<CouplingFormContribution>(
            const CouplingContext&,
            const CouplingFormBuilder&)>;

    CouplingDefinitionBuilder(std::string contract_type,
                              std::string contract_name);

    [[nodiscard]] const std::string& contractType() const noexcept;
    [[nodiscard]] const std::string& contractName() const noexcept;

    CouplingDefinitionBuilder& participant(
        std::string participant_name,
        CouplingRequirement requirement = CouplingRequirement::Required);
    CouplingDefinitionBuilder& field(
        std::string participant_name,
        std::string field_name,
        CouplingRequirement requirement = CouplingRequirement::Required);
    CouplingDefinitionBuilder& field(CouplingFieldUse field);
    CouplingDefinitionBuilder& fieldRequirement(
        CouplingFieldRequirement requirement);
    CouplingDefinitionBuilder& additionalField(
        CouplingAdditionalFieldDeclaration declaration);
    CouplingDefinitionBuilder& nonFieldDependency(
        CouplingNonFieldDependencyRequirement requirement);
    CouplingDefinitionBuilder& temporalRequirement(
        CouplingTemporalRequirement requirement);
    CouplingDefinitionBuilder& geometryRequirement(
        CouplingGeometryTerminalRequirement requirement);
    CouplingDefinitionBuilder& dependency(
        CouplingResidualDependency dependency);
    CouplingDefinitionBuilder& expectedBlock(
        CouplingBlockExpectation expectation);
    CouplingDefinitionBuilder& dependencyDeclarationMode(
        CouplingDependencyDeclarationMode mode);
    CouplingDefinitionBuilder& region(CouplingRegionUse region);
    CouplingDefinitionBuilder& sharedRegion(CouplingSharedRegionUse region);
    CouplingDefinitionBuilder& sharedInterface(
        CouplingSharedInterfaceRequirement requirement);
    CouplingDefinitionBuilder& regionRelation(
        CouplingRegionRelationRequirement requirement);
    CouplingDefinitionBuilder& group(
        std::string name,
        std::vector<std::string> participant_names);
    CouplingDefinitionBuilder& monolithic(
        MonolithicFormsCallback callback);

    [[nodiscard]] PartitionedExchangeBuilder exchange(
        std::string_view name,
        const CouplingFieldUse& producer_field,
        const CouplingFieldUse& consumer_field);
    [[nodiscard]] PartitionedExchangeBuilder exchange(
        std::string_view name,
        CouplingEndpointRef producer_endpoint,
        CouplingEndpointRef consumer_endpoint);

    [[nodiscard]] bool hasMonolithicForms() const noexcept;
    [[nodiscard]] bool hasPartitionedExchanges() const noexcept;
    [[nodiscard]] CouplingContractDeclaration compileDeclaration() const;
    [[nodiscard]] std::vector<CouplingFormContribution> buildMonolithicForms(
        const CouplingContext& context,
        const CouplingFormBuilder& forms) const;
    [[nodiscard]] std::vector<CouplingExchangeDeclaration>
    buildPartitionedExchangeDeclarations() const;

private:
    CouplingContractDeclaration declaration_;
    PartitionedCouplingBuilder partitioned_;
    std::vector<MonolithicFormsCallback> monolithic_callbacks_;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGDEFINITIONBUILDER_H

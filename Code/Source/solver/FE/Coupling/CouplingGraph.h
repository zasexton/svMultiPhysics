/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGGRAPH_H
#define SVMP_FE_COUPLING_COUPLINGGRAPH_H

/**
 * @file CouplingGraph.h
 * @brief Declaration and finalized setup graph for coupling validation.
 */

#include "Coupling/CouplingContext.h"
#include "Coupling/CouplingDeclaration.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

[[nodiscard]] analysis::VariableKind toAnalysisVariableKind(
    CouplingVariableKind kind) noexcept;

[[nodiscard]] std::string couplingVariableUseAnalysisName(
    const CouplingContext& context,
    const CouplingVariableUse& variable);

[[nodiscard]] std::optional<analysis::VariableKey> resolveCouplingVariableUse(
    const CouplingContext& context,
    const CouplingVariableUse& variable);

struct CouplingGraphParticipantNode {
    CouplingParticipantRef participant;
};

struct CouplingGraphFieldNode {
    CouplingFieldRef field;
};

struct CouplingGraphRegionNode {
    CouplingRegionRef region;
};

struct CouplingGraphSharedRegionNode {
    SharedRegionRef shared_region;
};

struct CouplingGraphContractTypeNode {
    std::string contract_type;
};

struct CouplingGraphContractInstanceNode {
    std::string contract_type;
    std::string contract_name;
};

struct CouplingGraphAdditionalFieldNode {
    std::string contract_name;
    CouplingAdditionalFieldDeclaration declaration;
};

struct CouplingGraphNonFieldVariableNode {
    std::string contract_name;
    CouplingNonFieldDependencyRequirement requirement;
    std::optional<analysis::VariableKey> variable;
};

struct CouplingGraphProviderMetadataRequirementNode {
    std::string contract_name;
    CouplingNonFieldDependencyRequirement requirement;
};

struct CouplingGraphTemporalRequirementNode {
    std::string contract_name;
    CouplingTemporalRequirement requirement;
};

struct CouplingGraphGeometryRequirementNode {
    std::string contract_name;
    CouplingGeometryTerminalRequirement requirement;
};

struct CouplingGraphPartitionedExchangeDeclarationNode {
    std::string contract_name;
    CouplingExchangeDeclaration declaration;
};

struct CouplingGraphResolvedPartitionedExchangeNode {
    CouplingExchange exchange;
};

struct CouplingGraphDependencyExpectationNode {
    std::string contract_name;
    CouplingResidualDependency declaration;
    std::optional<analysis::VariableKey> residual_row;
    std::optional<analysis::VariableKey> dependency;
};

struct CouplingGraphExpectedBlockNode {
    std::string contract_name;
    CouplingBlockExpectation declaration;
    std::optional<analysis::VariableKey> residual_row;
    std::optional<analysis::VariableKey> dependency;
};

struct CouplingGraphSnapshot {
    std::vector<CouplingGraphParticipantNode> participants;
    std::vector<CouplingGraphFieldNode> fields;
    std::vector<CouplingGraphRegionNode> regions;
    std::vector<CouplingGraphSharedRegionNode> shared_regions;
    std::vector<CouplingGraphContractTypeNode> contract_types;
    std::vector<CouplingGraphContractInstanceNode> contract_instances;
    std::vector<CouplingGraphAdditionalFieldNode> additional_fields;
    std::vector<CouplingGraphNonFieldVariableNode> non_field_variables;
    std::vector<CouplingGraphProviderMetadataRequirementNode>
        provider_metadata_requirements;
    std::vector<CouplingGraphTemporalRequirementNode> temporal_requirements;
    std::vector<CouplingGraphGeometryRequirementNode> geometry_requirements;
    std::vector<CouplingGraphPartitionedExchangeDeclarationNode>
        partitioned_exchange_declarations;
    std::vector<CouplingGraphResolvedPartitionedExchangeNode>
        resolved_partitioned_exchanges;
    std::vector<CouplingGraphDependencyExpectationNode> dependency_expectations;
    std::vector<CouplingGraphExpectedBlockNode> expected_blocks;
};

class CouplingGraph {
public:
    [[nodiscard]] CouplingValidationResult buildDeclarationGraph(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations);

    [[nodiscard]] CouplingValidationResult buildFinalizedGraph(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations,
        std::span<const CouplingFormAnalysisMetadata> installed_forms);

    [[nodiscard]] CouplingValidationResult buildFinalizedGraph(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations,
        std::span<const CouplingFormAnalysisMetadata> installed_forms,
        const PartitionedCouplingPlan& partitioned_plan);

    [[nodiscard]] CouplingValidationResult buildFinalizedGraph(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations,
        std::span<const CouplingFormAnalysisMetadata> installed_forms,
        const PartitionedCouplingPlan& partitioned_plan,
        std::span<const CouplingExchangeDeclaration> exchange_templates);

    [[nodiscard]] CouplingValidationResult validateTemporalRequirements(
        const CouplingTemporalAvailability& availability) const;

    [[nodiscard]] CouplingValidationResult validateGeometryTerminalRequirements(
        const CouplingContext& context,
        const CouplingGeometryTerminalAvailability& availability) const;

    [[nodiscard]] const std::vector<CouplingContractDeclaration>& declarations() const noexcept;
    [[nodiscard]] const std::vector<CouplingFormAnalysisMetadata>&
    installedFormMetadata() const noexcept;
    [[nodiscard]] const CouplingGraphSnapshot& snapshot() const noexcept;

private:
    std::vector<CouplingContractDeclaration> declarations_{};
    std::vector<CouplingFormAnalysisMetadata> installed_forms_{};
    CouplingGraphSnapshot snapshot_{};
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGGRAPH_H

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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

struct CouplingSidePairedInterfaceRequest {
    std::string relation_name;
    std::string shared_region_name;
    std::string first_participant_name;
    std::string second_participant_name;
    CouplingMode mode{CouplingMode::Monolithic};
    std::string enforcement_strategy;
    CouplingRelationLoweringFidelity monolithic_fidelity{
        CouplingRelationLoweringFidelity::Exact};
    CouplingRelationLoweringFidelity partitioned_fidelity{
        CouplingRelationLoweringFidelity::Lagged};
    std::vector<CouplingPartitionedSolveStrategy>
        partitioned_solve_strategies;
    bool supports_monolithic_forms{true};
    bool supports_partitioned_exchange{true};
    bool declare_shared_region{true};
    bool declare_shared_interface{true};
    bool require_all_participants{true};
    bool require_opposite_sides{true};
    bool require_common_monolithic_system_when_monolithic{true};
    bool require_registered_topology_when_monolithic{true};
    CouplingRegionKind required_region_kind{CouplingRegionKind::InterfaceFace};
};

struct CouplingContractInterfaceFieldRequest {
    std::string field_name;
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components{0};
    std::string field_namespace;
    std::optional<std::string> system_participant_name;
    std::string shared_region_name;
    CouplingRequirement requirement{CouplingRequirement::Required};
    bool enabled{true};
    CouplingLocalEliminationPolicy local_elimination_policy{
        CouplingLocalEliminationPolicy::None};
};

struct CouplingPartitionedFieldChannelRequest {
    std::string channel_name;
    std::string producer_participant_name;
    std::string producer_field_name;
    std::string consumer_participant_name;
    std::string consumer_field_name;
    std::string producer_port_name;
    std::string consumer_port_name;
    std::string shared_region_name;
    CouplingTransferDeclaration transfer{};
    CouplingPartitionedStrategyDeclaration strategy{};
    CouplingTemporalSlotDescriptor producer_temporal{};
    CouplingTemporalSlotDescriptor consumer_temporal{};
};

using CouplingMonolithicFormsCallback =
    std::function<std::vector<CouplingFormContribution>(
        const CouplingContext&,
        const CouplingFormBuilder&)>;

struct CouplingFieldRoleRequest {
    CouplingFieldUse field;
    CouplingValueDescriptor value;
    bool enabled{true};
};

struct CouplingOptionalFieldRoleRequest {
    std::optional<std::string> participant_name;
    std::optional<std::string> field_name;
    CouplingValueDescriptor value;
    CouplingDiagnostic missing_field_diagnostic;
    bool enabled{true};
};

struct CouplingPartitionedGroupRequest {
    std::string group_name;
    std::vector<std::string> participant_names;
};

struct CouplingSidePairedPDERequest {
    CouplingSidePairedInterfaceRequest interface;
    std::vector<CouplingFieldRoleRequest> required_fields;
    std::vector<CouplingOptionalFieldRoleRequest> optional_fields;
    std::optional<CouplingContractInterfaceFieldRequest> interface_field;
    std::vector<CouplingPartitionedFieldChannelRequest> partitioned_channels;
    std::optional<CouplingPartitionedGroupRequest> partitioned_group;
    bool infer_partitioned_group{true};
    bool infer_partitioned_channel_shared_regions{true};
    bool infer_partitioned_channel_ports{true};
    CouplingMonolithicFormsCallback monolithic_forms;
};

[[nodiscard]] CouplingSidePairedInterfaceRequest sidePairedInterface(
    std::string relation_name,
    std::string shared_region_name,
    std::string first_participant_name,
    std::string second_participant_name,
    CouplingMode mode,
    std::string enforcement_strategy,
    std::vector<CouplingPartitionedSolveStrategy>
        partitioned_solve_strategies = {});

[[nodiscard]] CouplingFieldRoleRequest fieldRole(
    CouplingFieldUse field,
    CouplingValueDescriptor value,
    bool enabled = true);
[[nodiscard]] CouplingFieldRoleRequest scalarFieldRole(
    std::string participant_name,
    std::string field_name,
    bool enabled = true);
[[nodiscard]] CouplingFieldRoleRequest vectorFieldRole(
    std::string participant_name,
    std::string field_name,
    int components,
    bool enabled = true);

[[nodiscard]] CouplingDiagnostic optionErrorDiagnostic(
    std::string contract_name,
    std::string message,
    std::string participant_name = {},
    std::string field_name = {});
[[nodiscard]] CouplingOptionalFieldRoleRequest optionalFieldRole(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    CouplingValueDescriptor value,
    CouplingDiagnostic missing_field_diagnostic,
    bool enabled = true);
[[nodiscard]] CouplingOptionalFieldRoleRequest optionalFieldRole(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    CouplingValueDescriptor value,
    bool enabled = true);
[[nodiscard]] CouplingOptionalFieldRoleRequest optionalFieldRole(
    std::string participant_name,
    std::optional<std::string> field_name,
    CouplingValueDescriptor value,
    bool enabled = true);
[[nodiscard]] CouplingOptionalFieldRoleRequest optionalVectorFieldRole(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    int components,
    CouplingDiagnostic missing_field_diagnostic,
    bool enabled = true);
[[nodiscard]] CouplingOptionalFieldRoleRequest optionalVectorFieldRole(
    std::optional<std::string> participant_name,
    std::optional<std::string> field_name,
    int components,
    bool enabled = true);
[[nodiscard]] CouplingOptionalFieldRoleRequest optionalVectorFieldRole(
    std::string participant_name,
    std::optional<std::string> field_name,
    int components,
    bool enabled = true);

[[nodiscard]] CouplingContractInterfaceFieldRequest contractInterfaceField(
    std::string field_name,
    std::shared_ptr<const spaces::FunctionSpace> space,
    std::string shared_region_name,
    int components = 0,
    std::string field_namespace = {},
    std::optional<std::string> system_participant_name = std::nullopt,
    CouplingRequirement requirement = CouplingRequirement::Required,
    bool enabled = true,
    CouplingLocalEliminationPolicy local_elimination_policy =
        CouplingLocalEliminationPolicy::None);

[[nodiscard]] CouplingPartitionedFieldChannelRequest partitionedFieldChannel(
    std::string channel_name,
    std::string producer_participant_name,
    std::string producer_field_name,
    std::string consumer_participant_name,
    std::string consumer_field_name,
    std::string shared_region_name,
    CouplingTransferDeclaration transfer = {},
    CouplingTemporalSlotDescriptor producer_temporal = {},
    CouplingTemporalSlotDescriptor consumer_temporal = {},
    std::string producer_port_name = {},
    std::string consumer_port_name = {},
    CouplingPartitionedStrategyDeclaration strategy = {});
[[nodiscard]] CouplingPartitionedFieldChannelRequest fieldExchange(
    std::string channel_name,
    std::string producer_participant_name,
    std::string producer_field_name,
    std::string consumer_participant_name,
    std::string consumer_field_name,
    CouplingTransferDeclaration transfer = {},
    CouplingTemporalSlotDescriptor producer_temporal = {},
    CouplingTemporalSlotDescriptor consumer_temporal = {},
    CouplingPartitionedStrategyDeclaration strategy = {});
[[nodiscard]] CouplingPartitionedGroupRequest partitionedGroup(
    std::string group_name,
    std::vector<std::string> participant_names);

class CouplingDefinitionBuilder {
public:
    using MonolithicFormsCallback = CouplingMonolithicFormsCallback;

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
    CouplingDefinitionBuilder& requiredField(
        CouplingFieldUse field,
        CouplingValueDescriptor value);
    CouplingDefinitionBuilder& requiredScalarField(
        std::string participant_name,
        std::string field_name);
    CouplingDefinitionBuilder& requiredVectorField(
        std::string participant_name,
        std::string field_name,
        int components);
    CouplingDefinitionBuilder& additionalField(
        CouplingAdditionalFieldDeclaration declaration);
    CouplingDefinitionBuilder& contractInterfaceField(
        CouplingContractInterfaceFieldRequest request);
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
    CouplingDefinitionBuilder& sidePairedInterface(
        CouplingSidePairedInterfaceRequest request);
    CouplingDefinitionBuilder& sidePairedPDECoupling(
        CouplingSidePairedPDERequest request);
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
    PartitionedExchangeBuilder partitionedFieldChannel(
        CouplingPartitionedFieldChannelRequest request);

    [[nodiscard]] bool hasMonolithicForms() const noexcept;
    [[nodiscard]] bool hasPartitionedExchanges() const noexcept;
    [[nodiscard]] CouplingValidationResult optionValidation() const;
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
    CouplingValidationResult option_validation_;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGDEFINITIONBUILDER_H

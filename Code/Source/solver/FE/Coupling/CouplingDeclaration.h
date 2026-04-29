/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGDECLARATION_H
#define SVMP_FE_COUPLING_COUPLINGDECLARATION_H

/**
 * @file CouplingDeclaration.h
 * @brief Declaration records used to build coupling graphs and setup plans.
 */

#include "Analysis/FormAnalysisBridge.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Coupling/CouplingGeometryRequirements.h"
#include "Coupling/CouplingTemporalRequirements.h"
#include "Coupling/PartitionedCouplingPlan.h"
#include "Forms/FormExpr.h"
#include "Systems/FormsInstaller.h"
#include "Systems/OperatorRegistry.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace spaces {
class FunctionSpace;
}
namespace coupling {

enum class CouplingVariableKind : std::uint8_t {
    Field,
    AuxiliaryState,
    AuxiliaryInput,
    AuxiliaryOutput,
    BoundaryFunctional,
    GlobalScalar,
};

struct CouplingVariableUse {
    CouplingVariableKind kind{CouplingVariableKind::Field};
    std::string participant_name;
    std::string name;
    int component{-1};
    CouplingRequirement requirement{CouplingRequirement::Required};
};

struct CouplingResidualDependency {
    CouplingVariableUse residual_row;
    CouplingVariableUse dependency;
    CouplingDependencyMode mode{CouplingDependencyMode::ImplicitMonolithic};
};

enum class CouplingNonFieldDependencyRequirementKind : std::uint8_t {
    Parameter,
    Coefficient,
    MaterialStateOld,
    MaterialStateWork,
    BoundaryFunctional,
    BoundaryIntegral,
    AuxiliaryState,
    AuxiliaryInput,
    AuxiliaryOutput,
};

struct CouplingNonFieldDependencyRequirement {
    CouplingNonFieldDependencyRequirementKind kind{
        CouplingNonFieldDependencyRequirementKind::AuxiliaryInput};
    std::string participant_name;
    std::string name;
    std::optional<CouplingRegionEndpointDeclaration> region;
    std::optional<CouplingRegionKind> required_region_kind;
    std::string expected_value_type;
    std::optional<std::uint64_t> material_state_byte_offset;
    CouplingRequirement requirement{CouplingRequirement::Required};
    bool require_analysis_variable_key{false};
};

struct CouplingBlockExpectation {
    CouplingVariableUse residual_row;
    CouplingVariableUse dependency;
    bool expected_nonzero{true};
    bool expect_matrix_block{true};
};

enum class CouplingAdditionalFieldScope : std::uint8_t {
    VolumeCell,
    InterfaceFace,
};

enum class CouplingAdditionalFieldNamespace : std::uint8_t {
    Participant,
    Contract,
};

struct CouplingAdditionalFieldDeclaration {
    CouplingAdditionalFieldNamespace field_namespace{
        CouplingAdditionalFieldNamespace::Participant};
    std::string namespace_name;
    std::string system_participant_name;
    std::string field_name;
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components{0};
    CouplingAdditionalFieldScope scope{CouplingAdditionalFieldScope::VolumeCell};
    std::optional<std::string> region_name;
    std::optional<std::string> shared_region_name;
    CouplingRequirement requirement{CouplingRequirement::Required};
};

struct ResolvedCouplingAdditionalFieldDeclaration {
    CouplingAdditionalFieldDeclaration declaration;
    std::string system_name;
    systems::FieldSpec field_spec;
    FieldId field_id{INVALID_FIELD_ID};
};

struct CouplingInstalledDependency {
    analysis::VariableKey residual_row;
    analysis::VariableKey dependency;
    CouplingDependencyMode mode{CouplingDependencyMode::ImplicitMonolithic};
    analysis::DomainKind domain{analysis::DomainKind::Cell};
    bool contributes_matrix_block{false};
    bool contributes_vector{true};
    std::string provider;
};

struct CouplingInstalledBlockProvenance {
    analysis::VariableKey residual_row;
    analysis::VariableKey dependency;
    std::vector<analysis::DomainKind> domains;
    bool has_matrix{false};
    bool has_vector{false};
};

struct CouplingInstallMetadata {
    std::string contribution_name;
    std::string origin;
    std::string system_name;
    systems::OperatorTag operator_name{"equations"};
    std::vector<CouplingInstalledDependency> installed_dependencies;
    std::vector<CouplingInstalledBlockProvenance> installed_blocks;
};

struct CouplingFormContribution {
    std::string contribution_name;
    std::string origin;
    std::string operator_name{"equations"};
    std::vector<CouplingFieldUse> field_uses;
    std::vector<CouplingFieldUse> extra_trial_field_uses;
    systems::FormInstallOptions install_options{};
    forms::FormExpr residual;
};

struct ResolvedCouplingFormContribution {
    std::string contribution_name;
    std::string origin;
    std::string system_name;
    systems::OperatorTag operator_name{"equations"};
    std::vector<FieldId> fields;
    std::vector<FieldId> extra_trial_fields;
    systems::FormInstallOptions install_options{};
    forms::FormExpr residual;
};

struct CouplingFormAnalysisMetadata {
    std::string contribution_name;
    std::string origin;
    std::string system_name;
    systems::OperatorTag operator_name{"equations"};
    std::vector<FieldId> installed_fields;
    std::vector<CouplingInstalledDependency> installed_dependencies;
    std::vector<CouplingInstalledBlockProvenance> installed_blocks;
    std::vector<analysis::FormBridgeFeatureGate> feature_gates;
};

struct CouplingContractDeclaration {
    std::string contract_type;
    std::string contract_name;
    std::vector<CouplingParticipantUse> participants;
    std::vector<CouplingFieldUse> fields;
    std::vector<CouplingRegionUse> regions;
    std::vector<CouplingSharedRegionUse> shared_regions;
    std::vector<CouplingAdditionalFieldDeclaration> additional_fields;
    std::vector<CouplingNonFieldDependencyRequirement> non_field_dependencies;
    std::vector<CouplingResidualDependency> dependencies;
    std::vector<CouplingTemporalRequirement> temporal_requirements;
    std::vector<CouplingGeometryTerminalRequirement> geometry_requirements;
    std::vector<CouplingBlockExpectation> expected_blocks;
    std::vector<CouplingExchangeDeclaration> partitioned_exchange_declarations;
    std::vector<CouplingGroupHint> group_hints;
};

[[nodiscard]] CouplingValidationResult validateContractDeclarationShape(
    const CouplingContractDeclaration& declaration);

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGDECLARATION_H

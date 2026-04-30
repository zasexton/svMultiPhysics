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
#include "Core/ParameterValue.h"
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
    std::optional<params::ValueType> expected_parameter_value_type;
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

struct CouplingGeometrySensitivityDeclaration {
    forms::GeometrySensitivityMode mode{
        forms::GeometrySensitivityMode::GeometryConstant};
    std::optional<CouplingFieldUse> mesh_motion_field;
    forms::GeometryTangentPath tangent_path{forms::GeometryTangentPath::Auto};
    bool use_symbolic_tangent{false};
};

enum class CouplingFormTerminalProvenanceKind : std::uint8_t {
    PreviousSolution,
    MeshTemporal,
    GeometryTerminal,
};

struct CouplingFormTerminalProvenanceDeclaration {
    CouplingFormTerminalProvenanceKind kind{
        CouplingFormTerminalProvenanceKind::GeometryTerminal};
    std::uint64_t terminal_sequence{0};
    std::optional<CouplingFieldUse> field;
    std::optional<CouplingGeometryTerminalScope> scope;
    CouplingTemporalQuantity temporal_quantity{CouplingTemporalQuantity::Time};
    CouplingGeometryTerminalQuantity geometry_quantity{
        CouplingGeometryTerminalQuantity::MeshDisplacement};
    std::optional<systems::MeshMotionFieldRole> mesh_motion_role;
    int derivative_order{0};
    int history_index{0};
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
    std::vector<CouplingFormTerminalProvenanceDeclaration> terminal_provenance;
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
    std::vector<CouplingFormTerminalProvenanceDeclaration> terminal_provenance;
    systems::FormInstallOptions install_options{};
    forms::FormExpr residual;
};

struct CouplingFormFieldProvenance {
    FieldId residual_row{INVALID_FIELD_ID};
    FieldId field{INVALID_FIELD_ID};
    bool appears_as_test_field{false};
    bool appears_as_state_field{false};
    bool appears_as_discrete_field{false};
    bool appears_as_geometry_sensitivity{false};
};

enum class CouplingGeometrySensitivityProvenanceKind : std::uint8_t {
    None,
    MeshMotionUnknowns,
    CutGeometry,
    DriverProvided,
};

struct CouplingGeometrySensitivityProvenance {
    CouplingGeometrySensitivityProvenanceKind kind{
        CouplingGeometrySensitivityProvenanceKind::None};
    FieldId mesh_motion_field{INVALID_FIELD_ID};
    std::string provenance_id;
    std::string construction_policy;
    std::string target_kind;
    std::uint64_t source_stable_id{0};
    std::uint64_t cut_topology_revision{0};
    std::uint64_t quadrature_policy_key{0};
    MeshIndex parent_entity{static_cast<MeshIndex>(-1)};
    bool ad_compatible{false};
    bool location_sensitivity_available{false};
    bool jacobian_sensitivity_available{false};
    bool measure_sensitivity_available{false};
    bool normal_sensitivity_available{false};
    bool quadrature_weight_sensitivity_available{false};
    std::vector<FieldId> geometry_fields;
    std::vector<MeshIndex> parent_geometry_dofs;
    std::vector<std::string> visible_to_assembly_paths;
    std::size_t sensitivity_sample_count{0};
};

struct CouplingFormTemporalProvenance {
    std::optional<FieldId> field;
    std::optional<FieldId> active_trial_field;
    std::optional<analysis::VariableKey> residual_row;
    std::optional<analysis::VariableKey> trial_dependency;
    std::optional<std::string> trial_block_id;
    std::optional<CouplingGeometryTerminalScope> mesh_motion_scope;
    std::optional<systems::MeshMotionFieldRole> mesh_motion_role;
    CouplingTemporalQuantity quantity{CouplingTemporalQuantity::Time};
    int derivative_order{0};
    int history_index{0};
};

struct CouplingFormGeometryTerminalProvenance {
    CouplingGeometryTerminalQuantity quantity{
        CouplingGeometryTerminalQuantity::MeshDisplacement};
    FieldId mesh_motion_field{INVALID_FIELD_ID};
    CouplingGeometryTerminalLocationProvenance location;
    analysis::DomainKind analysis_domain{analysis::DomainKind::Cell};
    std::optional<CouplingGeometryTerminalOwnerProvenance> owner;
    std::string provider;
    bool value_available{false};
    bool gradient_or_jacobian_available{false};
    bool normal_available{false};
    bool measure_available{false};
};

enum class CouplingFormNonFieldDependencyKind : std::uint8_t {
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

struct CouplingFormNonFieldDependencyProvenance {
    CouplingFormNonFieldDependencyKind kind{
        CouplingFormNonFieldDependencyKind::AuxiliaryInput};
    std::string participant_name;
    std::string system_name;
    std::string name;
    analysis::DomainKind domain{analysis::DomainKind::Cell};
    std::optional<std::string> region_name;
    std::optional<std::string> shared_region_name;
    int marker{-1};
    CouplingInterfaceSide side{CouplingInterfaceSide::None};
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::optional<svmp::search::LogicalInterfaceRegionId> logical_region;
#endif
    std::optional<std::uint32_t> slot;
    std::optional<std::uint32_t> output_id;
    std::optional<std::uint64_t> byte_offset;
    std::string provider;
    std::string value_type;
    std::optional<params::ValueType> parameter_value_type;
};

struct CouplingFormVariableDependencyProvenance {
    analysis::VariableKey residual_row;
    analysis::VariableKey dependency;
    CouplingDependencyMode mode{CouplingDependencyMode::ImplicitMonolithic};
    analysis::DomainKind domain{analysis::DomainKind::Cell};
    bool contributes_matrix_block{false};
    bool contributes_vector{true};
    std::string provider;
};

struct CouplingFormAnalysisMetadata {
    std::string contribution_name;
    std::string origin;
    std::string system_name;
    systems::OperatorTag operator_name{"equations"};
    std::vector<FieldId> installed_fields;
    std::vector<CouplingFormFieldProvenance> field_uses;
    std::vector<CouplingFormNonFieldDependencyProvenance> non_field_dependencies;
    std::vector<CouplingFormVariableDependencyProvenance> variable_dependencies;
    std::vector<CouplingFormTerminalProvenanceDeclaration>
        declaration_terminal_provenance;
    std::vector<CouplingFormTemporalProvenance> temporal_symbols;
    std::vector<CouplingFormGeometryTerminalProvenance> geometry_terminals;
    forms::GeometrySensitivityOptions geometry_sensitivity{};
    std::vector<CouplingGeometrySensitivityProvenance>
        geometry_sensitivity_provenance;
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

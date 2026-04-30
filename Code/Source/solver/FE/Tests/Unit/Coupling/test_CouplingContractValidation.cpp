#include "Coupling/CouplingDeclaration.h"
#include "Coupling/CouplingContract.h"
#include "Coupling/CouplingGraph.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <span>

using namespace svmp::FE::coupling;
using svmp::FE::InvalidArgumentException;

namespace {

std::shared_ptr<const svmp::FE::spaces::FunctionSpace> scalarSpace()
{
    return std::make_shared<svmp::FE::spaces::H1Space>(
        svmp::FE::ElementType::Triangle3, 1);
}

CouplingContractDeclaration minimalDeclaration()
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.regions.push_back({
        .participant_name = "left",
        .region_name = "surface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.shared_regions.push_back({
        .shared_region_name = "interface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    return declaration;
}

class MissingFieldContract final : public CouplingContract {
public:
    std::string name() const override { return "missing_field_contract"; }

    CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = name();
        declaration.contract_name = "missing_field_instance";
        declaration.participants.push_back({.participant_name = "left"});
        declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
        return declaration;
    }
};

class MismatchedTypeContract final : public CouplingContract {
public:
    std::string name() const override { return "actual_type"; }

    CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = "declared_type";
        declaration.contract_name = "mismatched_instance";
        return declaration;
    }
};

} // namespace

TEST(CouplingContractValidation, AcceptsMinimalTwoParticipantDeclaration)
{
    const auto declaration = minimalDeclaration();
    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());
}

TEST(CouplingContractValidation, AcceptsValidNParticipantDeclaration)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "multi_interface";
    declaration.contract_name = "multi_interface_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "middle"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.fields.push_back({.participant_name = "middle", .field_name = "primary"});
    declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
    declaration.regions.push_back({
        .participant_name = "left",
        .region_name = "surface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.regions.push_back({
        .participant_name = "middle",
        .region_name = "surface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.regions.push_back({
        .participant_name = "right",
        .region_name = "surface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.shared_regions.push_back({
        .shared_region_name = "triple_surface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.group_hints.push_back(CouplingGroupHint{
        .name = "all_participants",
        .participant_names = {"left", "middle", "right"},
    });

    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());
}

TEST(CouplingContractValidation, RejectsEmptyContractNames)
{
    auto declaration = minimalDeclaration();
    declaration.contract_name.clear();

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("configured contract name"),
              std::string::npos);
}

TEST(CouplingContractValidation, RejectsDuplicateRequirements)
{
    auto declaration = minimalDeclaration();
    declaration.participants.push_back({.participant_name = "left"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.regions.push_back({
        .participant_name = "left",
        .region_name = "surface",
    });
    declaration.shared_regions.push_back({.shared_region_name = "interface"});

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("duplicate participant requirement"), std::string::npos);
    EXPECT_NE(text.find("duplicate field requirement"), std::string::npos);
    EXPECT_NE(text.find("duplicate participant-local region requirement"), std::string::npos);
    EXPECT_NE(text.find("duplicate shared-region requirement"), std::string::npos);
}

TEST(CouplingContractValidation, ValidatesAdditionalFieldAttachmentRules)
{
    auto declaration = minimalDeclaration();
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "lambda",
        .components = -1,
    });
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "surface_field",
        .components = 1,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
    });

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("component count"), std::string::npos);
    EXPECT_NE(text.find("exactly one region attachment"), std::string::npos);
}

TEST(CouplingContractValidation, ValidatesAdditionalFieldSpaceAndComponents)
{
    auto declaration = minimalDeclaration();
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "missing_space",
    });
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "bad_components",
        .space = scalarSpace(),
        .components = 2,
    });

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("function space"), std::string::npos);
    EXPECT_NE(text.find("match the function space"), std::string::npos);

    declaration.additional_fields.clear();
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "inferred_components",
        .space = scalarSpace(),
        .components = 0,
    });

    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());
}

TEST(CouplingContractValidation, ValidatesOptionalAdditionalFieldSelection)
{
    auto declaration = minimalDeclaration();
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "optional_lambda",
        .requirement = CouplingRequirement::Optional,
        .enabled = false,
    });
    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());

    declaration.additional_fields[0].enabled = true;
    const auto selected_validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(selected_validation.ok());
    EXPECT_NE(formatDiagnostics(selected_validation).find("function space"),
              std::string::npos);

    declaration.additional_fields[0].space = scalarSpace();
    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());

    declaration.additional_fields[0].requirement = CouplingRequirement::Required;
    declaration.additional_fields[0].enabled = false;
    const auto required_disabled = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(required_disabled.ok());
    EXPECT_NE(formatDiagnostics(required_disabled).find("cannot be disabled"),
              std::string::npos);

    declaration.additional_fields[0].requirement = CouplingRequirement::Optional;
    declaration.dependencies.push_back(CouplingResidualDependency{
        .residual_row = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "generic_instance",
            .name = "optional_lambda",
        },
        .dependency = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "left",
            .name = "primary",
        },
    });
    const auto referenced_disabled = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(referenced_disabled.ok());
    EXPECT_NE(formatDiagnostics(referenced_disabled).find("disabled optional additional field"),
              std::string::npos);
}

TEST(CouplingContractValidation, FormAnalysisMetadataStoresDiagnosticProvenance)
{
    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "coupled_surface";
    metadata.origin = "surface_contract";
    metadata.system_name = "fluid";
    metadata.installed_fields = {1, 2};
    metadata.field_uses.push_back(CouplingFormFieldProvenance{
        .residual_row = 1,
        .field = 2,
        .appears_as_state_field = true,
        .appears_as_geometry_sensitivity = true,
    });
    metadata.non_field_dependencies.push_back(CouplingFormNonFieldDependencyProvenance{
        .kind = CouplingFormNonFieldDependencyKind::BoundaryIntegral,
        .participant_name = "fluid",
        .system_name = "fluid_system",
        .name = "traction",
        .domain = svmp::FE::analysis::DomainKind::Boundary,
        .region_name = "wall",
        .marker = 12,
        .provider = "forms",
        .value_type = "scalar",
        .parameter_value_type = svmp::FE::params::ValueType::Real,
    });
    metadata.variable_dependencies.push_back(CouplingFormVariableDependencyProvenance{
        .residual_row = svmp::FE::analysis::VariableKey::field(1),
        .dependency = svmp::FE::analysis::VariableKey::field(2),
        .domain = svmp::FE::analysis::DomainKind::Boundary,
        .contributes_matrix_block = true,
        .contributes_vector = true,
        .provider = "forms",
    });
    metadata.declaration_terminal_provenance.push_back(
        CouplingFormTerminalProvenanceDeclaration{
            .kind = CouplingFormTerminalProvenanceKind::PreviousSolution,
            .terminal_sequence = 3,
            .field = CouplingFieldUse{
                .participant_name = "fluid",
                .field_name = "velocity",
            },
            .temporal_quantity = CouplingTemporalQuantity::FieldHistoryValue,
            .history_index = 2,
        });
    metadata.temporal_symbols.push_back(CouplingFormTemporalProvenance{
        .field = 2,
        .active_trial_field = 2,
        .residual_row = svmp::FE::analysis::VariableKey::field(1),
        .trial_dependency = svmp::FE::analysis::VariableKey::field(2),
        .quantity = CouplingTemporalQuantity::FieldHistoryValue,
        .history_index = 2,
    });
    metadata.geometry_terminals.push_back(CouplingFormGeometryTerminalProvenance{
        .quantity = CouplingGeometryTerminalQuantity::CurrentNormal,
        .mesh_motion_field = 2,
        .location = CouplingGeometryTerminalLocationProvenance{
            .region_kind = CouplingRegionKind::Boundary,
            .marker = 12,
        },
        .analysis_domain = svmp::FE::analysis::DomainKind::Boundary,
        .owner = CouplingGeometryTerminalOwnerProvenance{
            .participant_name = "fluid",
            .system_name = "fluid_system",
            .region_name = "wall",
        },
        .provider = "forms",
        .normal_available = true,
    });
    metadata.geometry_sensitivity.mode =
        svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns;
    metadata.geometry_sensitivity.mesh_motion_field = 2;
    metadata.geometry_sensitivity_provenance.push_back(
        CouplingGeometrySensitivityProvenance{
            .kind = CouplingGeometrySensitivityProvenanceKind::MeshMotionUnknowns,
            .mesh_motion_field = 2,
            .provenance_id = "mesh-motion",
            .construction_policy = "registered_field",
            .target_kind = "surface",
            .ad_compatible = true,
            .measure_sensitivity_available = true,
            .geometry_fields = {2},
            .visible_to_assembly_paths = {"residual"},
            .sensitivity_sample_count = 4,
        });

    ASSERT_EQ(metadata.field_uses.size(), 1u);
    EXPECT_TRUE(metadata.field_uses[0].appears_as_state_field);
    EXPECT_TRUE(metadata.field_uses[0].appears_as_geometry_sensitivity);
    ASSERT_EQ(metadata.non_field_dependencies.size(), 1u);
    EXPECT_EQ(metadata.non_field_dependencies[0].kind,
              CouplingFormNonFieldDependencyKind::BoundaryIntegral);
    ASSERT_TRUE(metadata.non_field_dependencies[0].parameter_value_type.has_value());
    EXPECT_EQ(*metadata.non_field_dependencies[0].parameter_value_type,
              svmp::FE::params::ValueType::Real);
    ASSERT_EQ(metadata.variable_dependencies.size(), 1u);
    EXPECT_TRUE(metadata.variable_dependencies[0].contributes_matrix_block);
    ASSERT_EQ(metadata.declaration_terminal_provenance.size(), 1u);
    EXPECT_EQ(metadata.declaration_terminal_provenance[0].history_index, 2);
    ASSERT_EQ(metadata.temporal_symbols.size(), 1u);
    EXPECT_EQ(metadata.temporal_symbols[0].quantity,
              CouplingTemporalQuantity::FieldHistoryValue);
    ASSERT_EQ(metadata.geometry_terminals.size(), 1u);
    EXPECT_TRUE(metadata.geometry_terminals[0].normal_available);
    ASSERT_TRUE(metadata.geometry_terminals[0].owner.has_value());
    EXPECT_EQ(metadata.geometry_terminals[0].owner->participant_name, "fluid");
    EXPECT_EQ(metadata.geometry_sensitivity.mode,
              svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns);
    ASSERT_EQ(metadata.geometry_sensitivity_provenance.size(), 1u);
    EXPECT_TRUE(metadata.geometry_sensitivity_provenance[0].ad_compatible);
    EXPECT_EQ(metadata.geometry_sensitivity_provenance[0].sensitivity_sample_count, 4u);
}

TEST(CouplingContractValidation, ValidatesNonFieldDependencyRequirements)
{
    auto declaration = minimalDeclaration();
    declaration.non_field_dependencies = {
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::Parameter,
            .participant_name = "left",
            .name = "penalty",
            .expected_parameter_value_type = svmp::FE::params::ValueType::Real,
            .expected_value_type = "scalar",
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::Coefficient,
            .participant_name = "left",
            .name = "wall_weight",
            .region = CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
            },
            .required_region_kind = CouplingRegionKind::Boundary,
            .expected_value_type = "scalar",
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::MaterialStateOld,
            .participant_name = "left",
            .name = "history",
            .material_state_byte_offset = 16,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::BoundaryIntegral,
            .participant_name = "left",
            .name = "traction_integral",
            .require_analysis_variable_key = true,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryState,
            .participant_name = "left",
            .name = "state",
            .require_analysis_variable_key = true,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryInput,
            .participant_name = "left",
            .name = "input",
            .require_analysis_variable_key = true,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput,
            .participant_name = "left",
            .name = "output",
            .require_analysis_variable_key = true,
        },
    };
    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());

    declaration.non_field_dependencies = {
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::Parameter,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::Coefficient,
            .participant_name = "left",
            .name = "coefficient",
            .region = CouplingRegionEndpointDeclaration{
                .participant_name = "right",
                .region_name = "surface",
            },
            .expected_parameter_value_type = svmp::FE::params::ValueType::Real,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::Parameter,
            .participant_name = "left",
            .name = "bad_offset",
            .material_state_byte_offset = 8,
            .require_analysis_variable_key = true,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryInput,
            .participant_name = "left",
            .name = "duplicate",
            .require_analysis_variable_key = true,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryInput,
            .participant_name = "left",
            .name = "duplicate",
            .require_analysis_variable_key = true,
        },
    };

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("participant name"), std::string::npos);
    EXPECT_NE(text.find("requires a name"), std::string::npos);
    EXPECT_NE(text.find("region scope must match"), std::string::npos);
    EXPECT_NE(text.find("expected parameter value type"), std::string::npos);
    EXPECT_NE(text.find("material-state byte offset"), std::string::npos);
    EXPECT_NE(text.find("analysis variable identity"), std::string::npos);
    EXPECT_NE(text.find("duplicate non-field dependency requirement"), std::string::npos);
}

TEST(CouplingContractValidation, ValidatesTemporalRequirementsAndExchangeShape)
{
    auto declaration = minimalDeclaration();
    declaration.temporal_requirements.push_back({
        .quantity = CouplingTemporalQuantity::FieldDerivative,
        .derivative_order = 0,
    });
    declaration.partitioned_exchange_declarations.push_back({
        .producer_port = {.contract_instance_name = "generic_instance", .port_name = "out"},
        .consumer_port = {.contract_instance_name = "generic_instance", .port_name = "in"},
        .value = {.rank = CouplingValueRank::MixedBlock, .components = 2},
    });

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("positive derivative order"), std::string::npos);
    EXPECT_NE(text.find("mixed block values require component layout"), std::string::npos);
}

TEST(CouplingContractValidation, CouplingGraphRejectsDuplicateInstances)
{
    auto a = minimalDeclaration();
    auto b = minimalDeclaration();
    b.contract_type = "other_generic";

    CouplingGraph graph;
    CouplingContext context;
    const std::array<CouplingContractDeclaration, 2> declarations{a, b};
    const auto validation =
        graph.buildDeclarationGraph(context, std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("duplicate coupling contract instance name"),
              std::string::npos);
}

TEST(CouplingContractValidation, DefaultContractValidationChecksDeclarationGraph)
{
    const MissingFieldContract contract;

    EXPECT_THROW(contract.validate(CouplingContext{}), InvalidArgumentException);
}

TEST(CouplingContractValidation, DefaultContractValidationRejectsTypeMismatch)
{
    const MismatchedTypeContract contract;

    EXPECT_THROW(contract.validate(CouplingContext{}), InvalidArgumentException);
}

#include "Coupling/CouplingDeclaration.h"
#include "Coupling/CouplingContract.h"
#include "Coupling/CouplingGraph.h"
#include "Coupling/MonolithicCouplingBuilder.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>

using namespace svmp::FE::coupling;
using svmp::FE::InvalidArgumentException;

namespace {

std::shared_ptr<const svmp::FE::spaces::FunctionSpace> scalarSpace()
{
    return std::make_shared<svmp::FE::spaces::H1Space>(
        svmp::FE::ElementType::Triangle3, 1);
}

const svmp::FE::systems::FESystem* systemToken(std::uintptr_t value)
{
    return reinterpret_cast<const svmp::FE::systems::FESystem*>(value);
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

class ExpertInstallContract final : public CouplingContract {
public:
    [[nodiscard]] std::string name() const override { return "expert_install"; }

    [[nodiscard]] CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = name();
        declaration.contract_name = "expert_install_instance";
        return declaration;
    }

    [[nodiscard]] bool supportsMonolithicLowering() const override { return true; }

    [[nodiscard]] std::vector<CouplingInstallMetadata> installMonolithicTerms(
        MonolithicCouplingInstallContext& install,
        const CouplingContext& ctx) override
    {
        called_ = true;
        install_context_ = &install;
        coupling_context_ = &ctx;
        return {CouplingInstallMetadata{
            .contribution_name = "expert_term",
            .origin = "ExpertInstallContract",
            .system_name = "left_system",
        }};
    }

    [[nodiscard]] bool called() const noexcept { return called_; }
    [[nodiscard]] const MonolithicCouplingInstallContext* installContext() const noexcept
    {
        return install_context_;
    }
    [[nodiscard]] const CouplingContext* couplingContext() const noexcept
    {
        return coupling_context_;
    }

private:
    bool called_{false};
    const MonolithicCouplingInstallContext* install_context_{nullptr};
    const CouplingContext* coupling_context_{nullptr};
};

} // namespace

TEST(CouplingContractValidation, AcceptsMinimalTwoParticipantDeclaration)
{
    const auto declaration = minimalDeclaration();
    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());
}

TEST(CouplingContractValidation, PreservesDependencyDeclarationMode)
{
    static_assert(std::is_same_v<decltype(CouplingContractDeclaration::
                                              dependency_declaration_mode),
                                 CouplingDependencyDeclarationMode>);

    auto declaration = minimalDeclaration();
    EXPECT_EQ(declaration.dependency_declaration_mode,
              CouplingDependencyDeclarationMode::DeclareAndVerify);

    declaration.dependency_declaration_mode =
        CouplingDependencyDeclarationMode::InferFromInstalledForms;
    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());

    declaration.dependency_declaration_mode =
        CouplingDependencyDeclarationMode::ExpertProvided;
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

TEST(CouplingContractValidation, InvokesExpertInstallHooksThroughInstallContext)
{
    using InstallHook = std::vector<CouplingInstallMetadata> (CouplingContract::*)(
        MonolithicCouplingInstallContext&,
        const CouplingContext&);
    static_assert(std::is_same_v<decltype(&CouplingContract::installMonolithicTerms),
                                 InstallHook>);

    ExpertInstallContract contract;
    MonolithicCouplingInstallContext install_context;
    const CouplingContext coupling_context;
    const auto metadata =
        contract.installMonolithicTerms(install_context, coupling_context);

    EXPECT_TRUE(contract.called());
    EXPECT_EQ(contract.installContext(), &install_context);
    EXPECT_EQ(contract.couplingContext(), &coupling_context);
    ASSERT_EQ(metadata.size(), 1u);
    EXPECT_EQ(metadata[0].contribution_name, "expert_term");
    EXPECT_EQ(metadata[0].origin, "ExpertInstallContract");
}

TEST(CouplingContractValidation, ValidatesFormContributionNamesByContract)
{
    CouplingFormContribution unnamed;
    unnamed.origin = "UnnamedContract";
    const std::array<CouplingFormContribution, 1> unnamed_contributions{unnamed};
    const auto unnamed_validation = validateFormContributionDeclarations(
        std::span<const CouplingFormContribution>(unnamed_contributions));
    EXPECT_FALSE(unnamed_validation.ok());
    EXPECT_NE(formatDiagnostics(unnamed_validation).find("contribution name"),
              std::string::npos);

    CouplingFormContribution first;
    first.contribution_name = "surface_balance";
    first.origin = "FirstContract";
    first.operator_name = "equations";
    CouplingFormContribution duplicate = first;
    duplicate.origin = "FirstContractDuplicate";
    const std::array<CouplingFormContribution, 2> duplicates{first, duplicate};
    const auto duplicate_validation = validateFormContributionDeclarations(
        std::span<const CouplingFormContribution>(duplicates));
    EXPECT_FALSE(duplicate_validation.ok());
    EXPECT_NE(formatDiagnostics(duplicate_validation).find(
                  "duplicate coupling form contribution name"),
              std::string::npos);

    CouplingFormContribution second;
    second.contribution_name = "traction_balance";
    second.origin = "SecondContract";
    second.operator_name = "equations";
    const std::array<CouplingFormContribution, 1> first_contract{first};
    const std::array<CouplingFormContribution, 1> second_contract{second};
    EXPECT_TRUE(validateFormContributionDeclarations(
                    std::span<const CouplingFormContribution>(first_contract))
                    .ok());
    EXPECT_TRUE(validateFormContributionDeclarations(
                    std::span<const CouplingFormContribution>(second_contract))
                    .ok());
    EXPECT_EQ(first_contract[0].operator_name, second_contract[0].operator_name);
    EXPECT_NE(first_contract[0].contribution_name,
              second_contract[0].contribution_name);
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

TEST(CouplingContractValidation, AcceptsFieldShapeAndSharedInterfaceRequirements)
{
    auto declaration = minimalDeclaration();
    declaration.field_requirements.push_back({
        .field = {.participant_name = "left", .field_name = "primary"},
        .value = {.rank = CouplingValueRank::Vector, .components = 3},
        .required_scope = svmp::FE::systems::FieldScope::InterfaceFace,
    });
    declaration.shared_interface_requirements.push_back({
        .shared_region_name = "interface",
        .participant_names = {"left", "right"},
    });

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingContractValidation, ValidatesFieldShapeRequirements)
{
    auto declaration = minimalDeclaration();
    declaration.field_requirements.push_back({
        .field = {.participant_name = "left", .field_name = "primary"},
        .value = {.rank = CouplingValueRank::Scalar, .components = 2},
    });
    declaration.field_requirements.push_back(declaration.field_requirements.back());

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("scalar coupling values require exactly one component"),
              std::string::npos);
    EXPECT_NE(text.find("duplicate field-shape requirement"), std::string::npos);
}

TEST(CouplingContractValidation, ValidatesSharedInterfaceRequirements)
{
    auto declaration = minimalDeclaration();
    declaration.shared_interface_requirements.push_back({
        .shared_region_name = "interface",
        .participant_names = {"left", "", "left"},
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.shared_interface_requirements.push_back(
        declaration.shared_interface_requirements.back());

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("shared-interface requirement requires interface-face region kind"),
              std::string::npos);
    EXPECT_NE(text.find("shared-interface requirement requires nonempty participant names"),
              std::string::npos);
    EXPECT_NE(text.find("duplicate participant in shared-interface requirement"),
              std::string::npos);
    EXPECT_NE(text.find("duplicate shared-interface requirement"), std::string::npos);
}

TEST(CouplingContractValidation, AcceptsRegionRelationRequirements)
{
    auto declaration = minimalDeclaration();
    declaration.region_relation_requirements.push_back({
        .relation_name = "embedded_surface",
        .relation_kind = CouplingRegionRelationKind::EmbeddedRelation,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "right",
                .region_name = "surface",
                .shared_region_name = "interface",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            },
            CouplingRelationLoweringCapability{
                .lowering_kind =
                    CouplingRelationLoweringKind::PartitionedExchange,
            },
        },
        .require_distinct_participants = true,
        .require_registered_topology = true,
    });

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingContractValidation, ValidatesRegionRelationRequirements)
{
    auto declaration = minimalDeclaration();
    CouplingRegionRelationRequirement invalid_pair{
        .relation_name = "bad_pair",
        .relation_kind = CouplingRegionRelationKind::SidePairedInterface,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
                .supported = false,
            },
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            },
        },
        .required_region_kind = CouplingRegionKind::Boundary,
        .require_distinct_participants = true,
        .require_opposite_sides_for_side_pair = true,
    };
    declaration.region_relation_requirements.push_back(invalid_pair);
    declaration.region_relation_requirements.push_back(invalid_pair);
    declaration.region_relation_requirements.push_back({
        .relation_name = "short_n_way",
        .relation_kind = CouplingRegionRelationKind::NWayInterface,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "right",
                .region_name = "surface",
                .shared_region_name = "",
            },
        },
    });
    declaration.region_relation_requirements.push_back({
        .relation_name = "no_supported_lowering",
        .relation_kind = CouplingRegionRelationKind::EmbeddedRelation,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "right",
                .region_name = "surface",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
                .supported = false,
                .unsupported_reason = "requires custom assembly",
            },
        },
    });

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("opposite-side relation requires interface-face region kind"),
              std::string::npos);
    EXPECT_NE(text.find("unsupported relation lowering capability requires a reason"),
              std::string::npos);
    EXPECT_NE(text.find("duplicate relation lowering capability"),
              std::string::npos);
    EXPECT_NE(text.find("region-relation requirement requires lowering capabilities"),
              std::string::npos);
    EXPECT_NE(text.find(
                  "region-relation requirement requires at least one supported lowering capability"),
              std::string::npos);
    EXPECT_NE(text.find("duplicate region-relation requirement"), std::string::npos);
    EXPECT_NE(text.find("duplicate endpoint in region-relation requirement"),
              std::string::npos);
    EXPECT_NE(text.find("region-relation requirement requires distinct participants"),
              std::string::npos);
    EXPECT_NE(text.find("N-way interface relation requires at least two endpoints"),
              std::string::npos);
    EXPECT_NE(text.find("region-relation endpoint shared-region name cannot be empty"),
              std::string::npos);
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

TEST(CouplingContractValidation, ValidatesAdditionalFieldCollisionsAndUniqueFields)
{
    auto declaration = minimalDeclaration();
    const auto space = scalarSpace();
    const CouplingAdditionalFieldDeclaration participant_field{
        .field_namespace = CouplingAdditionalFieldNamespace::Participant,
        .namespace_name = "left",
        .field_name = "lambda",
        .space = space,
        .components = 1,
    };
    declaration.additional_fields = {participant_field, participant_field};

    const auto duplicate_validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(duplicate_validation.ok());
    EXPECT_NE(formatDiagnostics(duplicate_validation).find(
                  "duplicate additional field declaration"),
              std::string::npos);

    declaration.additional_fields = {
        CouplingAdditionalFieldDeclaration{
            .field_namespace = CouplingAdditionalFieldNamespace::Participant,
            .namespace_name = "left",
            .field_name = "lambda",
            .space = space,
            .components = 1,
        },
        CouplingAdditionalFieldDeclaration{
            .field_namespace = CouplingAdditionalFieldNamespace::Participant,
            .namespace_name = "right",
            .field_name = "lambda",
            .space = space,
            .components = 1,
        },
        CouplingAdditionalFieldDeclaration{
            .field_namespace = CouplingAdditionalFieldNamespace::Contract,
            .namespace_name = "generic_instance",
            .system_participant_name = "left",
            .field_name = "lambda",
            .space = space,
            .components = 1,
        },
    };
    const auto unique_validation = validateContractDeclarationShape(declaration);
    EXPECT_TRUE(unique_validation.ok()) << formatDiagnostics(unique_validation);
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
    const auto skipped_validation = validateContractDeclarationShape(declaration);
    EXPECT_TRUE(skipped_validation.ok());
    EXPECT_NE(formatDiagnostics(skipped_validation).find("disabled optional additional field is skipped"),
              std::string::npos);

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

    declaration.dependencies.clear();
    declaration.expected_blocks.push_back(CouplingBlockExpectation{
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
    const auto expected_block_reference = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(expected_block_reference.ok());
    EXPECT_NE(formatDiagnostics(expected_block_reference).find("disabled optional additional field"),
              std::string::npos);
}

TEST(CouplingContractValidation, ResolvesInterfaceAdditionalFieldMarkersFromContext)
{
    const auto* system = systemToken(1);
    const CouplingRegionRef left_surface{
        .participant_name = "left",
        .system_name = "shared_system",
        .system = system,
        .region_name = "surface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 17,
        .side = CouplingInterfaceSide::Minus,
    };
    const CouplingRegionRef right_surface{
        .participant_name = "right",
        .system_name = "shared_system",
        .system = system,
        .region_name = "surface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 17,
        .side = CouplingInterfaceSide::Plus,
    };

    CouplingContextBuilder context_builder;
    context_builder
        .addParticipant({
            .participant_name = "left",
            .system_name = "shared_system",
            .system = system,
        })
        .addParticipant({
            .participant_name = "right",
            .system_name = "shared_system",
            .system = system,
        })
        .addRegion(left_surface)
        .addRegion(right_surface)
        .addSharedRegion(SharedRegionRef{
            .name = "interface",
            .required_region_kind = CouplingRegionKind::InterfaceFace,
            .participant_regions = {left_surface, right_surface},
        });
    const auto context = context_builder.build();

    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.shared_regions.push_back({
        .shared_region_name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
    });
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Participant,
        .namespace_name = "left",
        .field_name = "trace",
        .space = scalarSpace(),
        .components = 1,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
        .region_name = "surface",
    });
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "lambda",
        .space = scalarSpace(),
        .components = 1,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
        .shared_region_name = "interface",
    });
    ASSERT_TRUE(validateContractDeclarationShape(declaration).ok());

    const MonolithicCouplingBuilder builder;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto resolved = builder.resolveAdditionalFields(
        context,
        std::span<const CouplingContractDeclaration>(declarations));

    ASSERT_EQ(resolved.size(), 2u);
    EXPECT_EQ(resolved[0].declaration.region_name.value(), "surface");
    EXPECT_FALSE(resolved[0].declaration.shared_region_name.has_value());
    EXPECT_EQ(resolved[0].field_spec.scope, svmp::FE::systems::FieldScope::InterfaceFace);
    EXPECT_EQ(resolved[0].field_spec.interface_marker, 17);
    EXPECT_EQ(resolved[0].field_id, svmp::FE::INVALID_FIELD_ID);

    EXPECT_FALSE(resolved[1].declaration.region_name.has_value());
    EXPECT_EQ(resolved[1].declaration.shared_region_name.value(), "interface");
    EXPECT_EQ(resolved[1].field_spec.scope, svmp::FE::systems::FieldScope::InterfaceFace);
    EXPECT_EQ(resolved[1].field_spec.interface_marker, 17);
    EXPECT_EQ(resolved[1].field_id, svmp::FE::INVALID_FIELD_ID);
}

TEST(CouplingContractValidation, ResolvesAdditionalFieldNamespacesAndTargets)
{
    const auto* left_system = systemToken(1);
    const auto* right_system = systemToken(2);
    CouplingContextBuilder context_builder;
    context_builder
        .addParticipant({
            .participant_name = "left",
            .system_name = "left_system",
            .system = left_system,
        })
        .addParticipant({
            .participant_name = "right",
            .system_name = "right_system",
            .system = right_system,
        });
    const auto context = context_builder.build();

    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Participant,
        .namespace_name = "left",
        .field_name = "lambda",
        .space = scalarSpace(),
        .components = 1,
    });
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .system_participant_name = "right",
        .field_name = "lambda",
        .space = scalarSpace(),
        .components = 1,
    });
    ASSERT_TRUE(validateContractDeclarationShape(declaration).ok());

    const MonolithicCouplingBuilder builder;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto resolved = builder.resolveAdditionalFields(
        context,
        std::span<const CouplingContractDeclaration>(declarations));

    ASSERT_EQ(resolved.size(), 2u);
    EXPECT_EQ(resolved[0].system_name, "left_system");
    EXPECT_EQ(resolved[0].field_spec.name, "left.lambda");
    EXPECT_EQ(resolved[1].system_name, "right_system");
    EXPECT_EQ(resolved[1].field_spec.name, "generic_instance.lambda");

    declaration.additional_fields[1].system_participant_name.clear();
    const std::array<CouplingContractDeclaration, 1> missing_target{declaration};
    EXPECT_THROW(static_cast<void>(builder.resolveAdditionalFields(
                     context,
                     std::span<const CouplingContractDeclaration>(missing_target))),
                 svmp::FE::InvalidArgumentException);
}

TEST(CouplingContractValidation, PreservesDependencyModesAndResolutionEvidence)
{
    auto declaration = minimalDeclaration();
    const CouplingVariableUse residual{
        .kind = CouplingVariableKind::Field,
        .participant_name = "left",
        .name = "primary",
    };
    const CouplingVariableUse dependency{
        .kind = CouplingVariableKind::Field,
        .participant_name = "right",
        .name = "primary",
    };
    declaration.dependencies.push_back(CouplingResidualDependency{
        .residual_row = residual,
        .dependency = dependency,
        .mode = CouplingDependencyMode::ImplicitMonolithic,
    });
    declaration.dependencies.push_back(CouplingResidualDependency{
        .residual_row = residual,
        .dependency = dependency,
        .mode = CouplingDependencyMode::ExternalLagged,
    });

    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok());
    ASSERT_EQ(declaration.dependencies.size(), 2u);
    EXPECT_EQ(declaration.dependencies[0].mode,
              CouplingDependencyMode::ImplicitMonolithic);
    EXPECT_EQ(declaration.dependencies[1].mode,
              CouplingDependencyMode::ExternalLagged);

    CouplingFormAnalysisMetadata metadata;
    metadata.field_uses.push_back(CouplingFormFieldProvenance{
        .residual_row = 1,
        .field = 2,
        .appears_as_state_field = true,
    });
    metadata.geometry_sensitivity_provenance.push_back(
        CouplingGeometrySensitivityProvenance{
            .kind = CouplingGeometrySensitivityProvenanceKind::MeshMotionUnknowns,
            .mesh_motion_field = 2,
            .provenance_id = "mesh-motion",
            .geometry_fields = {2},
        });
    metadata.variable_dependencies.push_back(CouplingFormVariableDependencyProvenance{
        .residual_row = svmp::FE::analysis::VariableKey::field(1),
        .dependency = svmp::FE::analysis::VariableKey::named(
            svmp::FE::analysis::VariableKind::BoundaryFunctional,
            "right/traction"),
        .mode = CouplingDependencyMode::ImplicitMonolithic,
        .domain = svmp::FE::analysis::DomainKind::Boundary,
        .contributes_matrix_block = true,
        .provider = "forms",
    });

    ASSERT_EQ(metadata.field_uses.size(), 1u);
    EXPECT_TRUE(metadata.field_uses[0].appears_as_state_field);
    ASSERT_EQ(metadata.geometry_sensitivity_provenance.size(), 1u);
    EXPECT_EQ(metadata.geometry_sensitivity_provenance[0].mesh_motion_field, 2);
    ASSERT_EQ(metadata.variable_dependencies.size(), 1u);
    EXPECT_EQ(metadata.variable_dependencies[0].mode,
              CouplingDependencyMode::ImplicitMonolithic);
    EXPECT_EQ(metadata.variable_dependencies[0].dependency.kind,
              svmp::FE::analysis::VariableKind::BoundaryFunctional);
}

TEST(CouplingContractValidation, PreservesVariableUseKindsInDependencies)
{
    auto declaration = minimalDeclaration();
    const CouplingVariableUse residual{
        .kind = CouplingVariableKind::Field,
        .participant_name = "left",
        .name = "primary",
    };
    const std::array<CouplingVariableUse, 6> dependencies{{
        CouplingVariableUse{
            .kind = CouplingVariableKind::Field,
            .participant_name = "right",
            .name = "shared",
            .component = 2,
        },
        CouplingVariableUse{
            .kind = CouplingVariableKind::AuxiliaryState,
            .participant_name = "right",
            .name = "shared",
        },
        CouplingVariableUse{
            .kind = CouplingVariableKind::AuxiliaryInput,
            .participant_name = "right",
            .name = "shared",
        },
        CouplingVariableUse{
            .kind = CouplingVariableKind::AuxiliaryOutput,
            .participant_name = "right",
            .name = "shared",
        },
        CouplingVariableUse{
            .kind = CouplingVariableKind::BoundaryFunctional,
            .participant_name = "right",
            .name = "shared",
        },
        CouplingVariableUse{
            .kind = CouplingVariableKind::GlobalScalar,
            .participant_name = "right",
            .name = "shared",
        },
    }};
    for (const auto& dependency : dependencies) {
        declaration.dependencies.push_back(CouplingResidualDependency{
            .residual_row = residual,
            .dependency = dependency,
        });
    }

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
    ASSERT_EQ(declaration.dependencies.size(), dependencies.size());
    for (std::size_t i = 0; i < dependencies.size(); ++i) {
        EXPECT_EQ(declaration.dependencies[i].dependency.kind, dependencies[i].kind);
        EXPECT_EQ(declaration.dependencies[i].dependency.participant_name, "right");
        EXPECT_EQ(declaration.dependencies[i].dependency.name, "shared");
    }
    EXPECT_EQ(declaration.dependencies[0].dependency.component, 2);

    declaration.dependencies.push_back(declaration.dependencies[2]);
    const auto duplicate_validation = validateContractDeclarationShape(declaration);
    EXPECT_FALSE(duplicate_validation.ok());
    EXPECT_NE(formatDiagnostics(duplicate_validation).find(
                  "duplicate residual dependency declaration"),
              std::string::npos);
}

TEST(CouplingContractValidation, AdaptsNonFieldVariableUsesToAnalysisKeys)
{
    CouplingContextBuilder context_builder;
    context_builder
        .addParticipant({
            .participant_name = "fluid",
            .system_name = "fluid_system",
            .system = systemToken(1),
        })
        .addParticipant({
            .participant_name = "solid",
            .system_name = "solid_system",
            .system = systemToken(2),
        });
    const auto context = context_builder.build();

    const CouplingVariableUse auxiliary_input{
        .kind = CouplingVariableKind::AuxiliaryInput,
        .participant_name = "fluid",
        .name = "inlet_pressure",
    };
    const auto auxiliary_input_key =
        resolveCouplingVariableUse(context, auxiliary_input);
    ASSERT_TRUE(auxiliary_input_key.has_value());
    EXPECT_EQ(auxiliary_input_key->kind,
              svmp::FE::analysis::VariableKind::AuxiliaryInput);
    EXPECT_EQ(auxiliary_input_key->name, "fluid_system/inlet_pressure");
    EXPECT_EQ(auxiliary_input_key->field_id, svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(auxiliary_input_key->component, -1);

    const CouplingVariableUse auxiliary_output{
        .kind = CouplingVariableKind::AuxiliaryOutput,
        .participant_name = "solid",
        .name = "wall_force",
    };
    const auto auxiliary_output_key =
        resolveCouplingVariableUse(context, auxiliary_output);
    ASSERT_TRUE(auxiliary_output_key.has_value());
    EXPECT_EQ(auxiliary_output_key->kind,
              svmp::FE::analysis::VariableKind::AuxiliaryOutput);
    EXPECT_EQ(auxiliary_output_key->name, "solid_system/wall_force");

    const CouplingVariableUse unresolved_owner{
        .kind = CouplingVariableKind::BoundaryFunctional,
        .participant_name = "interface",
        .name = "traction",
    };
    const auto unresolved_key =
        resolveCouplingVariableUse(context, unresolved_owner);
    ASSERT_TRUE(unresolved_key.has_value());
    EXPECT_EQ(unresolved_key->kind,
              svmp::FE::analysis::VariableKind::BoundaryFunctional);
    EXPECT_EQ(unresolved_key->name, "interface/traction");

    const CouplingVariableUse global_scalar{
        .kind = CouplingVariableKind::GlobalScalar,
        .name = "mass_balance",
    };
    const auto global_key = resolveCouplingVariableUse(context, global_scalar);
    ASSERT_TRUE(global_key.has_value());
    EXPECT_EQ(global_key->kind,
              svmp::FE::analysis::VariableKind::GlobalScalar);
    EXPECT_EQ(global_key->name, "mass_balance");

    CouplingFormAnalysisMetadata metadata;
    metadata.non_field_dependencies.push_back(
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::AuxiliaryInput,
            .participant_name = "fluid",
            .system_name = "fluid_system",
            .name = "inlet_pressure",
            .slot = 7,
            .provider = "forms",
        });
    metadata.non_field_dependencies.push_back(
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::AuxiliaryOutput,
            .participant_name = "solid",
            .system_name = "solid_system",
            .name = "wall_force",
            .output_id = 11,
            .provider = "forms",
        });
    metadata.variable_dependencies.push_back(CouplingFormVariableDependencyProvenance{
        .residual_row = svmp::FE::analysis::VariableKey::field(3),
        .dependency = *auxiliary_input_key,
        .provider = "forms",
    });
    metadata.variable_dependencies.push_back(CouplingFormVariableDependencyProvenance{
        .residual_row = svmp::FE::analysis::VariableKey::field(4),
        .dependency = *auxiliary_output_key,
        .provider = "forms",
    });

    ASSERT_EQ(metadata.non_field_dependencies.size(), 2u);
    ASSERT_TRUE(metadata.non_field_dependencies[0].slot.has_value());
    EXPECT_EQ(*metadata.non_field_dependencies[0].slot, 7u);
    ASSERT_TRUE(metadata.non_field_dependencies[1].output_id.has_value());
    EXPECT_EQ(*metadata.non_field_dependencies[1].output_id, 11u);
    ASSERT_EQ(metadata.variable_dependencies.size(), 2u);
    EXPECT_EQ(metadata.variable_dependencies[0].dependency.name,
              "fluid_system/inlet_pressure");
    EXPECT_EQ(metadata.variable_dependencies[1].dependency.name,
              "solid_system/wall_force");
}

TEST(CouplingContractValidation, HandlesOptionalAndRequiredContextDeclarations)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "optional_contract";
    declaration.contract_name = "optional_instance";
    declaration.participants.push_back({
        .participant_name = "left",
        .requirement = CouplingRequirement::Optional,
    });
    declaration.fields.push_back({
        .participant_name = "left",
        .field_name = "primary",
        .requirement = CouplingRequirement::Optional,
    });
    declaration.regions.push_back({
        .participant_name = "left",
        .region_name = "surface",
        .required_region_kind = CouplingRegionKind::Boundary,
        .requirement = CouplingRequirement::Optional,
    });
    declaration.shared_regions.push_back({
        .shared_region_name = "interface",
        .required_region_kind = CouplingRegionKind::Boundary,
        .requirement = CouplingRequirement::Optional,
    });

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> optional_declarations{declaration};
    EXPECT_TRUE(graph.buildDeclarationGraph(
                         CouplingContext{},
                         std::span<const CouplingContractDeclaration>(optional_declarations))
                    .ok());

    declaration.participants[0].requirement = CouplingRequirement::Required;
    declaration.fields[0].requirement = CouplingRequirement::Required;
    declaration.regions[0].requirement = CouplingRequirement::Required;
    declaration.shared_regions[0].requirement = CouplingRequirement::Required;

    CouplingGraph required_graph;
    const std::array<CouplingContractDeclaration, 1> required_declarations{declaration};
    const auto validation = required_graph.buildDeclarationGraph(
        CouplingContext{},
        std::span<const CouplingContractDeclaration>(required_declarations));
    EXPECT_FALSE(validation.ok());
    const auto diagnostics = formatDiagnostics(validation);
    EXPECT_NE(diagnostics.find("required coupling participant is missing"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("required coupling field is missing"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("required coupling region is missing"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("required shared region is missing"),
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
            .kind = CouplingNonFieldDependencyRequirementKind::MaterialStateWork,
            .participant_name = "left",
            .name = "work",
            .material_state_byte_offset = 24,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::BoundaryFunctional,
            .participant_name = "left",
            .name = "traction",
            .require_analysis_variable_key = true,
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
    ASSERT_EQ(declaration.non_field_dependencies.size(), 9u);
    EXPECT_TRUE(declaration.dependencies.empty());
    EXPECT_FALSE(declaration.non_field_dependencies[0].require_analysis_variable_key);
    EXPECT_FALSE(declaration.non_field_dependencies[1].require_analysis_variable_key);
    EXPECT_FALSE(declaration.non_field_dependencies[2].require_analysis_variable_key);
    EXPECT_FALSE(declaration.non_field_dependencies[3].require_analysis_variable_key);
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

TEST(CouplingContractValidation, SeparatesNonFieldRequirementsFromProvenance)
{
    auto declaration = minimalDeclaration();
    declaration.non_field_dependencies = {
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::Parameter,
            .participant_name = "left",
            .name = "penalty",
            .region = CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
                .shared_region_name = "interface",
            },
            .required_region_kind = CouplingRegionKind::Boundary,
            .expected_parameter_value_type = svmp::FE::params::ValueType::Real,
            .expected_value_type = "scalar",
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::MaterialStateWork,
            .participant_name = "left",
            .name = "history",
            .expected_value_type = "tensor",
            .material_state_byte_offset = 32,
        },
    };

    const auto validation = validateContractDeclarationShape(declaration);
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
    ASSERT_EQ(declaration.non_field_dependencies.size(), 2u);
    EXPECT_EQ(declaration.non_field_dependencies[0].participant_name, "left");
    EXPECT_EQ(declaration.non_field_dependencies[0].name, "penalty");
    ASSERT_TRUE(declaration.non_field_dependencies[0].region.has_value());
    EXPECT_EQ(declaration.non_field_dependencies[0].region->region_name, "surface");
    ASSERT_TRUE(
        declaration.non_field_dependencies[0].region->shared_region_name.has_value());
    EXPECT_EQ(*declaration.non_field_dependencies[0].region->shared_region_name,
              "interface");
    ASSERT_TRUE(
        declaration.non_field_dependencies[0].expected_parameter_value_type.has_value());
    EXPECT_EQ(*declaration.non_field_dependencies[0].expected_parameter_value_type,
              svmp::FE::params::ValueType::Real);
    ASSERT_TRUE(
        declaration.non_field_dependencies[1].material_state_byte_offset.has_value());
    EXPECT_EQ(*declaration.non_field_dependencies[1].material_state_byte_offset, 32u);

    CouplingFormAnalysisMetadata metadata;
    metadata.non_field_dependencies.push_back(
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::Parameter,
            .participant_name = "left",
            .system_name = "left_system",
            .name = "penalty",
            .domain = svmp::FE::analysis::DomainKind::Boundary,
            .region_name = "surface",
            .shared_region_name = "interface",
            .marker = 17,
            .side = CouplingInterfaceSide::Minus,
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
            .logical_region = svmp::search::LogicalInterfaceRegionId{
                .persistent_id = "left/surface",
                .physical_label = 17,
            },
#endif
            .slot = 5,
            .provider = "forms",
            .value_type = "scalar",
            .parameter_value_type = svmp::FE::params::ValueType::Real,
        });
    metadata.non_field_dependencies.push_back(
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::MaterialStateWork,
            .participant_name = "left",
            .system_name = "left_system",
            .name = "history",
            .byte_offset = 32,
            .provider = "forms",
            .value_type = "tensor",
        });
    metadata.non_field_dependencies.push_back(
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::AuxiliaryOutput,
            .participant_name = "left",
            .system_name = "left_system",
            .name = "traction_out",
            .output_id = 9,
            .provider = "forms",
            .value_type = "scalar",
        });

    ASSERT_EQ(metadata.non_field_dependencies.size(), 3u);
    EXPECT_EQ(metadata.non_field_dependencies[0].system_name, "left_system");
    EXPECT_EQ(metadata.non_field_dependencies[0].marker, 17);
    EXPECT_EQ(metadata.non_field_dependencies[0].side, CouplingInterfaceSide::Minus);
    ASSERT_TRUE(metadata.non_field_dependencies[0].slot.has_value());
    EXPECT_EQ(*metadata.non_field_dependencies[0].slot, 5u);
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    ASSERT_TRUE(metadata.non_field_dependencies[0].logical_region.has_value());
    EXPECT_EQ(metadata.non_field_dependencies[0].logical_region->physical_label, 17);
#endif
    ASSERT_TRUE(metadata.non_field_dependencies[1].byte_offset.has_value());
    EXPECT_EQ(*metadata.non_field_dependencies[1].byte_offset, 32u);
    ASSERT_TRUE(metadata.non_field_dependencies[2].output_id.has_value());
    EXPECT_EQ(*metadata.non_field_dependencies[2].output_id, 9u);
}

TEST(CouplingContractValidation, MapsBoundaryIntegralToBoundaryFunctionalIdentity)
{
    const auto declaration_kind = analysisVariableKindForNonFieldRequirement(
        CouplingNonFieldDependencyRequirementKind::BoundaryIntegral);
    ASSERT_TRUE(declaration_kind.has_value());
    EXPECT_EQ(*declaration_kind,
              svmp::FE::analysis::VariableKind::BoundaryFunctional);

    const auto boundary_functional_kind = analysisVariableKindForNonFieldRequirement(
        CouplingNonFieldDependencyRequirementKind::BoundaryFunctional);
    ASSERT_TRUE(boundary_functional_kind.has_value());
    EXPECT_EQ(*boundary_functional_kind,
              svmp::FE::analysis::VariableKind::BoundaryFunctional);

    EXPECT_FALSE(analysisVariableKindForNonFieldRequirement(
                     CouplingNonFieldDependencyRequirementKind::Parameter)
                     .has_value());

    const auto provenance_kind = analysisVariableKindForFormNonFieldDependency(
        CouplingFormNonFieldDependencyKind::BoundaryIntegral);
    ASSERT_TRUE(provenance_kind.has_value());
    EXPECT_EQ(*provenance_kind,
              svmp::FE::analysis::VariableKind::BoundaryFunctional);

    CouplingFormAnalysisMetadata metadata;
    metadata.non_field_dependencies.push_back(
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::BoundaryIntegral,
            .participant_name = "left",
            .system_name = "left_system",
            .name = "traction_integral",
            .domain = svmp::FE::analysis::DomainKind::Boundary,
            .provider = "forms",
            .value_type = "scalar",
        });
    metadata.variable_dependencies.push_back(CouplingFormVariableDependencyProvenance{
        .residual_row = svmp::FE::analysis::VariableKey::field(1),
        .dependency = svmp::FE::analysis::VariableKey::named(
            *provenance_kind,
            "left_system/traction_integral"),
        .domain = svmp::FE::analysis::DomainKind::Boundary,
        .provider = "forms",
    });

    ASSERT_EQ(metadata.non_field_dependencies.size(), 1u);
    EXPECT_EQ(metadata.non_field_dependencies[0].kind,
              CouplingFormNonFieldDependencyKind::BoundaryIntegral);
    ASSERT_EQ(metadata.variable_dependencies.size(), 1u);
    EXPECT_EQ(metadata.variable_dependencies[0].dependency.kind,
              svmp::FE::analysis::VariableKind::BoundaryFunctional);
    EXPECT_EQ(metadata.variable_dependencies[0].dependency.name,
              "left_system/traction_integral");
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

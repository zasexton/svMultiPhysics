#include "Coupling/MonolithicCouplingBuilder.h"

#include "Analysis/ProblemAnalysisTypes.h"
#include "Coupling/CouplingGraph.h"
#include "Coupling/CouplingFormBuilder.h"
#include "Core/FEException.h"
#include "Forms/Vocabulary.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

constexpr int kInterfaceMarker = 17;

struct BuilderFixture {
    std::shared_ptr<spaces::H1Space> space;
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh;
    systems::FESystem system;
    FieldId left_field{INVALID_FIELD_ID};
    FieldId right_field{INVALID_FIELD_ID};
    CouplingContext context;

    BuilderFixture()
        : space(std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1))
        , mesh(std::make_shared<forms::test::SingleTetraMeshAccess>())
        , system(mesh)
    {
        left_field = system.addField(systems::FieldSpec{
            .name = "left_primary",
            .space = space,
            .components = 1,
        });
        right_field = system.addField(systems::FieldSpec{
            .name = "right_primary",
            .space = space,
            .components = 1,
        });

        CouplingContextBuilder builder;
        builder.addParticipant({
            .participant_name = "left",
            .system_name = "shared_system",
            .system = &system,
        });
        builder.addParticipant({
            .participant_name = "right",
            .system_name = "shared_system",
            .system = &system,
        });
        builder.addField({
            .participant_name = "left",
            .system_name = "shared_system",
            .system = &system,
            .field_name = "primary",
            .field_id = left_field,
            .space = space,
            .components = 1,
        });
        builder.addField({
            .participant_name = "right",
            .system_name = "shared_system",
            .system = &system,
            .field_name = "primary",
            .field_id = right_field,
            .space = space,
            .components = 1,
        });
        context = builder.build();
    }
};

CouplingContext interfaceContext(BuilderFixture& fixture, int marker)
{
    CouplingRegionRef left_region{
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = marker,
        .side = CouplingInterfaceSide::Minus,
    };
    CouplingRegionRef right_region{
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = marker,
        .side = CouplingInterfaceSide::Plus,
    };

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
    });
    builder.addField({
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
        .field_name = "primary",
        .field_id = fixture.left_field,
        .space = fixture.space,
        .components = 1,
    });
    builder.addField({
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
        .field_name = "primary",
        .field_id = fixture.right_field,
        .space = fixture.space,
        .components = 1,
    });
    builder.addRegion(left_region);
    builder.addRegion(right_region);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {left_region, right_region},
    });
    return builder.build();
}

const CouplingInstalledDependency* findDependency(
    const CouplingFormAnalysisMetadata& metadata,
    analysis::VariableKey row,
    analysis::VariableKey dependency)
{
    const auto it = std::find_if(
        metadata.installed_dependencies.begin(),
        metadata.installed_dependencies.end(),
        [&](const CouplingInstalledDependency& installed) {
            return installed.residual_row == row &&
                   installed.dependency == dependency;
        });
    return it == metadata.installed_dependencies.end() ? nullptr : &*it;
}

class GenericTwoParticipantContract final : public CouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "generic";
    }

    [[nodiscard]] CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = name();
        declaration.contract_name = "generic_instance";
        declaration.participants.push_back({.participant_name = "left"});
        declaration.participants.push_back({.participant_name = "right"});
        declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
        declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
        declaration.dependencies.push_back(CouplingResidualDependency{
            .residual_row = {
                .kind = CouplingVariableKind::Field,
                .participant_name = "right",
                .name = "primary",
            },
            .dependency = {
                .kind = CouplingVariableKind::Field,
                .participant_name = "left",
                .name = "primary",
            },
        });
        declaration.expected_blocks.push_back(CouplingBlockExpectation{
            .residual_row = declaration.dependencies.back().residual_row,
            .dependency = declaration.dependencies.back().dependency,
        });
        return declaration;
    }

    [[nodiscard]] std::vector<CouplingFormContribution> buildMonolithicForms(
        const CouplingContext&,
        const CouplingFormBuilder& forms) const override
    {
        CouplingFormContribution contribution;
        contribution.contribution_name = "generic_cell_coupling";
        contribution.origin = "GenericTwoParticipantContract";
        contribution.operator_name = "equations";
        contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
        contribution.extra_trial_field_uses = {{
            .participant_name = "left",
            .field_name = "primary",
        }};
        contribution.residual =
            (forms.state("left", "primary", "a") *
             forms.test("right", "primary", "w")).dx();
        return {contribution};
    }
};

class GenericInterfaceContract final : public CouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "generic_interface";
    }

    [[nodiscard]] CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = name();
        declaration.contract_name = "generic_interface_instance";
        declaration.participants.push_back({.participant_name = "left"});
        declaration.participants.push_back({.participant_name = "right"});
        declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
        declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
        declaration.shared_regions.push_back({
            .shared_region_name = "interface",
            .required_region_kind = CouplingRegionKind::InterfaceFace,
        });
        declaration.dependencies.push_back(CouplingResidualDependency{
            .residual_row = {
                .kind = CouplingVariableKind::Field,
                .participant_name = "right",
                .name = "primary",
            },
            .dependency = {
                .kind = CouplingVariableKind::Field,
                .participant_name = "left",
                .name = "primary",
            },
        });
        declaration.expected_blocks.push_back(CouplingBlockExpectation{
            .residual_row = declaration.dependencies.back().residual_row,
            .dependency = declaration.dependencies.back().dependency,
        });
        return declaration;
    }

    [[nodiscard]] std::vector<CouplingFormContribution> buildMonolithicForms(
        const CouplingContext& context,
        const CouplingFormBuilder& forms) const override
    {
        const auto marker = context.sharedRegion("interface", "right").marker;
        CouplingFormContribution contribution;
        contribution.contribution_name = "generic_interface_coupling";
        contribution.origin = "GenericInterfaceContract";
        contribution.operator_name = "equations";
        contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
        contribution.extra_trial_field_uses = {{
            .participant_name = "left",
            .field_name = "primary",
        }};
        contribution.residual =
            (forms.state("left", "primary", "a") *
             forms.test("right", "primary", "w")).dI(marker);
        return {contribution};
    }
};

} // namespace

TEST(MonolithicCouplingBuilder, ResolvesFormContributionThroughContext)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "generic_cell_coupling";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{.participant_name = "left", .field_name = "primary"}};
    contribution.residual =
        (forms.state("left", "primary", "a") *
         forms.test("right", "primary", "w")).dx();

    const auto resolved = builder.resolveFormContribution(fixture.context, contribution);
    ASSERT_EQ(resolved.fields.size(), 1u);
    ASSERT_EQ(resolved.extra_trial_fields.size(), 1u);
    EXPECT_EQ(resolved.fields[0], fixture.right_field);
    EXPECT_EQ(resolved.extra_trial_fields[0], fixture.left_field);
    ASSERT_EQ(resolved.install_options.extra_trial_fields.size(), 1u);
    EXPECT_EQ(resolved.install_options.extra_trial_fields[0], fixture.left_field);
    EXPECT_EQ(resolved.system_name, "shared_system");
}

TEST(MonolithicCouplingBuilder, RejectsOverlappingPrimaryAndExtraTrialFields)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "bad_overlap";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.residual =
        (forms.state("right", "primary", "u") *
         forms.test("right", "primary", "w")).dx();

    EXPECT_THROW(static_cast<void>(builder.resolveFormContribution(fixture.context, contribution)),
                 InvalidArgumentException);
}

TEST(MonolithicCouplingBuilder, InstallsResolvedFormAndAdaptsBridgeMetadata)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "generic_cell_coupling";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.operator_name = "equations";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{.participant_name = "left", .field_name = "primary"}};
    contribution.residual =
        (forms.state("left", "primary", "a") *
         forms.test("right", "primary", "w")).dx();

    const auto resolved = builder.resolveFormContribution(fixture.context, contribution);
    const auto metadata =
        builder.installResolvedFormContribution(fixture.system, resolved);

    EXPECT_EQ(metadata.contribution_name, "generic_cell_coupling");
    EXPECT_EQ(metadata.origin, "MonolithicCouplingBuilderTest");
    EXPECT_EQ(metadata.system_name, "shared_system");
    EXPECT_NE(std::find(metadata.installed_fields.begin(),
                        metadata.installed_fields.end(),
                        fixture.right_field),
              metadata.installed_fields.end());
    EXPECT_NE(std::find(metadata.installed_fields.begin(),
                        metadata.installed_fields.end(),
                        fixture.left_field),
              metadata.installed_fields.end());

    const auto* dependency = findDependency(
        metadata,
        analysis::VariableKey::field(fixture.right_field),
        analysis::VariableKey::field(fixture.left_field));
    ASSERT_NE(dependency, nullptr);
    EXPECT_EQ(dependency->domain, analysis::DomainKind::Cell);
    EXPECT_TRUE(dependency->contributes_matrix_block);
    EXPECT_TRUE(dependency->contributes_vector);

    const auto block_it = std::find_if(
        metadata.installed_blocks.begin(),
        metadata.installed_blocks.end(),
        [&](const CouplingInstalledBlockProvenance& block) {
            return block.residual_row == analysis::VariableKey::field(fixture.right_field) &&
                   block.dependency == analysis::VariableKey::field(fixture.left_field);
        });
    ASSERT_NE(block_it, metadata.installed_blocks.end());
    EXPECT_TRUE(block_it->has_matrix);
    EXPECT_TRUE(block_it->has_vector);
}

TEST(MonolithicCouplingBuilder, RegistersContractOwnedInterfaceAdditionalFields)
{
    BuilderFixture fixture;
    fixture.system.setInterfaceMesh(kInterfaceMarker,
                                    std::make_shared<const svmp::InterfaceMesh>());
    const auto context = interfaceContext(fixture, kInterfaceMarker);
    const MonolithicCouplingBuilder builder;

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
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "lambda",
        .space = fixture.space,
        .components = 1,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
        .shared_region_name = "interface",
    });

    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto validation = builder.validateDeclarations(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto resolved = builder.registerAdditionalFields(
        context,
        std::span<const CouplingContractDeclaration>(declarations));

    ASSERT_EQ(resolved.size(), 1u);
    EXPECT_EQ(resolved[0].system_name, "shared_system");
    EXPECT_EQ(resolved[0].field_spec.name, "generic_instance.lambda");
    EXPECT_EQ(resolved[0].field_spec.scope, systems::FieldScope::InterfaceFace);
    EXPECT_EQ(resolved[0].field_spec.interface_marker, kInterfaceMarker);
    EXPECT_TRUE(fixture.system.hasField("generic_instance.lambda"));
}

TEST(MonolithicCouplingBuilder, GenericContractInstallsAndFinalizesGraph)
{
    BuilderFixture fixture;
    const GenericTwoParticipantContract contract;
    const MonolithicCouplingBuilder builder;
    const CouplingFormBuilder forms(fixture.context);

    const std::array<CouplingContractDeclaration, 1> declarations{contract.declare()};
    const auto declaration_validation = builder.validateDeclarations(
        fixture.context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(declaration_validation.ok()) << formatDiagnostics(declaration_validation);

    const auto contributions = contract.buildMonolithicForms(fixture.context, forms);
    const auto installed = builder.installFormContributions(
        fixture.system,
        fixture.context,
        std::span<const CouplingFormContribution>(contributions));

    CouplingGraph graph;
    const auto finalized_validation = graph.buildFinalizedGraph(
        fixture.context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed));

    EXPECT_TRUE(finalized_validation.ok()) << formatDiagnostics(finalized_validation);
    ASSERT_EQ(installed.size(), 1u);
    EXPECT_NE(findDependency(installed[0],
                             analysis::VariableKey::field(fixture.right_field),
                             analysis::VariableKey::field(fixture.left_field)),
              nullptr);
}

TEST(MonolithicCouplingBuilder, GenericInterfaceContractInstallsAndFinalizesGraph)
{
    BuilderFixture fixture;
    fixture.system.setInterfaceMesh(kInterfaceMarker,
                                    std::make_shared<const svmp::InterfaceMesh>());
    const auto context = interfaceContext(fixture, kInterfaceMarker);
    const GenericInterfaceContract contract;
    const MonolithicCouplingBuilder builder;
    const CouplingFormBuilder forms(context);

    const std::array<CouplingContractDeclaration, 1> declarations{contract.declare()};
    const auto declaration_validation = builder.validateDeclarations(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(declaration_validation.ok()) << formatDiagnostics(declaration_validation);

    const auto contributions = contract.buildMonolithicForms(context, forms);
    const auto installed = builder.installFormContributions(
        fixture.system,
        context,
        std::span<const CouplingFormContribution>(contributions));

    CouplingGraph graph;
    const auto finalized_validation = graph.buildFinalizedGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed));

    EXPECT_TRUE(finalized_validation.ok()) << formatDiagnostics(finalized_validation);
    ASSERT_EQ(installed.size(), 1u);
    const auto* dependency = findDependency(
        installed[0],
        analysis::VariableKey::field(fixture.right_field),
        analysis::VariableKey::field(fixture.left_field));
    ASSERT_NE(dependency, nullptr);
    EXPECT_EQ(dependency->domain, analysis::DomainKind::InterfaceFace);
}

TEST(MonolithicCouplingBuilder, RejectsInterfaceContractWithoutRegisteredTopology)
{
    BuilderFixture fixture;
    const auto context = interfaceContext(fixture, kInterfaceMarker);
    const GenericInterfaceContract contract;
    const MonolithicCouplingBuilder builder;

    const std::array<CouplingContractDeclaration, 1> declarations{contract.declare()};
    const auto validation = builder.validateDeclarations(
        context,
        std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "interface-face coupling region is missing registered interface topology"),
              std::string::npos);
}
